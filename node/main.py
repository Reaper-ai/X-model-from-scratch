"""
Main orchestration entry point for Neural ODE experiments (Damped Pendulum).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import DiscreteMLP, NeuralODE, ClassicalIntegrator, count_parameters
from data_env import (
    generate_trajectories, save_trajectories, load_trajectories,
    create_dataloaders, get_split_masks, make_irregular_view, pendulum_system
)
from train import DiscreteTrainer, EarlyStopping, TrajectoryTrainer
from evaluate import ClassicalEvaluator, evaluate_all_tests, evaluate_queries_on_loader
from viz import save_all_figures

# STEP 1: Device Setup
def setup_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# STEP 2: Model Setup
def setup_models(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Instantiate all models.

    Both learned models use the *same* MLP trunk (2 -> 64 -> 64 -> 64 -> 64 -> 2)
    and are seeded identically, so the comparison isolates the time
    parameterization (discrete map vs continuous vector field), not capacity.
    """
    seed = config.get('seed', 42)

    # Discrete map parameterization. 'residual' (default) = x_{k+1} = x + MLP(x)
    # with an identity initialisation; 'direct' = x_{k+1} = MLP(x). The direct
    # form is numerically unstable to roll out (a random map compounds), so the
    # delta form is used unless explicitly overridden.
    mlp_residual = config.get('mlp_form', 'residual') != 'direct'

    torch.manual_seed(seed)
    discrete_mlp = DiscreteMLP(
        input_dim=2,
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        output_dim=2,
        dt=config.get('dt', 0.05),
        residual=mlp_residual
    ).to(device)

    torch.manual_seed(seed)
    node = NeuralODE(
        input_dim=2,
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        output_dim=2
    ).to(device)

    classical = {
        'rk45': ClassicalIntegrator(method='RK45'),
        'dop853': ClassicalIntegrator(method='DOP853'),
        'euler': ClassicalIntegrator(method='Euler'),
        'rk4': ClassicalIntegrator(method='RK4')
    }

    print(f"Discrete MLP parameters: {count_parameters(discrete_mlp):,}")
    print(f"Neural ODE parameters:   {count_parameters(node):,}")

    return {
        'discrete_mlp': discrete_mlp,
        'node': node,
        'classical': classical
    }

# STEP 3: Optimizer Setup
def setup_optimizers(models: Dict[str, nn.Module], config: Dict[str, Any]) -> Dict[str, Any]:
    """Create optimizers and schedulers."""
    optimizers = {
        'discrete_mlp': Adam(models['discrete_mlp'].parameters(), lr=config['lr'], weight_decay=config['weight_decay']),
        'node': Adam(models['node'].parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    }
    schedulers = {
        'discrete_mlp': CosineAnnealingLR(optimizers['discrete_mlp'], T_max=config['epochs'], eta_min=config['lr'] * 0.01),
        'node': CosineAnnealingLR(optimizers['node'], T_max=config['epochs'], eta_min=config['lr'] * 0.01)
    }
    return {'optimizers': optimizers, 'schedulers': schedulers}

# STEP 4: Data Generation/Loading
def run_data_generation(config: Dict[str, Any], data_dir: Path) -> tuple:
    """Generate or load pendulum trajectories with 4-way labels."""
    data_path = data_dir / 'pendulum_trajectories.npz'

    if data_path.exists() and not config.get('regenerate_data', False):
        print(f"Loading existing data from {data_path}")
        t_eval, trajectories, traj_lengths, ic_labels = load_trajectories(str(data_path))
    else:
        print("Generating pendulum trajectories...")
        t_eval, trajectories, traj_lengths, ic_labels = generate_trajectories(
            n_trajectories=config['n_trajectories'],
            k_periods_train=config.get('k_periods_train', 3),
            test_B_max_periods=config.get('test_B_max_periods', 8),
            test_C_ic_range_factor=config.get('test_C_ic_range_factor', 1.5),
            dt=config['dt'],
            noise_std=config.get('noise_std', 0.0),
            seed=config.get('seed', 42),
            g=9.81, L=1.0, b=0.1,
            symmetric_augment=config.get('symmetric_augment', True)
        )
        save_trajectories(t_eval, trajectories, traj_lengths, ic_labels, str(data_path))
        print(f"Saved {len(trajectories)} trajectories to {data_path}")

    return t_eval, trajectories, traj_lengths, ic_labels

# STEP 5: Discrete MLP Training
def run_discrete_mlp_training(
    models: Dict[str, nn.Module],
    train_loader, val_loader,
    optimizers: Dict[str, Any],
    schedulers: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
    model_dir: Path
) -> Dict[str, Any]:
    """Train the discrete one-step flow map on the regular-sampled data."""
    trainer = DiscreteTrainer(models['discrete_mlp'], train_loader, val_loader, optimizers['discrete_mlp'],
                              device, scheduler=schedulers['discrete_mlp'],
                              window_size=config.get('window_size', 50),
                              n_windows_per_batch=config.get('n_windows_per_batch', 64),
                              seed=config.get('seed', 42))
    early_stopping = EarlyStopping(min_delta=1e-3, relative=False)
    history = trainer.train(config['epochs'], early_stopping)
    torch.save(models['discrete_mlp'].state_dict(), model_dir / 'discrete_mlp_model.pt')
    return {'history': history, 'model': models['discrete_mlp']}

# STEP 6: Neural ODE Training
def run_node_training(
    models: Dict[str, nn.Module],
    train_loader, val_loader,
    optimizers: Dict[str, Any],
    schedulers: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
    model_dir: Path,
    time_scale: float = 1.0
) -> Dict[str, Any]:
    """Train the Neural ODE on the *irregularly* sampled data.

    The ODE is supervised by observed states only (no analytic derivative
    loss by default), matching the information the discrete MLP gets.
    `time_scale` keeps the window's time span equal to the regular view's.
    """
    trainer = TrajectoryTrainer(models['node'], train_loader, val_loader, optimizers['node'],
                                device, scheduler=schedulers['node'],
                                window_size=config.get('window_size', 50),
                                n_windows_per_batch=config.get('n_windows_per_batch', 64),
                                time_scale=time_scale, seed=config.get('seed', 42))
    early_stopping = EarlyStopping(min_delta=1e-3, relative=False)
    history = trainer.train(config['epochs'], early_stopping)
    torch.save(models['node'].state_dict(), model_dir / 'node_model.pt')
    return {'history': history, 'model': models['node']}

# STEP 7: Evaluation
def run_evaluation(
    models: Dict[str, Any],
    reg_loaders,
    irr_loaders,
    t_eval: np.ndarray,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate all methods under two protocols.

    Protocol R (regular grid): both models are queried on the dense regular
        grid the discrete MLP was trained on. The Neural ODE integrates at
        those same times. This is the "dense regular sampling" regime where
        the two hypothesis classes are expected to be on par.

    Protocol I (irregular queries): both models are queried on the SAME
        irregular timestamps. The Neural ODE integrates there natively; the
        discrete map can only be stepped at its own dt and is linearly
        interpolated to the query times (its structural handicap).
    """
    results = {}

    for name in ['discrete_mlp', 'node']:
        is_node = (name == 'node')
        print(f"[{name}] Protocol R (regular grid):")
        res = evaluate_all_tests(models[name], *reg_loaders[2:], t_eval,
                                 is_neural_ode=is_node, is_discrete=not is_node, device=device)

        print(f"[{name}] Protocol I (irregular query times):")
        res['irregular_queries'] = {
            split: evaluate_queries_on_loader(models[name], loader, is_discrete=not is_node, device=device)
            for split, loader in zip(['test_A', 'test_B', 'test_C'], irr_loaders[2:])
        }
        results[name] = res

    # Classical integrators (only on test_A for speed)
    from data_env import pendulum_system
    classical_results = {}
    for cname, integrator in models['classical'].items():
        evaluator = ClassicalEvaluator(integrator, device)
        # Evaluate on test_A only
        all_metrics = []
        for batch in reg_loaders[2]:
            x = batch['x'].to(device)
            lengths = batch['lengths']
            for i in range(x.shape[0]):
                L_i = lengths[i].item()
                true_traj = x[i, :L_i]
                t_i = torch.arange(L_i, device=device) * (t_eval[1] - t_eval[0])
                m = evaluator.evaluate(
                    true_traj.unsqueeze(0), t_i, pendulum_system
                )
                all_metrics.append({k: v[0] for k, v in m.items()})
        classical_results[f'classical_{cname}'] = {
            k: np.array([m[k] for m in all_metrics]) for k in all_metrics[0].keys()
        }

    results['classical'] = classical_results
    return results

# STEP 8: Visualization
def run_visualization(
    results: Dict[str, Any],
    true_trajectories: torch.Tensor,
    t_eval: torch.Tensor,
    config: Dict[str, Any],
    images_dir: Path,
    lengths: Optional[np.ndarray] = None,
    ts_trajectories: Optional[torch.Tensor] = None,
    ts_lengths: Optional[np.ndarray] = None,
    node_query_times: Optional[List[np.ndarray]] = None
) -> None:
    """Generate all visualization figures."""
    save_all_figures(results, true_trajectories, t_eval, config, images_dir,
                     lengths=lengths, ts_trajectories=ts_trajectories, ts_lengths=ts_lengths,
                     node_query_times=node_query_times)

# STEP 9: Full Pipeline
def run_full_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete pipeline: data -> train -> evaluate -> visualize."""
    device = setup_device()
    print(f"Using device: {device}")

    root_dir = Path(__file__).parent
    data_dir = root_dir / 'data'
    model_dir = root_dir / 'data'
    images_dir = root_dir.parent / 'docs' / 'images' / 'node'

    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data generation (variable-length, 4-way labels)
    t_eval, trajectories, traj_lengths, ic_labels = run_data_generation(config, data_dir)

    # 2. Two views of the SAME trajectories (identical split membership):
    #    - regular grid  -> discrete MLP (a map can only be stepped at a fixed dt)
    #    - irregular     -> Neural ODE   (a field can be integrated anywhere)
    split_masks = get_split_masks(ic_labels, val_fraction=0.1)

    reg_loaders = create_dataloaders(
        trajectories, t_eval, traj_lengths, ic_labels,
        batch_size=config['batch_size'],
        compute_deriv=False, split_masks=split_masks
    )

    irr_times, irr_trajs, irr_lengths = make_irregular_view(
        trajectories, t_eval, traj_lengths,
        frac=config.get('irregular_frac', 0.5),
        strategy=config.get('irregular_strategy', 'random'),
        seed=config.get('seed', 42)
    )
    irr_loaders = create_dataloaders(
        irr_trajs, t_eval, irr_lengths, ic_labels,
        batch_size=config['batch_size'],
        compute_deriv=False, split_masks=split_masks, times=irr_times
    )
    # Mean sampling gap of the irregular view, relative to dt. This keeps the
    # *time span* of a training window the same for both models (the irregular
    # view simply contains fewer observation points over the same horizon).
    dt_data = float(t_eval[1] - t_eval[0])
    time_scale = float(np.mean(np.concatenate([np.diff(t) for t in irr_times]))) / dt_data
    print(f"Irregular view: mean sampling gap = {time_scale:.2f} x dt "
          f"(frac={config.get('irregular_frac', 0.5)})")

    # 3. Setup models and optimizers
    models = setup_models(config, device)
    optim = setup_optimizers(models, config)
    optimizers = optim['optimizers']
    schedulers = optim['schedulers']

    # 4. Train — discrete MLP on regular data, Neural ODE on irregular data
    discrete_mlp_results = run_discrete_mlp_training(
        models, reg_loaders[0], reg_loaders[1], optimizers, schedulers, config, device, model_dir
    )
    node_results = run_node_training(
        models, irr_loaders[0], irr_loaders[1], optimizers, schedulers, config, device, model_dir,
        time_scale=time_scale
    )

    # 5. Evaluate on all 3 test types, under both protocols
    eval_results = run_evaluation(
        models, reg_loaders, irr_loaders, t_eval, config, device
    )

    # Combine results
    all_results = {
        'discrete_mlp': {**discrete_mlp_results, **eval_results.get('discrete_mlp', {})},
        'node': {**node_results, **eval_results.get('node', {})},
        'classical': eval_results.get('classical', {})
    }

    # 6. Save metrics
    metrics_path = data_dir / 'metrics.json'

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items() if k not in ('model', 'trajectories')}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, nn.Module):
            return f'<{type(obj).__name__}>'
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    serializable = make_serializable(all_results)

    with open(metrics_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # 7. Visualize
    # Phase portrait + error growth use test_A (in-distribution);
    # timeseries uses test_B (time extrapolation) with train-horizon split;
    # ic_sensitivity sweeps test_C IC bounds internally.
    period = 2.0 * np.pi * np.sqrt(1.0 / 9.81)
    train_horizon_steps = int(round(config.get('k_periods_train', 3) * period / config.get('dt', 0.05)))

    sample_trajs = []
    for batch in reg_loaders[2]:
        x = batch['x']
        lengths = batch['lengths']
        for i in range(x.shape[0]):
            L_i = lengths[i].item()
            sample_trajs.append(x[i, :L_i])
        if len(sample_trajs) >= 10:
            break

    # Test B samples: collect all longer than train horizon, keep the LONGEST
    # ones so the timeseries plot shows the full extrapolation (up to ~16 s).
    # The irregular loader holds the SAME trajectories in the SAME order, so we
    # can pick up the timestamps the Neural ODE actually was trained on.
    def _collect_test_b():
        out = []
        for reg_batch, irr_batch in zip(reg_loaders[3], irr_loaders[3]):
            x = reg_batch['x']
            lengths = reg_batch['lengths']
            tq = irr_batch['t']
            lq = irr_batch['lengths']
            for i in range(x.shape[0]):
                L_i = lengths[i].item()
                out.append((x[i, :L_i], tq[i, :lq[i].item()].numpy()))
        return out

    pairs = _collect_test_b()
    long_pairs = [p for p in pairs if p[0].shape[0] > train_horizon_steps]
    if not long_pairs:  # fallback: any test_B traj
        long_pairs = pairs
    long_pairs.sort(key=lambda p: p[0].shape[0], reverse=True)
    long_pairs = long_pairs[:10]
    ts_trajs = [p[0] for p in long_pairs]
    ts_query_times = [p[1] for p in long_pairs]

    def _pad(traj_list):
        if not traj_list:
            return None, None
        lens = np.array([t.shape[0] for t in traj_list])
        max_L = max(t.shape[0] for t in traj_list)
        padded = torch.zeros(len(traj_list), max_L, 2)
        for i, t in enumerate(traj_list):
            padded[i, :t.shape[0]] = t
        return padded, lens

    padded, sample_lengths = _pad(sample_trajs)
    ts_padded, ts_lengths = _pad(ts_trajs)

    if padded is not None:
        t_eval_vis = torch.from_numpy(t_eval[:padded.shape[1]]).float()
        # For timeseries, extend t_eval to cover test_B sample if longer
        if ts_padded is not None and ts_padded.shape[1] > padded.shape[1]:
            t_eval_vis = torch.from_numpy(t_eval[:ts_padded.shape[1]]).float()
        run_visualization(all_results, padded, t_eval_vis, config, images_dir,
                          lengths=sample_lengths,
                          ts_trajectories=ts_padded, ts_lengths=ts_lengths,
                          node_query_times=ts_query_times)

    return all_results

# STEP 9b: Visualization-Only (load saved models + data, no training)
def run_visualize_only(config: Dict[str, Any]) -> None:
    """Regenerate all 5 figures from saved models and data (no training)."""
    root_dir = Path(__file__).parent
    data_dir = root_dir / 'data'
    images_dir = root_dir.parent / 'docs' / 'images' / 'node'
    images_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / 'pendulum_trajectories.npz'
    if not data_path.exists():
        raise FileNotFoundError(f"No data at {data_path}. Run 'generate-data' first.")
    for m in ['discrete_mlp_model.pt', 'node_model.pt']:
        if not (data_dir / m).exists():
            raise FileNotFoundError(f"No model at {data_dir / m}. Train first.")

    print(f"Loading data from {data_path}")
    t_eval, trajectories, traj_lengths, ic_labels = load_trajectories(str(data_path))

    split_masks = get_split_masks(ic_labels, val_fraction=0.1)

    reg_loaders = create_dataloaders(
        trajectories, t_eval, traj_lengths, ic_labels,
        batch_size=config['batch_size'],
        compute_deriv=False, split_masks=split_masks
    )

    # Irregular view — only needed to show which timestamps the NODE saw
    irr_times, irr_trajs, irr_lengths = make_irregular_view(
        trajectories, t_eval, traj_lengths,
        frac=config.get('irregular_frac', 0.5),
        strategy=config.get('irregular_strategy', 'random'),
        seed=config.get('seed', 42)
    )
    irr_loaders = create_dataloaders(
        irr_trajs, t_eval, irr_lengths, ic_labels,
        batch_size=config['batch_size'],
        compute_deriv=False, split_masks=split_masks, times=irr_times
    )

    # Empty results — save_all_figures regenerates all model trajectories
    all_results: Dict[str, Any] = {'discrete_mlp': {}, 'node': {}, 'classical': {}}

    period = 2.0 * np.pi * np.sqrt(1.0 / 9.81)
    train_horizon_steps = int(round(config.get('k_periods_train', 3) * period / config.get('dt', 0.05)))

    # test_A sample (phase portrait + error growth)
    sample_trajs = []
    for batch in reg_loaders[2]:
        x = batch['x']
        lens = batch['lengths']
        for i in range(x.shape[0]):
            sample_trajs.append(x[i, :lens[i].item()])
        if len(sample_trajs) >= 10:
            break

    # test_B sample (timeseries): all longer than train horizon, longest first
    # so the plot shows the full extrapolation (up to ~16 s)
    def _collect_test_b():
        out = []
        for reg_batch, irr_batch in zip(reg_loaders[3], irr_loaders[3]):
            x = reg_batch['x']
            lens = reg_batch['lengths']
            tq = irr_batch['t']
            lq = irr_batch['lengths']
            for i in range(x.shape[0]):
                L_i = lens[i].item()
                out.append((x[i, :L_i], tq[i, :lq[i].item()].numpy()))
        return out

    pairs = _collect_test_b()
    long_pairs = [p for p in pairs if p[0].shape[0] > train_horizon_steps]
    if not long_pairs:
        long_pairs = pairs
    long_pairs.sort(key=lambda p: p[0].shape[0], reverse=True)
    long_pairs = long_pairs[:10]
    ts_trajs = [p[0] for p in long_pairs]
    ts_query_times = [p[1] for p in long_pairs]

    def _pad(traj_list):
        if not traj_list:
            return None, None
        lens = np.array([t.shape[0] for t in traj_list])
        max_L = max(t.shape[0] for t in traj_list)
        padded = torch.zeros(len(traj_list), max_L, 2)
        for i, t in enumerate(traj_list):
            padded[i, :t.shape[0]] = t
        return padded, lens

    padded, sample_lengths = _pad(sample_trajs)
    ts_padded, ts_lengths = _pad(ts_trajs)

    if padded is None:
        raise RuntimeError("No test_A trajectories found in data.")

    max_L = padded.shape[1]
    if ts_padded is not None:
        max_L = max(max_L, ts_padded.shape[1])
    t_eval_vis = torch.from_numpy(t_eval[:max_L]).float()

    print("Generating figures...")
    run_visualization(all_results, padded, t_eval_vis, config, images_dir,
                      lengths=sample_lengths,
                      ts_trajectories=ts_padded, ts_lengths=ts_lengths,
                      node_query_times=ts_query_times)

def run_summarize(config: Dict[str, Any]) -> None:
    """Print a compact Protocol R vs Protocol I comparison table from metrics.json."""
    import json
    metrics_path = Path(__file__).parent / 'data' / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics at {metrics_path}. Run 'full-pipeline' first.")
    m = json.load(open(metrics_path))

    def _mse(entry, split):
        vals = entry[split]['mse']
        return float(np.mean(vals)) if len(vals) else float('nan')

    header = f"{'':<28}{'test_A':>12}{'test_B':>12}{'test_C':>12}"
    print("=" * len(header))
    print("Protocol R  — regular grid (dense, uniform)")
    print(header)
    for name, label in [('discrete_mlp', 'Discrete MLP'), ('node', 'Neural ODE')]:
        if name in m:
            e = m[name]
            print(f"{label:<28}{_mse(e, 'test_A'):>12.4e}{_mse(e, 'test_B'):>12.4e}{_mse(e, 'test_C'):>12.4e}")
    print()
    print("Protocol I  — irregular query times (same for both models)")
    print(header)
    for name, label in [('discrete_mlp', 'Discrete MLP'), ('node', 'Neural ODE')]:
        if name in m and 'irregular_queries' in m[name]:
            e = m[name]['irregular_queries']
            print(f"{label:<28}{_mse(e, 'test_A'):>12.4e}{_mse(e, 'test_B'):>12.4e}{_mse(e, 'test_C'):>12.4e}")
    print("=" * len(header))
    print("(irregular_frac = %s, irregular_strategy = '%s')" % (
        config.get('irregular_frac', 0.5), config.get('irregular_strategy', 'random')))

# STEP 10: CLI Entry Point
def main():
    parser = argparse.ArgumentParser(description='Neural ODE Experiments (Damped Pendulum)')
    parser.add_argument('command', nargs='?', default='full-pipeline', choices=[
        'generate-data', 'train-mlp', 'train-node', 'evaluate', 'visualize',
        'summarize', 'full-pipeline'
    ], help='Command to run (default: full-pipeline)')
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension size')
    parser.add_argument('--n-layers', type=int, help='Number of network layers')
    parser.add_argument('--n-trajectories', type=int, help='Number of trajectories to generate')
    parser.add_argument('--k-periods-train', type=int, help='Training periods (k)')
    parser.add_argument('--test-b-max-periods', type=int, help='Test B max periods (m)')
    parser.add_argument('--dt', type=float, help='Time step')
    parser.add_argument('--irregular-frac', type=float, help='Fraction of regular samples kept for the Neural ODE (irregular view)')
    parser.add_argument('--irregular-strategy', type=str, help="Irregular sampling strategy: 'random' | 'poisson' | 'log'")
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--window-size', type=int, help='Window size for trajectory matching')
    parser.add_argument('--n-windows-per-batch', type=int, help='Windows per batch for training')
    parser.add_argument('--regenerate-data', action='store_true', help='Force regeneration of trajectories')
    args = parser.parse_args()
    
    default_config = {
        'epochs': 200,
        'lr': 1e-3,
        'batch_size': 64,
        'hidden_dim': 64,
        'n_layers': 4,
        'n_trajectories': 300,
        'k_periods_train': 3,
        'test_B_max_periods': 8,
        'test_C_ic_range_factor': 1.5,
        'dt': 0.05,
        'noise_std': 0.01,
        # discrete MLP (one-step flow map)
        'one_step_weight': 0.5,
        'multi_step_weight': 0.5,
        # 'residual' -> x_{k+1} = x + MLP(x) (identity init, stable)
        # 'direct'   -> x_{k+1} = MLP(x)        (unstable to roll out)
        'mlp_form': 'residual',
        # Neural ODE on irregular samples
        'irregular_frac': 0.5,
        'irregular_strategy': 'random',
        # Fair comparison: the ODE is supervised by observed states only.
        # Set True to enable the (unfair) analytic derivative-matching term.
        'use_derivative_loss': False,
        'integration_method': 'dopri5',
        'rtol': 1e-4,
        'atol': 1e-6,
        'use_adjoint': True,
        'adjoint_method': 'rk4',
        'patience': 20,
        'weight_decay': 1e-5,
        'seed': 42,
        'regenerate_data': False,
        'symmetric_augment': True,
        'window_size': 50,
        'window_size_start': 15,
        'n_windows_per_batch': 64,
        'curriculum_epochs': 50
    }
    
    if args.config:
        with open(args.config) as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    # Map CLI args to config keys
    cli_to_config = {
        'epochs': 'epochs',
        'lr': 'lr',
        'batch_size': 'batch_size',
        'hidden_dim': 'hidden_dim',
        'n_layers': 'n_layers',
        'n_trajectories': 'n_trajectories',
        'k_periods_train': 'k_periods_train',
        'test_B_max_periods': 'test_B_max_periods',
        'dt': 'dt',
        'irregular_frac': 'irregular_frac',
        'irregular_strategy': 'irregular_strategy',
        'seed': 'seed',
        'window_size': 'window_size',
        'n_windows_per_batch': 'n_windows_per_batch',
        'regenerate_data': 'regenerate_data',
    }
    
    for key, value in vars(args).items():
        if value is not None and key not in ('command', 'config'):
            config_key = cli_to_config.get(key, key)
            default_config[config_key] = value
    
    if args.command == 'generate-data':
        data_dir = Path(__file__).parent / 'data'
        run_data_generation(default_config, data_dir)
    elif args.command == 'full-pipeline':
        run_full_pipeline(default_config)
    elif args.command == 'visualize':
        run_visualize_only(default_config)
    elif args.command == 'summarize':
        run_summarize(default_config)
    elif args.command in ('train-mlp', 'train-node', 'evaluate'):
        # These commands work through full-pipeline internally
        print(f"Command '{args.command}' runs via full-pipeline. Use 'full-pipeline'.")
        run_full_pipeline(default_config)
    else:
        print(f"Command '{args.command}' not recognized. Available: generate-data, train-mlp, train-node, evaluate, visualize, full-pipeline")

if __name__ == '__main__':
    main()