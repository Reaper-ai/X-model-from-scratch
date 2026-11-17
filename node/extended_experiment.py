"""
Extended-range experiment: train on the FULL pendulum swing.

Main experiment (main.py):      train IC box  θ0 ∈ ±π/2,  ω0 ∈ ±2,  test at 1.5x.
This script (extended_experiment.py): train IC box  θ0 ∈ ±π (full swing), ω0 ∈ ±4;
test_C still at 1.5x of the (now larger) train box → θ0 ∈ ±1.5π, ω0 ∈ ±6.

Everything else (architecture, seeds, dt, horizon, irregular-sampling recipe)
matches main.py exactly, so the only change is the state-space coverage the
networks are trained on.

Trains the Discrete MLP (regular grid) and the Neural ODE (irregular samples),
prints dual-protocol metrics, and saves ONLY two structure-focused figures
with `extended_` prefixes so the main figures are never overwritten:

    docs/images/node/extended_vector_field.png
    docs/images/node/extended_ic_sensitivity.png

Artifacts (all separate from the main experiment):
    node/data/extended_trajectories.npz
    node/data/extended/discrete_mlp_model.pt
    node/data/extended/node_model.pt
    node/data/extended_metrics.json

Usage:
    python extended_experiment.py                  # full run (NODE ~22 s/epoch)
    python extended_experiment.py --epochs 5       # quick smoke test
    python extended_experiment.py --regenerate-data
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

from data_env import (
    generate_trajectories, save_trajectories, load_trajectories,
    create_dataloaders, get_split_masks, make_irregular_view
)
from main import (
    setup_device, setup_models, setup_optimizers,
    run_discrete_mlp_training, run_node_training, run_evaluation
)
from viz import plot_vector_field, plot_ic_sensitivity

# Config: extended IC ranges, otherwise the main recipe
def base_config() -> Dict[str, Any]:
    return {
        # ranges (the point of this experiment)
        'train_theta_range': (-np.pi, np.pi),   # full swing (main: ±π/2)
        'train_omega_range': (-4.0, 4.0),       # main: ±2
        'test_C_ic_range_factor': 1.5,          # test_C = 1.5x the train box
        # plotted grid extents for the extended vector-field figure
        'grid_theta_half': 2 * np.pi,
        'grid_omega_half': 6.0,
        # training (identical to main.py)
        'epochs': 200,
        'lr': 1e-3,
        'batch_size': 64,
        'hidden_dim': 64,
        'n_layers': 4,
        'n_trajectories': 300,
        'k_periods_train': 3,
        'test_B_max_periods': 8,
        'dt': 0.05,
        'noise_std': 0.01,
        'one_step_weight': 0.5,
        'multi_step_weight': 0.5,
        'mlp_form': 'residual',
        'irregular_frac': 0.5,
        'irregular_strategy': 'random',
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
        'curriculum_epochs': 50,
    }

# Pipeline: data -> train -> evaluate -> 2 figures
def run_extended_pipeline(config: Dict[str, Any]) -> None:
    device = setup_device()
    print(f"Using device: {device}")
    print(f"Train IC box: θ0 ∈ ±{abs(config['train_theta_range'][0]):.3f} (full swing), "
          f"ω0 ∈ ±{config['train_omega_range'][1]:.1f}; "
          f"test_C at {config['test_C_ic_range_factor']}x the train box")

    root_dir = Path(__file__).parent
    data_dir = root_dir / 'data'
    model_dir = data_dir / 'extended'
    images_dir = root_dir.parent / 'docs' / 'images' / 'node'
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data with the wider IC box — separate file, main data untouched
    data_path = data_dir / 'extended_trajectories.npz'
    if data_path.exists() and not config.get('regenerate_data', False):
        print(f"Loading existing data from {data_path}")
        t_eval, trajectories, traj_lengths, ic_labels = load_trajectories(str(data_path))
    else:
        print("Generating extended-range pendulum trajectories...")
        t_eval, trajectories, traj_lengths, ic_labels = generate_trajectories(
            n_trajectories=config['n_trajectories'],
            k_periods_train=config['k_periods_train'],
            test_B_max_periods=config['test_B_max_periods'],
            test_C_ic_range_factor=config['test_C_ic_range_factor'],
            dt=config['dt'],
            noise_std=config['noise_std'],
            seed=config['seed'],
            train_theta_range=config['train_theta_range'],
            train_omega_range=config['train_omega_range'],
            symmetric_augment=config['symmetric_augment']
        )
        save_trajectories(t_eval, trajectories, traj_lengths, ic_labels, str(data_path))
        print(f"Saved {len(trajectories)} trajectories to {data_path}")

    # 2. Same two views as main: regular grid -> MLP, irregular -> NODE
    split_masks = get_split_masks(ic_labels, val_fraction=0.1)
    reg_loaders = create_dataloaders(
        trajectories, t_eval, traj_lengths, ic_labels,
        batch_size=config['batch_size'],
        compute_deriv=False, split_masks=split_masks
    )
    irr_times, irr_trajs, irr_lengths = make_irregular_view(
        trajectories, t_eval, traj_lengths,
        frac=config['irregular_frac'],
        strategy=config['irregular_strategy'],
        seed=config['seed']
    )
    irr_loaders = create_dataloaders(
        irr_trajs, t_eval, irr_lengths, ic_labels,
        batch_size=config['batch_size'],
        compute_deriv=False, split_masks=split_masks, times=irr_times
    )
    dt_data = float(t_eval[1] - t_eval[0])
    time_scale = float(np.mean(np.concatenate([np.diff(t) for t in irr_times]))) / dt_data
    print(f"Irregular view: mean sampling gap = {time_scale:.2f} x dt "
          f"(frac={config['irregular_frac']})")

    # 3. Models + training — same recipe, checkpoints in data/extended/
    models = setup_models(config, device)
    optim = setup_optimizers(models, config)
    run_discrete_mlp_training(
        models, reg_loaders[0], reg_loaders[1],
        optim['optimizers'], optim['schedulers'], config, device, model_dir
    )
    run_node_training(
        models, irr_loaders[0], irr_loaders[1],
        optim['optimizers'], optim['schedulers'], config, device, model_dir,
        time_scale=time_scale
    )

    # 4. Dual-protocol metrics (Protocol R regular grid / Protocol I irregular)
    eval_results = run_evaluation(models, reg_loaders, irr_loaders, t_eval, config, device)

    def _ser(obj):
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_ser(v) for v in obj]
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    metrics_path = data_dir / 'extended_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(_ser({'config': _ser(config), 'results': eval_results}), f, indent=2)
    print(f"Extended metrics saved to {metrics_path}")

    # 5. Figures — only the two structure plots, renamed with extended_ prefix
    train_half = (abs(config['train_theta_range'][0]), config['train_omega_range'][1])
    print("Generating extended figures...")
    plot_vector_field(
        models['discrete_mlp'], models['node'],
        images_dir / 'extended_vector_field.png',
        train_half=train_half,
        grid_half=(config['grid_theta_half'], config['grid_omega_half']),
        n_grid=31,
        figsize=(16, 5.2),
        title='Vector Fields: True vs Learned (Extended IC Range)'
    )
    plot_ic_sensitivity(
        models['discrete_mlp'], models['node'],
        torch.from_numpy(t_eval).float(),
        images_dir / 'extended_ic_sensitivity.png',
        train_half=train_half,
        sweep_factor=config['test_C_ic_range_factor'],
        title='IC Sensitivity: Discrete MLP vs Neural ODE (Extended IC Range)'
    )
    print(f"Extended figures saved to {images_dir}")

# CLI
def main():
    parser = argparse.ArgumentParser(
        description='Extended IC-range experiment (Damped Pendulum): '
                    'train θ0 ∈ ±π, ω0 ∈ ±4; test at 1.5x.')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--n-trajectories', type=int, help='Number of trajectories to generate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--regenerate-data', action='store_true',
                        help='Force regeneration of extended trajectories')
    args = parser.parse_args()

    config = base_config()
    for key, value in vars(args).items():
        if value is not None and key != 'regenerate_data':
            config[key] = value
    if args.regenerate_data:
        config['regenerate_data'] = True

    run_extended_pipeline(config)

if __name__ == '__main__':
    main()
