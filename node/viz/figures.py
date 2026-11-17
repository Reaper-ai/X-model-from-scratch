"""Orchestrator: regenerate all 5 key figures from saved models + data."""
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

from .phase import plot_phase_portrait
from .fields import plot_vector_field
from .timeseries import plot_timeseries_seen_unseen
from .growth import plot_error_growth
from .sensitivity import plot_ic_sensitivity
from data_env import pendulum_system
from model import DiscreteMLP, NeuralODE, ClassicalIntegrator, integrate_neural_ode


def save_all_figures(results: dict, true_trajectories: Tensor, t_eval: Tensor, config: dict, images_dir: Path,
                     lengths: np.ndarray | None = None, ts_trajectories: Tensor | None = None,
                     ts_lengths: np.ndarray | None = None,
                     node_query_times: list[np.ndarray] | None = None) -> None:
    """Generate the 5 key comparison figures.

    true_trajectories/lengths: the in-distribution test sample (phase portrait +
    error growth). ts_trajectories/ts_lengths: the long-horizon sample for the
    timeseries figure (falls back to true_trajectories). node_query_times: the
    irregular timestamps the Neural ODE trained on (markers in the timeseries).
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    device = true_trajectories.device

    # Train horizon in steps: k_periods_train * T / dt  (T = 2π√(L/g))
    period = 2.0 * np.pi * np.sqrt(config.get('L', 1.0) / config.get('g', 9.81))
    train_horizon_steps = int(round(config.get('k_periods_train', 3) * period / config.get('dt', 0.05)))

    # Build models with config dimensions and load saved weights if available
    discrete_mlp_model = DiscreteMLP(input_dim=2, hidden_dim=config.get('hidden_dim', 64),
                                     n_layers=config.get('n_layers', 4), output_dim=2,
                                     residual=config.get('mlp_form', 'residual') != 'direct').to(device)
    node_model = NeuralODE(input_dim=2, hidden_dim=config.get('hidden_dim', 64),
                           n_layers=config.get('n_layers', 4), output_dim=2).to(device)
    model_dir = Path(__file__).parent.parent / 'data'  # path relative to this file, works from any cwd
    if (model_dir / 'discrete_mlp_model.pt').exists():
        discrete_mlp_model.load_state_dict(torch.load(model_dir / 'discrete_mlp_model.pt', map_location=device))
    if (model_dir / 'node_model.pt').exists():
        node_model.load_state_dict(torch.load(model_dir / 'node_model.pt', map_location=device))

    def _integrate_sample(trajs: Tensor) -> dict[str, Tensor]:
        """Generate all model trajectories over a padded sample from its ICs."""
        discrete_out, node_out = [], []
        classical_out = {k: [] for k in ['rk45', 'dop853', 'euler', 'rk4']}
        method_map = {'rk45': 'RK45', 'dop853': 'DOP853', 'euler': 'Euler', 'rk4': 'RK4'}
        n_steps = int(t_eval.numel()) - 1
        for i in range(len(trajs)):
            x0 = trajs[i, 0].to(device)
            discrete_out.append(discrete_mlp_model.rollout(x0, n_steps).cpu())
            node_out.append(integrate_neural_ode(node_model, x0, t_eval.to(device), use_adjoint=False).cpu())
            for name, m in method_map.items():
                classical_out[name].append(
                    ClassicalIntegrator(method=m).integrate(pendulum_system, x0, t_eval.to(device)).cpu())
        out = {'discrete_mlp': torch.stack(discrete_out), 'node': torch.stack(node_out)}
        for k, v in classical_out.items():
            out[f'classical_{k}'] = torch.stack(v)
        return out

    # Always regenerate trajectories for the in-distribution sample (phase + error growth)
    print("Regenerating trajectories for visualization...")
    pred_A = _integrate_sample(true_trajectories)
    results.setdefault('discrete_mlp', {})['trajectories'] = pred_A['discrete_mlp']
    results.setdefault('node', {})['trajectories'] = pred_A['node']
    for k in ['rk45', 'dop853', 'euler', 'rk4']:
        results.setdefault(f'classical_{k}', {})['trajectories'] = pred_A[f'classical_{k}']

    # Separate predictions for the long-horizon sample (timeseries)
    if ts_trajectories is not None:
        pred_B = _integrate_sample(ts_trajectories)
        results.setdefault('discrete_mlp', {})['trajectories_ts'] = pred_B['discrete_mlp']
        results.setdefault('node', {})['trajectories_ts'] = pred_B['node']
        ts_true, ts_len = ts_trajectories, (ts_lengths if ts_lengths is not None else lengths)
    else:
        ts_true, ts_len = true_trajectories, lengths

    # The timeseries uses the train horizon as seen/unseen split (extrapolation)
    split_idx = min(train_horizon_steps, len(t_eval) - 1)

    plot_phase_portrait(true_trajectories, results, t_eval, images_dir / 'phase_portrait.png', lengths=lengths)
    plot_vector_field(discrete_mlp_model, node_model, images_dir / 'vector_field.png')
    plot_timeseries_seen_unseen(ts_true, results, t_eval, images_dir / 'timeseries_seen_unseen.png',
                                split_idx, lengths=ts_len, pred_key='trajectories_ts',
                                node_query_times=node_query_times)
    plot_error_growth(results, true_trajectories, t_eval, images_dir / 'error_growth.png', lengths=lengths)
    plot_ic_sensitivity(discrete_mlp_model, node_model, t_eval, images_dir / 'ic_sensitivity.png')

    print(f"All 5 figures saved to {images_dir}")
