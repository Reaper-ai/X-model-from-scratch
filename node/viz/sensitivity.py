"""IC sensitivity figure: trajectory MSE vs starting condition for both models."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .style import plt
from data_env import pendulum_system
from model import DiscreteMLP, NeuralODE, integrate_neural_ode
from scipy.integrate import solve_ivp


def plot_ic_sensitivity(discrete_mlp_model: DiscreteMLP, node_model: NeuralODE, t_eval: Tensor, save_path: Path,
                        node_method: str = 'dopri5', rtol: float = 1e-4, atol: float = 1e-6,
                        train_half: tuple[float, float] = (np.pi / 2, 2.0), sweep_factor: float = 1.5,
                        title: str = 'IC Sensitivity: Discrete MLP vs Neural ODE') -> None:
    """Sweep θ0 and ω0 (to `sweep_factor` x the training box) and plot rollout MSE.

    Green shading marks the training range, red the extrapolation beyond it.
    """
    device = next(discrete_mlp_model.parameters()).device
    discrete_mlp_model.eval()
    node_model.eval()

    sweep_theta, sweep_omega = sweep_factor * train_half[0], sweep_factor * train_half[1]
    sweep_params = [
        {'dim': 0, 'range': (-sweep_theta, sweep_theta), 'fixed': (0.0, 0.0), 'label': 'θ0',
         'train_range': (-train_half[0], train_half[0])},
        {'dim': 1, 'range': (-sweep_omega, sweep_omega), 'fixed': (0.0, 0.0), 'label': 'ω0',
         'train_range': (-train_half[1], train_half[1])},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t_eval_np = t_eval.cpu().numpy()

    for ax_idx, sweep in enumerate(sweep_params):
        ax = axes[ax_idx]
        dim, sweep_range, train_range = sweep['dim'], sweep['range'], sweep['train_range']
        sweep_vals = np.linspace(sweep_range[0], sweep_range[1], 50)
        discrete_errors, node_errors = [], []

        for val in sweep_vals:
            x0 = np.zeros(2)
            x0[dim] = val
            x0_torch = torch.tensor(x0, dtype=torch.float32, device=device)

            sol = solve_ivp(lambda t, s: pendulum_system(t, s), (t_eval_np[0], t_eval_np[-1]), x0,
                            t_eval=t_eval_np, method='RK45', rtol=1e-9, atol=1e-12)
            true_traj = torch.tensor(sol.y.T, dtype=torch.float32, device=device)

            # Discrete MLP: iterate the learned map (no solver, fixed dt)
            pred_discrete = discrete_mlp_model.rollout(x0_torch, int(t_eval.numel()) - 1)
            discrete_errors.append(torch.mean((pred_discrete - true_traj) ** 2).item())

            # Neural ODE (adaptive solver, forward-only — no adjoint needed)
            pred_node = integrate_neural_ode(node_model, x0_torch, t_eval.to(device), method=node_method,
                                             use_adjoint=False, rtol=rtol, atol=atol)
            node_errors.append(torch.mean((pred_node - true_traj) ** 2).item())

        ax.semilogy(sweep_vals, discrete_errors, 'o-', color='#ff7f0e', linewidth=2, markersize=4, label='Discrete MLP')
        ax.semilogy(sweep_vals, node_errors, 's-', color='#2ca02c', linewidth=2, markersize=4, label='Neural ODE')
        ax.set_xlabel(f'Initial {sweep["label"]}')
        ax.set_ylabel('Trajectory MSE (log scale)')
        ax.set_title(f'IC Sensitivity: Error vs {sweep["label"]}')
        ax.grid(True, alpha=0.3)

        # Training range (green) and test region (red, outside training)
        ax.axvspan(train_range[0], train_range[1], alpha=0.1, color='green', label='Training range')
        if sweep_range[0] < train_range[0]:
            ax.axvspan(sweep_range[0], train_range[0], alpha=0.08, color='red', label='Test (extrapolation)')
        if sweep_range[1] > train_range[1]:
            ax.axvspan(train_range[1], sweep_range[1], alpha=0.08, color='red',
                       label='Test (extrapolation)' if sweep_range[0] >= train_range[0] else '')

        ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
