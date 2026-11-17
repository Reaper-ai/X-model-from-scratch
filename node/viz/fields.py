"""Vector field figure: true pendulum field vs the fields the networks learned."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from .style import plt
from data_env import pendulum_system_batch
from model import DiscreteMLP, NeuralODE


def plot_vector_field(discrete_mlp_model: DiscreteMLP, node_model: NeuralODE, save_path: Path, n_grid: int = 25,
                      train_half: tuple[float, float] = (np.pi / 2, 2.0),
                      grid_half: tuple[float, float] = (2 * np.pi, 4.0),
                      figsize: tuple[float, float] = (16, 4.3),
                      title: str = 'Vector Fields: True vs Learned') -> None:
    """1x3 quiver plots: true field vs learned fields over the state space.

    The discrete MLP has no field of its own, so we draw the one it *implies*:
    the Euler-equivalent f(x) = (MLP(x) - x) / dt. The train IC box is shaded
    green; arrow color encodes |f| on a shared scale capped at the 99th
    percentile of the TRUE field (learned fields blow up far outside the data).
    """
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    discrete_mlp_model.eval()
    node_model.eval()
    device = next(discrete_mlp_model.parameters()).device

    theta = np.linspace(-grid_half[0], grid_half[0], n_grid)
    omega = np.linspace(-grid_half[1], grid_half[1], n_grid)
    TH, OM = np.meshgrid(theta, omega)
    states = np.stack([TH.ravel(), OM.ravel()], axis=1)  # (N, 2)

    true_field = pendulum_system_batch(states)
    with torch.no_grad():
        x_t = torch.tensor(states, dtype=torch.float32, device=device)
        mlp_field = discrete_mlp_model.implied_field(x_t).cpu().numpy()
        node_field = node_model(torch.zeros(x_t.shape[0], device=device), x_t).cpu().numpy()

    fields = [
        ('True Pendulum', true_field, '#1f77b4'),
        ('Discrete MLP (implied)', mlp_field, '#ff7f0e'),
        ('Neural ODE', node_field, '#2ca02c'),
    ]

    # Shared magnitude scale, capped at the 99th percentile of the TRUE field
    max_mag = float(np.percentile(np.linalg.norm(true_field, axis=1), 99))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True, layout='constrained')

    for panel_idx, (ax, (label, field, color)) in enumerate(zip(axes, fields)):
        # Train/val IC box behind the arrows
        ax.add_patch(Rectangle((-train_half[0], -train_half[1]), 2 * train_half[0], 2 * train_half[1],
                               facecolor='green', alpha=0.12, edgecolor='green', linewidth=1.5,
                               linestyle='--', label='Train/Val range', zorder=0))

        U, V = field[:, 0].reshape(TH.shape), field[:, 1].reshape(TH.shape)
        mag = np.sqrt(U**2 + V**2)
        # Normalize arrow lengths (direction emphasis), keep color = magnitude
        q = ax.quiver(TH, OM, U / (mag + 1e-12), V / (mag + 1e-12), mag, cmap='viridis',
                      scale=30, width=0.003, alpha=0.85, norm=Normalize(vmin=0, vmax=max_mag))
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('θ (rad)')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-grid_half[0], grid_half[0])
        ax.set_ylim(-grid_half[1], grid_half[1])
        ax.set_aspect('equal', adjustable='box')
        if panel_idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    axes[0].set_ylabel('ω (rad/s)')
    fig.colorbar(q, ax=axes, shrink=0.8, pad=0.02, label='|f(x)|', extend='max')
    plt.suptitle(title, fontsize=14)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
