"""Phase portrait figure: 1x3 panels, one per method."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor

from .style import plt
from .utils import noiseless_reference, wrap_angle


def plot_phase_portrait(true_trajectories: Tensor, results: dict, t_eval: Tensor, save_path: Path,
                        lengths: np.ndarray | None = None) -> None:
    """True orbit vs each method's rollout for the biggest-IC test sample.

    Predictions are drawn light and semi-transparent so the dark true
    trajectory stays visible underneath. Axes fit the plotted data with 15%
    padding, symmetrized around the origin (panels stretch to fill their slots).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    # Pick the sample with the biggest normalized IC magnitude:
    # |θ0| / (π/2) + |ω0| / 2  (both terms in [0, 1] over the train IC box)
    ics = true_trajectories[:, 0, :].cpu().numpy()
    idx = int(np.argmax(np.abs(ics[:, 0]) / (np.pi / 2) + np.abs(ics[:, 1]) / 2.0))

    L_i = min(int(lengths[idx]), true_trajectories.shape[1]) if lengths is not None else true_trajectories.shape[1]
    true_traj = noiseless_reference(true_trajectories[idx, :L_i], t_eval.cpu().numpy()[:L_i])
    true_theta, true_omega = wrap_angle(true_traj[:, 0]), true_traj[:, 1]

    xmin, xmax = float(true_theta.min()), float(true_theta.max())
    ymin, ymax = float(true_omega.min()), float(true_omega.max())

    method_configs = [
        ('discrete_mlp', 'Discrete MLP', '#ff7f0e'),
        ('node', 'Neural ODE', '#2ca02c'),
        ('classical_rk45', 'Classical RK45', '#1f77b4'),
    ]

    for ax, (method_key, label, color) in zip(axes, method_configs):
        ax.plot(true_theta, true_omega, color='black', linewidth=2.5, alpha=0.9, label='True', zorder=3)

        if method_key in results and 'trajectories' in results[method_key]:
            pred = results[method_key]['trajectories'][idx, :L_i].detach().cpu().numpy()
            theta, omega = wrap_angle(pred[:, 0]), pred[:, 1]
            ax.plot(theta, omega, color=color, linewidth=4.0, alpha=0.35, label=label, zorder=2)
            ax.plot(theta, omega, color=color, linewidth=1.2, alpha=0.9, linestyle='--', zorder=4)
            xmin, xmax = min(xmin, float(theta.min())), max(xmax, float(theta.max()))
            ymin, ymax = min(ymin, float(omega.min())), max(ymax, float(omega.max()))

        ax.scatter(true_theta[0], true_omega[0], color='green', s=70, zorder=5)
        ax.scatter(true_theta[-1], true_omega[-1], color='red', s=70, zorder=5)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('θ (rad)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    # Fit limits with padding, symmetrized around the origin
    xhalf = max(abs(xmin - max(0.15 * (xmax - xmin), 0.2)), abs(xmax + max(0.15 * (xmax - xmin), 0.2)))
    yhalf = max(abs(ymin - max(0.15 * (ymax - ymin), 0.2)), abs(ymax + max(0.15 * (ymax - ymin), 0.2)))
    for ax in axes:
        ax.set_xlim(-xhalf, xhalf)
        ax.set_ylim(-yhalf, yhalf)

    axes[0].set_ylabel('ω (rad/s)')
    axes[0].scatter([], [], color='green', s=70, label='Start')
    axes[0].scatter([], [], color='red', s=70, label='End')
    axes[0].legend(fontsize=8, loc='upper right')

    plt.suptitle('Phase Portrait: True vs Each Method', fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.08)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
