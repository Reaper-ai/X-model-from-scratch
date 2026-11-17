"""Error growth figure: rollout distance vs time for every method."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .style import plt
from .utils import noiseless_reference


def plot_error_growth(results: dict, true_trajectories: Tensor, t_eval: Tensor, save_path: Path,
                      lengths: np.ndarray | None = None) -> None:
    """Log-scale error growth over time for all methods (vs noise-free reference)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    t = t_eval.cpu().numpy()
    methods = ['discrete_mlp', 'node', 'classical_rk45', 'classical_dop853', 'classical_euler', 'classical_rk4']
    colors = {'discrete_mlp': '#ff7f0e', 'node': '#2ca02c', 'classical_rk45': '#1f77b4',
              'classical_dop853': '#bcbd22', 'classical_euler': '#d62728', 'classical_rk4': '#9467bd'}
    labels = {'discrete_mlp': 'Discrete MLP', 'node': 'Neural ODE', 'classical_rk45': 'RK45',
              'classical_dop853': 'DOP853', 'classical_euler': 'Euler', 'classical_rk4': 'RK4'}

    max_plotted = 0
    for method in methods:
        if method not in results or 'trajectories' not in results[method]:
            continue
        pred_trajs = results[method]['trajectories']
        if not isinstance(pred_trajs, torch.Tensor):
            continue
        pred_trajs_np = pred_trajs.detach().cpu().numpy()

        n_show = min(5, len(true_trajectories))
        all_dist = []
        for i in range(n_show):
            L_i = min(int(lengths[i]) if lengths is not None else true_trajs_np.shape[1],
                      pred_trajs_np.shape[1], true_trajectories.shape[1])
            ref = noiseless_reference(true_trajectories[i, :L_i], t[:L_i])
            dist = np.linalg.norm(pred_trajs_np[i, :L_i] - ref, axis=1)
            if np.sum(dist > 1e-10) > 5:
                all_dist.append(dist)

        if not all_dist:
            continue

        # Individual trajectories (thin)
        for dist in all_dist:
            ax.semilogy(t[:len(dist)], dist, color=colors[method], alpha=0.15, linewidth=0.5)

        # Mean (thick): NaN-pad to longest so no premature cutoff at the shortest traj
        max_len = max(len(d) for d in all_dist)
        padded = np.full((len(all_dist), max_len), np.nan)
        for j, d in enumerate(all_dist):
            padded[j, :len(d)] = d
        ax.semilogy(t[:max_len], np.nanmean(padded, axis=0), color=colors[method],
                    linewidth=2.5, label=labels.get(method, method), alpha=0.9)
        max_plotted = max(max_plotted, max_len)

    ax.axhline(y=1e2, color='black', linestyle=':', alpha=0.5, label='Saturation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State Distance (log scale)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # X-limit: longest trajectory actually plotted (t may extend beyond the sample)
    ax.set_xlim(t[0], t[min(max_plotted, len(t) - 1)] if max_plotted > 0 else t[-1])
    ax.set_ylim(1e-8, 1e2)

    plt.suptitle('Error Growth Over Time', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
