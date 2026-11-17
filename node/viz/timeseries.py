"""Time series figure with seen/unseen regions highlighted."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .style import plt
from .utils import noiseless_reference


def plot_timeseries_seen_unseen(true_trajectories: Tensor, results: dict, t_eval: Tensor, save_path: Path,
                                split_idx: int, lengths: np.ndarray | None = None,
                                pred_key: str = 'trajectories',
                                node_query_times: list[np.ndarray] | None = None) -> None:
    """θ(t) and ω(t) for a few trajectories, seen region left / unseen right of the dotted line.

    The solid black curve is the noise-free reference re-integrated from the
    observed IC; the faint gray curve is the raw (noisy) training data.
    `node_query_times` are the irregular timestamps the Neural ODE trained on,
    drawn as markers so the sampling sparsity is visible.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    t = t_eval.cpu().numpy()
    true_trajs = true_trajectories.cpu().numpy()
    n_show = min(3, len(true_trajs))
    traj_lens = [int(lengths[i]) if lengths is not None else true_trajs.shape[1] for i in range(n_show)]

    # Y-limits from TRUE data only, excluding padding
    valid_theta = np.concatenate([true_trajs[i, :traj_lens[i], 0] for i in range(n_show)])
    valid_omega = np.concatenate([true_trajs[i, :traj_lens[i], 1] for i in range(n_show)])
    y_limits = [(valid_theta.min() - 0.5, valid_theta.max() + 0.5),
                (valid_omega.min() - 0.5, valid_omega.max() + 0.5)]

    refs = [noiseless_reference(true_trajectories[j, :traj_lens[j]], t[:traj_lens[j]]) for j in range(n_show)]

    for dim, ax in enumerate(axes):
        for i in range(n_show):
            L_i = traj_lens[i]
            ax.plot(t[:L_i], refs[i][:, dim], color='black', linewidth=1.8, alpha=0.85,
                    label='True (noise-free)' if i == 0 else '')
            ax.plot(t[:L_i], true_trajs[i, :L_i, dim], color='gray', linewidth=0.8, alpha=0.45,
                    label='Data (noisy)' if i == 0 else '')

            for method_key, mlabel, color in [('discrete_mlp', 'Discrete MLP', '#ff7f0e'),
                                              ('node', 'Neural ODE', '#2ca02c')]:
                if method_key not in results or pred_key not in results[method_key]:
                    continue
                pred_traj = results[method_key][pred_key][i, :L_i].detach().cpu().numpy()
                seen = min(split_idx + 1, L_i)
                ax.plot(t[:seen], pred_traj[:seen, dim], color=color, linewidth=1.5, linestyle='--',
                        alpha=0.8, label=f'{mlabel} seen' if i == 0 else '')
                if L_i > split_idx:
                    ax.plot(t[split_idx:L_i], pred_traj[split_idx:L_i, dim], color=color, linewidth=1.5,
                            linestyle=':', alpha=0.8, label=f'{mlabel} unseen' if i == 0 else '')

            # Irregular timestamps the Neural ODE was trained on
            if dim == 0 and node_query_times is not None and i < len(node_query_times):
                qt = np.asarray(node_query_times[i])
                pos = np.minimum(np.searchsorted(t[:L_i], qt), L_i - 1)
                ax.plot(qt, refs[i][pos, dim], linestyle='none', marker='o', markersize=3.5,
                        color='#2ca02c', alpha=0.55, label='NODE training samples' if i == 0 else '')

            ax.axvline(x=t[split_idx], color='gray', linestyle=':', alpha=0.7, linewidth=2)

        ax.set_ylabel(['θ (rad)', 'ω (rad/s)'][dim])
        ax.set_ylim(*y_limits[dim])
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=9, ncol=3, loc='upper center')

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_xlim(t[0], t[-1])
    plt.suptitle('Time Series: Seen (Train horizon) vs Unseen (Extrapolation)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
