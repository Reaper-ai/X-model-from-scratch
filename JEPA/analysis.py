"""
Analysis and Visualization Functions for JEPA World Model.

Generates:
- Architecture diagram (U-shaped block diagram)
- Training curves animated GIF (with val points + horizon-shift markers)
- MPC trajectory visualization
- Linear probe accuracy plots
- Animated full pipeline visualization
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import imageio.v2 as imageio
import seaborn as sns

from env import env_reset, env_step, render_observation
from encoder import encode_batch
from predictor import predict_next_embedding, predictor_forward
from planning import run_mpc_episode

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


def _box(ax, x, y, w, h, text, fc='#4C9BE8', ec='black', fs=14, tc='white'):
    """Draw a rounded block."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold', wrap=True)


def _arrow(ax, x1, y1, x2, y2, label=None, color='black', style='-|>',
            lx=None, ly=None, fs=13):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                          mutation_scale=16, linewidth=1.5, color=color,
                          shrinkA=2, shrinkB=2)
    ax.add_patch(arr)
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2,
                ly if ly is not None else (y1 + y2) / 2 + 0.12,
                label, ha='center', va='bottom', fontsize=fs,
                fontstyle='italic', color=color)


def plot_architecture(
    output_path: str = "images/architecture.png",
    room_size: int = 6,
    latent_dim: int = 16,
    hidden_dim: int = 32,
) -> None:
    """U-shaped JEPA block diagram: Encoder | Target Encoder | Predictor | Probe."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.1)  # tight above top boxes (y max 7.8)
    ax.axis('off')
    # Title + dim labels stacked tightly; diagram starts right below
    fig.subplots_adjust(top=0.84, bottom=0.06, left=0.02, right=0.98)
    fig.suptitle('JEPA Architecture', fontsize=20, fontweight='bold', y=0.97, va='top')
    fig.text(0.5, 0.91,
             f'{room_size}×{room_size} room  ·  latent dim {latent_dim}  ·  hidden {hidden_dim}',
             ha='center', va='top', fontsize=13, color='#666')

    # --- Top branch (online path, left to right) ---
    _box(ax, 0.3, 6.8, 1.8, 1.0, f'obs\n{room_size}×{room_size}', fc='#999', fs=13)
    _arrow(ax, 2.2, 7.3, 2.9, 7.3)
    _box(ax, 3.0, 6.8, 2.2, 1.0, 'Encoder\n(online)', fc='#4C9BE8')
    _arrow(ax, 5.3, 7.3, 6.4, 7.3, label='z', ly=7.42)
    _box(ax, 6.5, 6.8, 2.2, 1.0, 'Predictor', fc='#E8A838')
    _arrow(ax, 8.8, 7.3, 9.9, 7.3, label=r"$\hat{z}'$", ly=7.42)
    _box(ax, 10.0, 6.8, 2.6, 1.0, 'JEPA Loss', fc='#D9534F')

    # action -> predictor (upward)
    _box(ax, 6.8, 5.0, 1.6, 0.7, 'action', fc='#999', fs=13)
    _arrow(ax, 7.6, 5.75, 7.6, 6.75)

    # --- Bottom branch (target path, feeds loss from below) ---
    _box(ax, 10.0, 1.2, 2.6, 1.0, f'next_obs\n{room_size}×{room_size}', fc='#999', fs=13)
    _arrow(ax, 11.3, 2.25, 11.3, 3.15)
    _box(ax, 10.0, 3.2, 2.6, 1.0, 'Target Encoder\n(EMA, frozen)', fc='#5CB85C')
    _arrow(ax, 11.3, 4.25, 11.3, 6.75, label="z′", lx=11.65, ly=5.4, color='#5CB85C')
    ax.text(11.55, 4.7, 'stop-grad', fontsize=11, color='#5CB85C', ha='left', rotation=90, va='center')

    # --- z taps down to probe (left side of U) ---
    ax.plot([5.8, 5.8], [4.6, 7.3], color='#4C9BE8', lw=1.5, zorder=1)
    _arrow(ax, 5.8, 4.6, 5.8, 3.75, color='#4C9BE8')
    _box(ax, 4.5, 2.7, 2.6, 1.0, 'Linear Probe', fc='#9B59B6')
    _arrow(ax, 7.2, 3.2, 8.3, 3.2, label='(x, y)', ly=3.32, color='#9B59B6')
    _box(ax, 8.4, 2.7, 1.4, 1.0, 'position', fc='#999', fs=13)

    ax.text(6.5, 0.5,
            'U-shape: online path (top, left→right) · target path (bottom, feeds loss from below)',
            ha='center', fontsize=12, color='#888', style='italic')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fmt = output_path.rsplit('.', 1)[-1] if '.' in output_path else 'png'
    # Top pad 0.5 = 2× the original 0.25
    fig.savefig(output_path, format=fmt, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f"Architecture diagram saved to {output_path}")


def plot_training_curves(history: list, output_path: str = "images/training_curves.gif",
                          fps: int = 10, max_frames: int = 250, steps_per_epoch: int = 12) -> None:
    """Generate animated GIF of training loss, val loss, and collapse metric over epochs."""
    # Extract val points and horizon-shift steps from FULL history before subsampling,
    # so subsampling never drops irregularly-spaced val checks.
    val_steps = [h['step'] for h in history if 'val_loss' in h]
    val_losses = [h['val_loss'] for h in history if 'val_loss' in h]

    horizon_shifts = []
    prev_h = None
    for h in history:
        if prev_h is not None and h['horizon'] != prev_h:
            horizon_shifts.append((h['step'], h['horizon']))
        prev_h = h['horizon']

    # Subsample history to at most max_frames so GIF save stays cheap
    n = len(history)
    if n > max_frames:
        idxs = np.linspace(0, n - 1, max_frames).astype(int)
        history = [history[i] for i in idxs]

    train_losses = [h['train_loss'] for h in history]
    collapses = [h['collapse'] for h in history]
    # Convert step -> epoch for the x-axis (1 epoch = steps_per_epoch steps)
    steps = [h['step'] / steps_per_epoch for h in history]
    total_epochs = steps[-1] if steps else 1
    val_epochs = [s / steps_per_epoch for s in val_steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('JEPA Training Progress', fontsize=14, fontweight='bold')

    # Setup axes
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('JEPA Loss (Train + Val)')
    ax1.set_xlim(0, total_epochs)
    ax1.set_ylim(0, max(train_losses) * 1.1 if train_losses else 1)

    # Mark horizon curriculum shifts (never silent)
    y_top = ax1.get_ylim()[1]
    for shift_step, shift_horizon in horizon_shifts:
        shift_epoch = shift_step / steps_per_epoch
        if shift_epoch <= total_epochs:
            ax1.axvline(x=shift_epoch, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
            # Label near mid-height so it never sits under the legend
            ax1.text(shift_epoch, y_top * 0.55, f' h→{shift_horizon}',
                     color='purple', fontsize=11, fontweight='bold',
                     va='center', ha='left',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='purple', alpha=0.9))

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Collapse Metric (mean std)')
    ax2.set_title('Collapse Metric (target ≈ 1.0)')
    ax2.set_xlim(0, total_epochs)
    ax2.set_ylim(0, max(collapses) * 1.1 if collapses else 1.5)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target (γ=1.0)')
    ax2.legend(fontsize=9)

    line_train, = ax1.plot([], [], 'b-', linewidth=2, label='Train Loss')
    line_val, = ax1.plot([], [], 'orange', linewidth=2, marker='o', markersize=3, label='Val Loss')
    line_collapse, = ax2.plot([], [], 'r-', linewidth=2, label='Collapse')

    # Legend top-left: shift lines sit later in training (right side)
    ax1.legend(fontsize=9, loc='upper left')

    def init():
        line_train.set_data([], [])
        line_val.set_data([], [])
        line_collapse.set_data([], [])
        return line_train, line_val, line_collapse

    def animate(i):
        line_train.set_data(steps[:i+1], train_losses[:i+1])
        line_collapse.set_data(steps[:i+1], collapses[:i+1])
        if val_epochs and val_losses:
            n_val = sum(1 for s in val_epochs if s <= steps[i])
            if n_val > 0:
                line_val.set_data(val_epochs[:n_val], val_losses[:n_val])
        return line_train, line_val, line_collapse

    ani = animation.FuncAnimation(fig, animate, frames=len(history),
                                   init_func=init, blit=True, interval=1000/fps)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Training curves GIF saved to {output_path} ({len(history)} frames)")


def plot_mpc_trajectories(
    encoder_params: dict,
    predictor_params: dict,
    room_size: int = 8,
    max_steps: int = 20,
    n_sequences: int = 64,
    horizon: int = 5,
    n_actions: int = 4,
    output_path: str = "images/mpc_trajectories.png"
) -> None:
    """Visualize MPC agent trajectories: top row successes, bottom row failures."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    fig.suptitle('MPC Planner: 5 Successes (top) vs 5 Failures (bottom)', fontsize=14, fontweight='bold')

    successes, failures = [], []
    for _ in range(200):
        if len(successes) >= 5 and len(failures) >= 5:
            break
        goal_pos = torch.randint(0, room_size, (2,)).float()
        result = run_mpc_episode(
            encoder_params, predictor_params, goal_pos,
            room_size, agent_size=1, max_steps=max_steps,
            n_sequences=n_sequences, horizon=horizon, n_actions=n_actions
        )
        item = (goal_pos, result)
        if result['success'] and len(successes) < 5:
            successes.append(item)
        elif not result['success'] and len(failures) < 5:
            failures.append(item)

    panels = [(g, r, True) for g, r in successes[:5]] + [(g, r, False) for g, r in failures[:5]]

    for ep, (goal_pos, result, is_success) in enumerate(panels):
        ax = axes[ep]
        trajectory = np.array(result['trajectory'])
        success = result['success']

        # Plot room grid
        ax.set_xlim(-0.5, room_size - 0.5)
        ax.set_ylim(-0.5, room_size - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(room_size))
        ax.set_yticks(range(room_size))
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Match render_observation coordinates

        # Plot trajectory
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color='blue' if success else 'red',
                    markersize=6, linewidth=2, alpha=0.7, label=f'Path ({len(trajectory)-1} steps)')

        # Start position
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start')
        # Goal position
        ax.plot(goal_pos[0].item(), goal_pos[1].item(), 'r*', markersize=15, label='Goal')
        # Final position
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'bX' if success else 'rX', markersize=12, label='End')

        ax.set_title(f'{"Success" if success else "Failed"} ({len(trajectory)-1} steps)')

    # Hide unused subplots
    for ep in range(len(panels), 10):
        axes[ep].set_visible(False)

    # Shared legend OUTSIDE the axes (below), so it never overlaps a path
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='blue', marker='o', lw=2, label='Path (success)'),
        Line2D([0], [0], color='red', marker='o', lw=2, label='Path (failure)'),
        Line2D([0], [0], color='green', marker='o', ls='', ms=10, label='Start'),
        Line2D([0], [0], color='red', marker='*', ls='', ms=14, label='Goal'),
        Line2D([0], [0], color='blue', marker='X', ls='', ms=9, label='End'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=10, frameon=False)

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"MPC trajectories saved to {output_path}")


def plot_probe_accuracy(
    probe_results: dict,
    dataset: dict,
    encoder_params: dict,
    output_path: str = "images/probe_accuracy.png"
) -> None:
    """Visualize linear probe accuracy: predicted vs actual positions."""
    obs = dataset['observations']
    states = dataset['states']
    room_size = obs.shape[-1]
    lim_max = room_size - 1

    with torch.no_grad():
        embeddings = encode_batch(obs, encoder_params)
        w = probe_results['probe_params']['w'].to(embeddings.device)
        b = probe_results['probe_params']['b'].to(embeddings.device)
        predicted_states = embeddings @ w.T + b

    pred_np = predicted_states.cpu().numpy()
    true_np = states.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Linear Probe: Latent → Agent Position', fontsize=14, fontweight='bold')

    # X coordinate: predicted vs actual
    axes[0].scatter(true_np[:, 0], pred_np[:, 0], alpha=0.6, s=20, c='blue')
    axes[0].plot([0, lim_max], [0, lim_max], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True X Position')
    axes[0].set_ylabel('Predicted X Position')
    axes[0].set_title(f'X Coordinate (MAE: {probe_results["mean_abs_error"]:.3f})')
    axes[0].legend()
    axes[0].set_xlim(-0.5, lim_max + 0.5)
    axes[0].set_ylim(-0.5, lim_max + 0.5)
    axes[0].set_aspect('equal')

    # Y coordinate: predicted vs actual
    axes[1].scatter(true_np[:, 1], pred_np[:, 1], alpha=0.6, s=20, c='green')
    axes[1].plot([0, lim_max], [0, lim_max], 'r--', label='Perfect prediction')
    axes[1].set_xlabel('True Y Position')
    axes[1].set_ylabel('Predicted Y Position')
    axes[1].set_title(f'Y Coordinate (MAE: {probe_results["mean_abs_error"]:.3f})')
    axes[1].legend()
    axes[1].set_xlim(-0.5, lim_max + 0.5)
    axes[1].set_ylim(-0.5, lim_max + 0.5)
    axes[1].set_aspect('equal')

    # Error distribution
    errors = np.abs(pred_np - true_np)
    axes[2].hist(errors.flatten(), bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[2].axvline(probe_results['mean_abs_error'], color='red', linestyle='--', linewidth=2,
                    label=f'Mean Abs Error: {probe_results["mean_abs_error"]:.3f}')
    axes[2].set_xlabel('Absolute Error (cells)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution')
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Probe accuracy plot saved to {output_path}")


class LinearDecoder(nn.Module):
    """Simple linear head: latent (D) -> cell softmax (1, H, W) for visualization."""
    def __init__(self, latent_dim: int = 32, room_size: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.room_size = room_size
        self.output_size = room_size * room_size
        self.fc = nn.Linear(latent_dim, self.output_size)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation grid (cell probabilities)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        logits = self.fc(z)
        return torch.softmax(logits, dim=-1).view(-1, 1, self.room_size, self.room_size)


def train_linear_decoder(
    encoder_params: dict,
    dataset: dict,
    room_size: int = 8,
    latent_dim: int = None,
    num_steps: int = 1500,
    lr: float = 1e-2
) -> LinearDecoder:
    """Train a linear cell classifier to decode observations from latents."""
    device = dataset['observations'].device
    if latent_dim is None:
        with torch.no_grad():
            test_emb = encode_batch(dataset['observations'][:1], encoder_params)
            latent_dim = test_emb.shape[1]

    decoder = LinearDecoder(latent_dim, room_size).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    obs = dataset['observations'][:512]  # Use subset for speed
    target = obs.flatten(1).argmax(1)    # agent cell index

    with torch.no_grad():
        embeddings = encode_batch(obs, encoder_params).detach()

    decoder.train()
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(decoder.fc(embeddings), target)
        loss.backward()
        optimizer.step()

    decoder.eval()
    return decoder


def plot_animated_pipeline(
    dataset: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    probe_results: dict,
    room_size: int = 8,
    num_steps: int = 15,
    output_path: str = "images/pipeline_animation.gif",
    fps: int = 3
) -> None:
    """
    Animated full JEPA pipeline visualization.
    
    Layout (3 columns):
    LEFT:   Room -> Encoder -> Latent z (vector)
    CENTER: Predictor(z, action) -> z' predicted -> Decode to room
    RIGHT:  Probe(z) -> Position, Target Encoder (ghost) -> z' target
    
    Shows the complete data flow for a single trajectory.
    """
    # Train decoder for visualization
    print("Training linear decoder for pipeline animation...")
    with torch.no_grad():
        test_emb = encode_batch(dataset['observations'][:1], encoder_params)
        latent_dim = test_emb.shape[1]
    
    decoder = train_linear_decoder(encoder_params, dataset, room_size, latent_dim)

    # Probe predictions over full dataset (for scoring windows)
    with torch.no_grad():
        probe_w = probe_results['probe_params']['w']
        probe_b = probe_results['probe_params']['b']
        z_all = encode_batch(dataset['observations'], encoder_params)
        pos_all = z_all @ probe_w.T + probe_b  # (N, 2)

    # Pick a window WITHIN one episode (dataset = several concatenated rollouts;
    # crossing a boundary teleports the agent and makes the probe look dead).
    # Score = path movement - 3 * probe error (prefer a window the probe tracks well).
    states_all = dataset['states']  # (N, 2)
    n_total = len(states_all)
    # Episode breaks: consecutive states jump by >1 (max real step is 1)
    if n_total > 1:
        jumps = torch.norm(states_all[1:] - states_all[:-1], dim=-1)
        breaks = [0] + (torch.where(jumps > 1.5)[0] + 1).tolist() + [n_total]
    else:
        breaks = [0, n_total]

    best_idx, best_score = 0, -1e9
    for b0, b1 in zip(breaks, breaks[1:]):
        for i in range(b0, max(b0, b1 - num_steps)):
            if i + num_steps + 1 > b1:
                break
            window = states_all[i:i + num_steps + 1]
            move = float(torch.norm(window[1:] - window[:-1], dim=-1).sum())
            err = float((pos_all[i:i + num_steps + 1] - window).abs().mean())
            score = move - 3.0 * err
            if score > best_score:
                best_score, best_idx = score, i
    idx = best_idx
    print(f"Pipeline example: start={idx}, score={best_score:.2f} "
          f"(episodes at {breaks})")
    obs_seq = dataset['observations'][idx:idx+num_steps+1]
    action_seq = dataset['actions'][idx:idx+num_steps]
    state_seq = dataset['states'][idx:idx+num_steps+1]
    
    with torch.no_grad():
        # Encode all observations
        z_seq = encode_batch(obs_seq, encoder_params)
        
        # Target encodings
        z_target_seq = encode_batch(obs_seq[1:], target_params)
        
        # Rollout predictions
        z_pred = z_seq[0:1]  # Start with first latent
        z_pred_seq = [z_pred]
        for t in range(num_steps):
            z_pred = predictor_forward(z_pred, action_seq[t:t+1], predictor_params)
            z_pred_seq.append(z_pred)
        z_pred_seq = torch.cat(z_pred_seq, dim=0)  # (T+1, 1, D)
        
        # Decode predictions - decode each timestep separately
        z_pred_decoded = []
        for t in range(num_steps):
            decoded = decoder(z_pred_seq[t+1])  # (1, 1, H, W)
            z_pred_decoded.append(decoded)
        z_pred_decoded = torch.cat(z_pred_decoded, dim=0)  # (T, 1, H, W)
        
        z_target_decoded = []
        for t in range(num_steps):
            decoded = decoder(z_target_seq[t:t+1])  # (1, 1, H, W)
            z_target_decoded.append(decoded)
        z_target_decoded = torch.cat(z_target_decoded, dim=0)  # (T, 1, 1, H, W)
        
        # Probe predictions (recomputed above as pos_all)
        pos_pred_seq = pos_all[idx:idx+num_steps+1]
        pos_true_seq = state_seq

    # Create animation — bottom row: thin barh strips for latent vectors
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.15], width_ratios=[1, 1, 1],
                          hspace=0.55, left=0.05, right=0.97, top=0.88, bottom=0.08)

    # Axes
    ax_room = fig.add_subplot(gs[0, 0])      # Current room
    ax_encoder = fig.add_subplot(gs[1, 0])   # Latent z barh strip
    ax_predictor = fig.add_subplot(gs[:, 1]) # Center: predictor + decode
    ax_probe = fig.add_subplot(gs[0, 2])     # Probe output
    ax_target = fig.add_subplot(gs[1, 2])    # Target z' barh strip

    fig.suptitle('JEPA Pipeline Animation', fontsize=18, fontweight='bold', y=0.97)

    # Colormap for vector values (fixed scale so frames are comparable)
    vec_norm = matplotlib.colors.Normalize(vmin=-3, vmax=3)
    vec_cmap = plt.cm.RdBu_r

    def draw_vector_strip(ax, values, title):
        """Single horizontal bar: one segment per dim, colored by value."""
        ax.clear()
        for i, v in enumerate(values):
            ax.barh(0, 1.0, left=i, height=1.0,
                    color=vec_cmap(vec_norm(float(v))), edgecolor='none')
        ax.set_xlim(0, len(values))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks(range(0, len(values), 4))
        ax.set_xticklabels(range(0, len(values), 4), fontsize=11)
        ax.set_title(title, fontsize=14, pad=4)
        ax.set_xlabel('')

    # Room setup
    ax_room.set_xlim(-0.5, room_size - 0.5)
    ax_room.set_ylim(-0.5, room_size - 0.5)
    ax_room.set_aspect('equal')
    ax_room.set_xticks(range(room_size))
    ax_room.set_yticks(range(room_size))
    ax_room.tick_params(labelsize=11)
    ax_room.grid(True, alpha=0.3)
    ax_room.invert_yaxis()

    # Predictor center panel
    ax_predictor.set_xlim(-0.5, room_size - 0.5)
    ax_predictor.set_ylim(-0.5, room_size - 0.5)
    ax_predictor.set_aspect('equal')
    ax_predictor.set_xticks(range(room_size))
    ax_predictor.set_yticks(range(room_size))
    ax_predictor.tick_params(labelsize=11)
    ax_predictor.grid(True, alpha=0.3)
    ax_predictor.invert_yaxis()
    ax_predictor.set_title("Predictor: z + action → ẑ' → Decode", fontsize=15)

    # Probe panel
    ax_probe.set_xlim(-0.5, room_size - 0.5)
    ax_probe.set_ylim(-0.5, room_size - 0.5)
    ax_probe.set_aspect('equal')
    ax_probe.set_xticks(range(room_size))
    ax_probe.set_yticks(range(room_size))
    ax_probe.tick_params(labelsize=11)
    ax_probe.grid(True, alpha=0.3)
    ax_probe.invert_yaxis()
    ax_probe.set_title('Probe: z → Position', fontsize=15)

    # Initial strips
    draw_vector_strip(ax_encoder, z_seq[0].cpu().numpy().flatten(), 'Latent z (t=0)')
    if num_steps > 0:
        draw_vector_strip(ax_target, z_target_seq[0].cpu().numpy().flatten(), "Target z' (t=1)")

    def init():
        return []

    def animate(t):
        ax_room.clear()
        ax_predictor.clear()
        ax_probe.clear()

        for ax in [ax_room, ax_predictor, ax_probe]:
            ax.set_xlim(-0.5, room_size - 0.5)
            ax.set_ylim(-0.5, room_size - 0.5)
            ax.set_aspect('equal')
            ax.set_xticks(range(room_size))
            ax.set_yticks(range(room_size))
            ax.tick_params(labelsize=11)
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()

        # LEFT: room observation
        obs_t = obs_seq[t, 0].cpu().numpy()
        ax_room.imshow(obs_t, cmap='Blues', vmin=0, vmax=1)
        ax_room.set_title(f'Room at t={t}', fontsize=15)

        # Latent z strip (barh)
        z_t = z_seq[t].cpu().numpy().flatten()
        draw_vector_strip(ax_encoder, z_t, f'Latent z (t={t})')

        # CENTER: action + decoded prediction
        if t < num_steps:
            action = action_seq[t].item()
            action_names = ['Up', 'Down', 'Left', 'Right']
            ax_predictor.text(0.02, 0.98, f'Action: {action_names[action]}',
                            transform=ax_predictor.transAxes, fontsize=14,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            pred_room = z_pred_decoded[t, 0].cpu().numpy()
            ax_predictor.imshow(pred_room, cmap='Oranges', vmin=0, vmax=1, alpha=0.8)
            ax_predictor.set_title("Predicted Room at t+1", fontsize=15)

        # RIGHT TOP: probe
        pos_pred = pos_pred_seq[t].cpu().numpy()
        pos_true = pos_true_seq[t].cpu().numpy()
        ax_probe.plot(pos_true[0], pos_true[1], 'go', markersize=14, label='True')
        ax_probe.plot(pos_pred[0], pos_pred[1], 'rx', markersize=14, label='Probe')
        ax_probe.legend(fontsize=12, loc='upper right')
        ax_probe.set_title(f'Probe: True vs Pred (t={t})', fontsize=15)

        # RIGHT BOTTOM: target z' strip (barh)
        if t < num_steps:
            z_target_t = z_target_seq[t].cpu().numpy().flatten()
            draw_vector_strip(ax_target, z_target_t, f"Target z' (t={t+1})")

        return []

    ani = animation.FuncAnimation(fig, animate, frames=min(num_steps, len(z_seq)-1),
                                   init_func=init, blit=False, interval=1000/fps)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Pipeline animation saved to {output_path}")


def generate_all_visualizations(
    dataset: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    probe_results: dict,
    history: list,
    room_size: int = 8,
    hidden_dim: int = 32,
    steps_per_epoch: int = 12,
    output_dir: str = "images"
) -> None:
    """Master function to generate all visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    latent_dim = next(iter(encoder_params.values())).shape[0]

    print("Generating all visualizations...")

    # 1. Architecture diagram (hand-drawn U-shaped block diagram)
    plot_architecture(f"{output_dir}/architecture.png",
                      room_size=room_size, latent_dim=latent_dim, hidden_dim=hidden_dim)

    # 2. Training curves animated GIF (x-axis in epochs)
    plot_training_curves(history, f"{output_dir}/training_curves.gif",
                         steps_per_epoch=steps_per_epoch)

    # 3. Animated full pipeline (long horizon, most-mobile trajectory)
    plot_animated_pipeline(dataset, encoder_params, target_params, predictor_params, probe_results,
                            room_size=room_size, num_steps=15,
                            output_path=f"{output_dir}/pipeline_animation.gif")

    # 4. MPC trajectories
    plot_mpc_trajectories(encoder_params, predictor_params, room_size=room_size,
                           output_path=f"{output_dir}/mpc_trajectories.png")

    # 5. Probe accuracy
    plot_probe_accuracy(probe_results, dataset, encoder_params, f"{output_dir}/probe_accuracy.png")

    print(f"All visualizations saved to {output_dir}/")