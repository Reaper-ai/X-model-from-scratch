"""
Analysis and Visualization Functions for JEPA World Model.

Generates:
- Architecture diagram (matplotlib -> SVG/PNG)
- Training curves animated GIF (matplotlib + imageio)
- Latent prediction comparison grid (3-panel: GT | Pred | Target)
- MPC trajectory visualization
- Linear probe accuracy plots
"""

import os
import torch
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
from predictor import predict_next_embedding
from evaluation import rollout_latent_dynamics
from planning import run_mpc_episode

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


def plot_architecture(output_path: str = "images/architecture.png") -> None:
    """Generate JEPA architecture diagram using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.suptitle('JEPA World Model Architecture', fontsize=16, fontweight='bold', y=0.95)

    # Colors
    c_obs = '#e3f2fd'
    c_act = '#fff3e0'
    c_enc = '#e8f5e9'
    c_tgt = '#f3e5f5'
    c_pred = '#fff8e1'
    c_loss = '#ffebee'
    c_out = '#e0f7fa'
    c_pred_out = '#fff3e0'

    # Helper to draw boxes
    def draw_box(x, y, w, h, text, fill, edge, fontsize=9, dashed=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=fill, edgecolor=edge, linewidth=2,
                             linestyle='--' if dashed else '-')
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

    # Helper to draw arrows
    def draw_arrow(x1, y1, x2, y2, label='', color='black', dashed=False):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='->', mutation_scale=15,
                                color=color, linewidth=1.5,
                                linestyle='--' if dashed else '-')
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha='center', va='bottom', fontsize=8, color=color)

    # === Input Layer ===
    draw_box(0.5, 2.5, 2.5, 1.2, 'Observation\n(1, 8, 8)', c_obs, '#1976d2')
    draw_box(0.5, 0.8, 2.5, 1.0, 'Action\n(0-3)', c_act, '#f57c00')

    # === Encoder ===
    draw_box(3.5, 2.0, 3.0, 2.0,
             'Encoder\nConv2d(1→16, 3×3, s=1)\nReLU\nConv2d(16→32, 3×3, s=2)\nReLU\nFlatten\nLinear(32×4×4 → 32)',
             c_enc, '#388e3c', fontsize=8)

    # === Target Encoder (EMA) ===
    draw_box(3.5, -0.2, 3.0, 1.5,
             'Target Encoder (EMA)\nτ = 0.99\n(no gradients)',
             c_tgt, '#7b1fa2', fontsize=8, dashed=True)

    # === Predictor ===
    draw_box(7.5, 1.0, 3.5, 2.5,
             'Predictor\nAction Embedding (4→32)\nConcat [z, a_embed] → (64,)\nLinear(64→64) + ReLU\nLinear(64→32)',
             c_pred, '#fbc02d', fontsize=8)

    # === Losses ===
    draw_box(11.5, 3.2, 3.0, 1.2, 'Prediction Loss\nMSE(pred, target)', c_loss, '#c62828', fontsize=8)
    draw_box(11.5, 1.8, 3.0, 1.2, 'Variance Loss\nmax(0, γ - std(z))', c_loss, '#c62828', fontsize=8)
    draw_box(11.5, 0.4, 3.0, 1.2, 'Covariance Loss\noff-diag(cov(z))²', c_loss, '#c62828', fontsize=8)
    draw_box(11.5, -1.0, 3.0, 1.0, 'JEPA Loss\npred + var + cov', '#b71c1c', '#fff', fontsize=9)

    # === Outputs ===
    draw_box(7.5, 4.0, 3.5, 1.0, 'Latent z (online)\n(32,)', c_out, '#0097a7', fontsize=9)
    draw_box(7.5, -1.0, 3.5, 1.0, 'Predicted z\'\n(32,)', c_pred_out, '#f57c00', fontsize=9)
    draw_box(3.5, 4.5, 3.0, 1.0, 'Target z\'\n(next obs)', c_out, '#0097a7', fontsize=9)

    # === Edges ===
    # Obs -> Encoder
    draw_arrow(3.0, 3.1, 3.5, 3.1)
    # Action -> Predictor (embed)
    draw_arrow(3.0, 1.3, 5.5, 2.5, 'embed', '#f57c00')
    # Encoder -> z (online)
    draw_arrow(6.5, 3.0, 7.5, 4.3)
    # z (online) -> Predictor
    draw_arrow(9.0, 3.5, 9.0, 2.5)  # vertical down
    # Predictor -> Pred z'
    draw_arrow(11.0, 2.0, 11.5, 2.0)
    # Next obs -> Target Encoder (dashed)
    draw_arrow(3.0, 3.1, 3.5, 1.0, 'next_obs', '#7b1fa2', dashed=True)
    # Target Encoder -> Target z'
    draw_arrow(6.5, 0.5, 7.5, 4.5, '', '#7b1fa2', dashed=True)
    # Target z' -> Pred Loss
    draw_arrow(6.5, 4.5, 11.5, 3.8, 'target', '#c62828', dashed=True)
    # Pred z' -> Pred Loss
    draw_arrow(11.0, 2.0, 11.5, 3.8, 'pred', '#c62828')
    # z (online) -> Var Loss
    draw_arrow(7.5, 4.0, 11.5, 2.4, '', '#c62828')
    # z (online) -> Cov Loss
    draw_arrow(7.5, 4.0, 11.5, 1.0, '', '#c62828')
    # Losses -> JEPA Loss
    draw_arrow(12.5, 3.2, 13.0, -0.5)
    draw_arrow(12.5, 1.8, 13.0, -0.5)
    draw_arrow(12.5, 0.4, 13.0, -0.5)

    # EMA update arrow (dotted)
    draw_arrow(5.0, 4.0, 5.0, 0.8, 'EMA update', '#7b1fa2', dashed=True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=c_obs, edgecolor='#1976d2', label='Input'),
        mpatches.Patch(facecolor=c_enc, edgecolor='#388e3c', label='Online Encoder'),
        mpatches.Patch(facecolor=c_tgt, edgecolor='#7b1fa2', label='Target Encoder (EMA)', linestyle='--'),
        mpatches.Patch(facecolor=c_pred, edgecolor='#fbc02d', label='Predictor'),
        mpatches.Patch(facecolor=c_loss, edgecolor='#c62828', label='Loss Components'),
        mpatches.Patch(facecolor=c_out, edgecolor='#0097a7', label='Latent Outputs'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05), fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Architecture diagram saved to {output_path}")


def plot_training_curves(history: list, output_path: str = "images/training_curves.gif", fps: int = 10) -> None:
    """Generate animated GIF of training loss and collapse metric over epochs."""
    losses = [h['loss'] for h in history]
    collapses = [h['collapse'] for h in history]
    epochs = list(range(1, len(history) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('JEPA Training Progress', fontsize=14, fontweight='bold')

    # Setup axes
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('JEPA Loss')
    ax1.set_xlim(0, len(history))
    ax1.set_ylim(0, max(losses) * 1.1 if losses else 1)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Collapse Metric (mean std)')
    ax2.set_title('Collapse Metric (target ≈ 1.0)')
    ax2.set_xlim(0, len(history))
    ax2.set_ylim(0, max(collapses) * 1.1 if collapses else 1.5)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target (γ=1.0)')
    ax2.legend(fontsize=9)

    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Loss')
    line2, = ax2.plot([], [], 'r-', linewidth=2, label='Collapse')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        line1.set_data(epochs[:i+1], losses[:i+1])
        line2.set_data(epochs[:i+1], collapses[:i+1])
        return line1, line2

    ani = animation.FuncAnimation(fig, animate, frames=len(history),
                                   init_func=init, blit=True, interval=1000/fps)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Training curves GIF saved to {output_path}")


def plot_latent_prediction_grid(
    dataset: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    horizon: int = 5,
    num_samples: int = 4,
    output_path: str = "images/prediction_comparison.png"
) -> None:
    """
    3-panel grid visualization:
    Left: Ground truth next observations (8x8 grids)
    Center: Predicted latent embeddings (heatmaps)
    Right: Target encoder embeddings of actual next observations (heatmaps)
    """
    N = min(num_samples, len(dataset['actions']) - horizon)
    indices = torch.arange(N)

    obs = dataset['observations'][indices]
    actions = dataset['actions'][indices[:, None] + torch.arange(horizon)]
    future_obs = dataset['next_observations'][indices[:, None] + torch.arange(horizon)]

    # Encode initial observations
    with torch.no_grad():
        embeddings = encode_batch(obs, encoder_params)

        # Rollout predictions
        rollout = rollout_latent_dynamics(embeddings, actions, predictor_params)[1:]  # (H, B, D)

        # Target embeddings of actual future observations
        B, H, C, W, W2 = future_obs.shape
        target = encode_batch(future_obs.reshape(B * H, C, W, W2), target_params)
        target = target.reshape(B, H, -1).transpose(0, 1)  # (H, B, D)

    # Create figure: 3 columns (GT obs | Pred latent | Target latent), H rows per sample
    fig, axes = plt.subplots(N * horizon, 3, figsize=(9, 3 * N * horizon))
    if N * horizon == 1:
        axes = axes.reshape(1, 3)

    fig.suptitle(f'Latent Prediction Comparison (horizon={horizon})', fontsize=14, fontweight='bold')

    for sample_idx in range(N):
        for h in range(horizon):
            row = sample_idx * horizon + h

            # Left: Ground truth observation (8x8 grid)
            gt_obs = future_obs[sample_idx, h, 0].cpu().numpy()
            im1 = axes[row, 0].imshow(gt_obs, cmap='Blues', vmin=0, vmax=1)
            axes[row, 0].set_title(f'Sample {sample_idx}, Step {h+1}\nGT Observation')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            if h == 0 and sample_idx == 0:
                axes[row, 0].set_ylabel('Ground Truth\n(8×8 grid)', fontsize=10)

            # Center: Predicted latent (heatmap of 32-dim vector)
            pred_latent = rollout[h, sample_idx].cpu().numpy()
            im2 = axes[row, 1].imshow(pred_latent.reshape(1, -1), cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
            axes[row, 1].set_title(f'Predicted Latent z\'')
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])
            if h == 0 and sample_idx == 0:
                axes[row, 1].set_ylabel('Predicted\n(32-dim)', fontsize=10)

            # Right: Target latent (heatmap of 32-dim vector)
            target_latent = target[h, sample_idx].cpu().numpy()
            im3 = axes[row, 2].imshow(target_latent.reshape(1, -1), cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
            axes[row, 2].set_title(f'Target Latent z\'')
            axes[row, 2].set_xticks([])
            axes[row, 2].set_yticks([])
            if h == 0 and sample_idx == 0:
                axes[row, 2].set_ylabel('Target (EMA)\n(32-dim)', fontsize=10)

    # Add colorbars
    plt.colorbar(im2, ax=axes[:, 1], shrink=0.6, label='Latent Value')
    plt.colorbar(im3, ax=axes[:, 2], shrink=0.6, label='Latent Value')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Prediction comparison grid saved to {output_path}")


def plot_mpc_trajectories(
    encoder_params: dict,
    predictor_params: dict,
    room_size: int = 8,
    n_episodes: int = 10,
    max_steps: int = 20,
    n_sequences: int = 64,
    horizon: int = 5,
    n_actions: int = 4,
    output_path: str = "images/mpc_trajectories.png"
) -> None:
    """Visualize MPC agent trajectories in the 2D room."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    fig.suptitle('MPC Planner Trajectories in 2D Room', fontsize=14, fontweight='bold')

    for ep in range(min(n_episodes, 10)):
        ax = axes[ep]
        goal_pos = torch.randint(0, room_size, (2,)).float()

        result = run_mpc_episode(
            encoder_params, predictor_params, goal_pos,
            room_size, agent_size=1, max_steps=max_steps,
            n_sequences=n_sequences, horizon=horizon, n_actions=n_actions
        )

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

        ax.set_title(f'Episode {ep+1}: {"Success" if success else "Failed"}')
        if ep == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for ep in range(n_episodes, 10):
        axes[ep].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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

    with torch.no_grad():
        embeddings = encode_batch(obs, encoder_params)
        predicted_states = embeddings @ probe_results['probe_params']['w'].T + probe_results['probe_params']['b']

    pred_np = predicted_states.cpu().numpy()
    true_np = states.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Linear Probe: Latent → Agent Position', fontsize=14, fontweight='bold')

    # X coordinate: predicted vs actual
    axes[0].scatter(true_np[:, 0], pred_np[:, 0], alpha=0.6, s=20, c='blue')
    axes[0].plot([0, 7], [0, 7], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True X Position')
    axes[0].set_ylabel('Predicted X Position')
    axes[0].set_title(f'X Coordinate (MAE: {probe_results["mean_abs_error"]:.3f})')
    axes[0].legend()
    axes[0].set_xlim(-0.5, 7.5)
    axes[0].set_ylim(-0.5, 7.5)
    axes[0].set_aspect('equal')

    # Y coordinate: predicted vs actual
    axes[1].scatter(true_np[:, 1], pred_np[:, 1], alpha=0.6, s=20, c='green')
    axes[1].plot([0, 7], [0, 7], 'r--', label='Perfect prediction')
    axes[1].set_xlabel('True Y Position')
    axes[1].set_ylabel('Predicted Y Position')
    axes[1].set_title(f'Y Coordinate (MAE: {probe_results["mean_abs_error"]:.3f})')
    axes[1].legend()
    axes[1].set_xlim(-0.5, 7.5)
    axes[1].set_ylim(-0.5, 7.5)
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


def generate_all_visualizations(
    dataset: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    probe_results: dict,
    history: list,
    room_size: int = 8,
    output_dir: str = "images"
) -> None:
    """Master function to generate all visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating all visualizations...")

    # 1. Architecture diagram
    plot_architecture(f"{output_dir}/architecture.png")

    # 2. Training curves animated GIF
    plot_training_curves(history, f"{output_dir}/training_curves.gif")

    # 3. Latent prediction comparison grid
    plot_latent_prediction_grid(dataset, encoder_params, target_params, predictor_params,
                                 horizon=5, num_samples=4, output_path=f"{output_dir}/prediction_comparison.png")

    # 4. MPC trajectories
    plot_mpc_trajectories(encoder_params, predictor_params, room_size=room_size,
                           output_path=f"{output_dir}/mpc_trajectories.png")

    # 5. Probe accuracy
    plot_probe_accuracy(probe_results, dataset, encoder_params, f"{output_dir}/probe_accuracy.png")

    print(f"All visualizations saved to {output_dir}/")