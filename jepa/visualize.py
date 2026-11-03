"""
JEPA World Model - Visualization
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path
from typing import List, Dict, Any

from model import Encoder, TargetEncoder, Predictor, LinearProbe
from data_env import build_transition_dataset


OUTPUT_DIR = Path(__file__).parent.parent / 'docs' / 'images' / 'JEPA'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_architecture_diagram(save_path: str = None):
    """Draw JEPA architecture diagram: U-shape with L-shaped arrows and LaTeX math."""
    if save_path is None:
        save_path = OUTPUT_DIR / 'architecture.png'

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Colors
    encoder_color = '#3498db'
    target_color = '#2ecc71'
    predictor_color = '#e74c3c'
    probe_color = '#f39c12'
    arrow_color = '#2c3e50'
    text_color = '#2c3e50'

    # Larger font for data flow labels (matching JEPA loss text size)
    data_flow_fontsize = 11

    # Online Encoder (bottom left)
    enc_rect = Rectangle((0.5, 0.8), 2, 1.4, facecolor=encoder_color, alpha=0.2, edgecolor=encoder_color, linewidth=2)
    ax.add_patch(enc_rect)
    ax.text(1.5, 1.5, 'Online\nEncoder', ha='center', va='center', fontsize=14, fontweight='bold', color=text_color)
    ax.text(1.5, 0.4, 'CNN + FC\n$o_t \\rightarrow z_t$', ha='center', va='center', fontsize=10, color=encoder_color)

    # Target Encoder (bottom right)
    tgt_rect = Rectangle((7.5, 0.8), 2, 1.4, facecolor=target_color, alpha=0.2, edgecolor=target_color, linewidth=2)
    ax.add_patch(tgt_rect)
    ax.text(8.5, 1.5, 'Target\nEncoder (EMA)', ha='center', va='center', fontsize=14, fontweight='bold', color=text_color)
    ax.text(8.5, 0.4, 'EMA of Online\n$\\tau=0.996$', ha='center', va='center', fontsize=10, color=target_color)

    # EMA Arrow between encoders (horizontal, single-sided arrow pointing to target encoder)
    ema_arrow = FancyArrowPatch((2.5, 1.5), (7.5, 1.5),
                                 arrowstyle='-|>', mutation_scale=20,
                                 color=arrow_color, linewidth=2, linestyle='--')
    ax.add_patch(ema_arrow)
    ax.text(5, 1.0, 'EMA Update', ha='center', va='center', fontsize=12, fontweight='bold', color=arrow_color)

    # Predictor (middle - lowered)
    pred_rect = Rectangle((3.5, 4.2), 3, 1.5, facecolor=predictor_color, alpha=0.2, edgecolor=predictor_color, linewidth=2)
    ax.add_patch(pred_rect)
    ax.text(5, 4.95, 'Predictor', ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)
    ax.text(5, 4.4, '$z_t + a_t \\rightarrow z_{t+1}$', ha='center', va='center', fontsize=11, color=predictor_color)

    # L-shaped arrows from encoders to SIDES of predictor
    # Online encoder (left) -> LEFT side of predictor (3.5, 4.7)
    arrow1 = FancyArrowPatch((1.5, 2.2), (3.5, 4.7),
                              connectionstyle="angle,angleA=90,angleB=180,rad=10",
                              arrowstyle='->', mutation_scale=20,
                              color=arrow_color, linewidth=2)
    ax.add_patch(arrow1)
    ax.text(1.7, 3.0, '$z_t$', ha='center', va='center', fontsize=data_flow_fontsize, color=arrow_color)

    # Target encoder (right) -> RIGHT side of predictor (6.5, 4.7)
    arrow2 = FancyArrowPatch((8.5, 2.2), (6.5, 4.7),
                              connectionstyle="angle,angleA=90,angleB=0,rad=10",
                              arrowstyle='->', mutation_scale=20,
                              color=arrow_color, linewidth=2)
    ax.add_patch(arrow2)
    ax.text(8.2, 3.0, '$z_{t+1}^{\\text{target}}$', ha='center', va='center', fontsize=data_flow_fontsize, color=arrow_color)

    # Action input (straight down to top of predictor)
    ax.text(5, 6.2, 'Action $a_t$', ha='center', va='center', fontsize=12, fontweight='bold', color=arrow_color)
    action_arrow = FancyArrowPatch((5, 6.1), (5, 5.7),
                                    arrowstyle='->', mutation_scale=15,
                                    color=arrow_color, linewidth=2)
    ax.add_patch(action_arrow)

    # Linear Probe (moved up)
    probe_rect = Rectangle((7, 6.0), 2, 1.2, facecolor=probe_color, alpha=0.2, edgecolor=probe_color, linewidth=2)
    ax.add_patch(probe_rect)
    ax.text(7.9, 6.7, 'Linear\nProbe', ha='center', va='center', fontsize=14, fontweight='bold', color=text_color)
    ax.text(7.9, 6.3, '$z_{t+1} \\rightarrow (x, y)$', ha='center', va='center', fontsize=11, color=probe_color)

    # Arrow from predictor RIGHT to probe BOTTOM (L-shape: RIGHT then UP)
    probe_arrow = FancyArrowPatch((6.5, 4.95), (7.8, 6.0),
                                   connectionstyle="angle,angleA=0,angleB=90,rad=10",
                                   arrowstyle='->', mutation_scale=20,
                                   color=probe_color, linewidth=2, linestyle=':')
    ax.add_patch(probe_arrow)

    # Loss labels
    ax.text(5, 3.2, 'JEPA Loss = Prediction Loss + VICReg (Variance + Covariance)',
            ha='center', va='center', fontsize=11, fontweight='bold', color=text_color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=arrow_color, alpha=0.8))

    # Data flow labels - moved to SIDE to avoid arrow intersection
    # Observation o_t: left of online encoder (shifted right slightly, gap before encoder)
    ax.text(0, 1.5, 'Observation\n$o_t$', ha='right', va='center', fontsize=data_flow_fontsize, color=encoder_color)
    obs_arrow = FancyArrowPatch((0, 1.5), (0.5, 1.5),
                                 arrowstyle='->', mutation_scale=15,
                                 color=encoder_color, linewidth=2)
    ax.add_patch(obs_arrow)

    # Next Observation o_{t+1}: right of target encoder (gap between encoder and text, margin after)
    ax.text(10.0, 1.5, 'Next\nObservation\n$o_{t+1}$', ha='left', va='center', fontsize=data_flow_fontsize, color=target_color)
    obs_arrow2 = FancyArrowPatch((10.0, 1.5), (9.5, 1.5),
                                  arrowstyle='->', mutation_scale=15,
                                  color=target_color, linewidth=2)
    ax.add_patch(obs_arrow2)

    # Title
    ax.text(5, 8.0, 'JEPA World Model Architecture', ha='center', va='center', fontsize=18, fontweight='bold', color=text_color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Architecture diagram saved to {save_path}")


def create_animated_training_gif(history_path: str, save_path: str = None, fps: int = 10):
    """Create animated GIF of loss and collapse metric over training."""
    if save_path is None:
        save_path = OUTPUT_DIR / 'training_animation.gif'

    with open(history_path, 'r') as f:
        history = json.load(f)

    train_loss = history['train_loss']
    train_collapse = history['train_collapse']
    val_loss = history['val_loss']
    val_collapse = history['val_collapse']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Set up loss plot - use linear scale to avoid log issues
    ax1.set_xlim(0, len(train_loss))
    ax1.set_ylim(0, max(max(train_loss), max(val_loss)) * 1.1)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('JEPA Training Loss (Live)')
    ax1.grid(True, alpha=0.3)

    # Set up collapse plot
    ax2.set_xlim(0, len(train_collapse))
    ax2.set_ylim(0, max(max(train_collapse), max(val_collapse)) * 1.2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Collapse Metric')
    ax2.set_title('Collapse Metric (Live)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='g', linestyle='--', alpha=0.5)

    train_line, = ax1.plot([], [], 'b-', label='Train', linewidth=1)
    val_line, = ax1.plot([], [], 'r-', label='Val', linewidth=1.5)
    ax1.legend()

    collapse_train_line, = ax2.plot([], [], 'b-', label='Train', linewidth=1)
    collapse_val_line, = ax2.plot([], [], 'r-', label='Val', linewidth=1.5)
    ax2.legend()

    val_steps = [i * 50 for i in range(len(val_loss))]

    def init():
        train_line.set_data([], [])
        val_line.set_data([], [])
        collapse_train_line.set_data([], [])
        collapse_val_line.set_data([], [])
        return train_line, val_line, collapse_train_line, collapse_val_line

    def animate(frame):
        # Show every 10th frame for smoother animation
        step = frame * 10
        if step >= len(train_loss):
            step = len(train_loss) - 1

        train_line.set_data(range(step + 1), train_loss[:step + 1])

        # Val data up to current step
        val_idx = sum(1 for s in val_steps if s <= step)
        if val_idx > 0:
            val_line.set_data(val_steps[:val_idx], val_loss[:val_idx])

        collapse_train_line.set_data(range(step + 1), train_collapse[:step + 1])
        if val_idx > 0:
            collapse_val_line.set_data(val_steps[:val_idx], val_collapse[:val_idx])

        return train_line, val_line, collapse_train_line, collapse_val_line

    n_frames = len(train_loss) // 10 + 1
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                    interval=100, blit=True, repeat=True)

    try:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animated training GIF saved to {save_path}")
    except Exception as e:
        print(f"Failed to save animated GIF: {e}")
    finally:
        plt.close()


def draw_room_trajectory(ax, room_size: int, trajectory: List[Dict], goal: tuple,
                         start: tuple, title: str, success: bool, show_terminated: bool = True):
    """Draw a single room with trajectory."""
    ax.set_xlim(-0.5, room_size - 0.5)
    ax.set_ylim(-0.5, room_size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Match grid coordinates

    # Draw grid
    for i in range(room_size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)

    # Draw trajectory
    if trajectory:
        states = [start] + [t['state'] for t in trajectory]
        xs = [s[0] for s in states]
        ys = [s[1] for s in states]

        # Path line - GREEN for success, RED for failure
        line_color = 'green' if success else 'red'
        ax.plot(xs, ys, color=line_color, linewidth=2.5, alpha=0.8, zorder=2)

        # Path points with arrows
        for i in range(len(states) - 1):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            ax.arrow(xs[i], ys[i], dx * 0.6, dy * 0.6,
                     head_width=0.2, head_length=0.15, fc=line_color, ec=line_color,
                     alpha=0.8, zorder=3, length_includes_head=True)

    # Start position
    ax.plot(start[0], start[1], 'go', markersize=15, markeredgecolor='darkgreen',
            markeredgewidth=2, zorder=5, label='Start')

    # Goal position
    ax.plot(goal[0], goal[1], 'r*', markersize=20, markeredgecolor='darkred',
            markeredgewidth=2, zorder=5, label='Goal')

    # Terminated marker for unsuccessful
    if not success and show_terminated and trajectory:
        last_state = trajectory[-1]['state']
        ax.plot(last_state[0], last_state[1], 'rx', markersize=15, markeredgewidth=3,
                zorder=6, label='Terminated')

    ax.set_title(title, fontsize=11, fontweight='bold',
                 color='green' if success else 'red')
    ax.set_xticks(range(room_size))
    ax.set_yticks(range(room_size))
    ax.tick_params(axis='both', which='major', labelsize=8)


def plot_trajectory_grid(predictions_path: str, save_path: str = None, room_size: int = 6):
    """Plot 2x5 grid of trajectories: 5 successful top, 5 unsuccessful bottom."""
    if save_path is None:
        save_path = OUTPUT_DIR / 'trajectory_grid.png'

    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    successful = predictions['successful'][:5]
    unsuccessful = predictions['unsuccessful'][:5]

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle('Test Trajectories: Successful (Top) vs Unsuccessful (Bottom)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Top row: successful
    for i, traj in enumerate(successful):
        ax = axes[0, i]
        goal = traj['goal']
        # Initial state is first state in trajectory or (0,0) if empty
        if traj['trajectory']:
            start = (traj['trajectory'][0]['state'][0], traj['trajectory'][0]['state'][1])
        else:
            start = (0, 0)
        draw_room_trajectory(ax, room_size, traj['trajectory'], goal, start,
                             f'Success: Goal {goal}', traj['success'])

    # Bottom row: unsuccessful
    for i, traj in enumerate(unsuccessful):
        ax = axes[1, i]
        goal = traj['goal']
        if traj['trajectory']:
            start = (traj['trajectory'][0]['state'][0], traj['trajectory'][0]['state'][1])
        else:
            start = (0, 0)
        draw_room_trajectory(ax, room_size, traj['trajectory'], goal, start,
                             f'Failed: Goal {goal}', traj['success'], show_terminated=True)

    # Add legend to one subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, top=0.92, bottom=0.12)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Trajectory grid saved to {save_path}")


def run_all_visualizations(data_dir: str = None):
    """Run all visualizations and save to docs/images/JEPA/"""
    if data_dir is None:
        data_dir = Path(__file__).parent / 'data'

    data_dir = Path(data_dir)

    print("Generating JEPA visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Architecture diagram
    draw_architecture_diagram()

    # # 2. Animated training GIF
    # history_path = data_dir / 'history.json'
    # if history_path.exists():
    #     create_animated_training_gif(history_path)

    # # 3. Trajectory grid
    # predictions_path = data_dir / 'test_predictions.json'
    # if predictions_path.exists():
    #     plot_trajectory_grid(predictions_path)

    # print("\nAll visualizations complete!")


if __name__ == '__main__':
    run_all_visualizations()