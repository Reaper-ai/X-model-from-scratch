"""
Test script to run JEPA experiment and generate all visualizations.
Run with: python test_visualizations.py
"""

import torch
import numpy as np

from data import build_transition_dataset
from encoder import init_encoder_params, init_target_encoder
from predictor import init_predictor_params
from training import train_jepa
from evaluation import probe_state_recovery
from planning import evaluate_planner
from analysis import generate_all_visualizations


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    room_size = 8
    latent_dim = 32
    seed = 0

    print("=== Building dataset ===")
    dataset = build_transition_dataset(num_transitions=2048, room_size=room_size, seed=seed)
    print(f"Dataset: {dataset['observations'].shape[0]} transitions")

    print("=== Initializing models ===")
    encoder_params = init_encoder_params(obs_channels=1, room_size=room_size, latent_dim=latent_dim, seed=seed)
    target_params = init_target_encoder(encoder_params)
    predictor_params = init_predictor_params(latent_dim=latent_dim, action_dim=4, hidden_dim=64, seed=seed)

    print("=== Training JEPA ===")
    encoder_params, target_params, predictor_params, history = train_jepa(
        dataset=dataset,
        encoder_params=encoder_params,
        target_params=target_params,
        predictor_params=predictor_params,
        num_steps=500,  # Reduced for testing
        batch_size=64,
        lr=1e-3, tau=0.99, seed=seed,
    )
    print(f"Loss: {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}")

    print("=== Linear probe ===")
    probe_dataset = {k: v[:512] for k, v in dataset.items()}
    probe_results = probe_state_recovery(
        dataset=probe_dataset,
        encoder_params=encoder_params,
        probe_params=None,
        num_probe_steps=200
    )
    print(f"Probe MAE: {probe_results['mean_abs_error']:.3f}")

    print("=== Generating visualizations ===")
    generate_all_visualizations(
        dataset=dataset,
        encoder_params=encoder_params,
        target_params=target_params,
        predictor_params=predictor_params,
        probe_results=probe_results,
        history=history,
        room_size=room_size,
        output_dir="images"
    )

    print("=== Done! ===")


if __name__ == "__main__":
    main()