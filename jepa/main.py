"""
JEPA World Model - Main Orchestration
"""

import torch
import numpy as np
from training_eval import jepa_world_model_experiment


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 60)
    print("JEPA World Model Experiment")
    print("=" * 60)

    results = jepa_world_model_experiment(
        room_size=6,
        latent_dim=16,
        n_train_transitions=512,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=32,
        num_steps=1500,
        lr=3e-4,
        tau=0.996,
        seed=0,
        n_probe_samples=512,
        num_probe_steps=300,
        n_eval_episodes=10,
        max_steps=20,
        n_sequences=64,
        horizon=5,
        save_dir='./jepa/data',
        device='cpu',
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4
    )

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print(f"Results saved to ./jepa/data/")


if __name__ == "__main__":
    main()