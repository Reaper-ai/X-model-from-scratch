"""
Full JEPA World Model Experiment.
Step 38 from the JEPA implementation.
"""

import torch

from data import build_transition_dataset
from encoder import init_encoder_params, init_target_encoder
from predictor import init_predictor_params
from training import train_jepa
from evaluation import probe_state_recovery
from planning import evaluate_planner


def jepa_world_model_experiment(room_size, agent_size, embed_dim, n_train_transitions, n_epochs, batch_size, n_probe_samples, n_eval_episodes, max_steps, n_sequences, horizon):

    torch.manual_seed(0)

    dataset = build_transition_dataset(n_train_transitions, room_size=room_size, seed=0)
    encoder_params = init_encoder_params(room_size=room_size, latent_dim=embed_dim, seed=0)

    target_params = init_target_encoder(encoder_params)

    predictor_params = init_predictor_params(latent_dim=embed_dim, action_dim=4, hidden_dim=64, seed=0)

    encoder_params, target_params, predictor_params, history = train_jepa(
        dataset=dataset,
        encoder_params=encoder_params,
        target_params=target_params,
        predictor_params=predictor_params,
        num_steps=n_epochs,
        batch_size=batch_size,
        lr=1e-3, tau=0.99, seed=0
    )

    train_losses = [float(entry["loss"]) for entry in history]
    collapse_metrics = [float(entry["collapse"]) for entry in history]

    probe_dataset = {key: value[:n_probe_samples] for key, value in dataset.items()}

    probe_results = probe_state_recovery(
        dataset=probe_dataset,
        encoder_params=encoder_params,
        probe_params=None,
        num_probe_steps=100
    )

    # R^2 = 1 - MSE / (Var(states) + eps)
    states = probe_dataset["states"]

    state_variance = torch.mean(
        (states - states.mean(dim=0, keepdim=True)) ** 2
    )

    probe_mse = torch.as_tensor(
        probe_results["mse"],
        dtype=state_variance.dtype,
        device=state_variance.device
    )

    probe_r2 = 1.0 - (
        probe_mse / (state_variance + 1e-8)
    )

    planner_results = evaluate_planner(
        encoder_params=encoder_params,
        predictor_params=predictor_params,
        room_size=room_size,
        agent_size=agent_size,
        n_episodes=n_eval_episodes,
        max_steps=max_steps,
        n_sequences=n_sequences,
        horizon=horizon,
        n_actions=4
    )

    return {
        "train_losses": train_losses,
        "collapse_metrics": collapse_metrics,
        "probe_r2": float(probe_r2.item()),
        "success_rate": float(planner_results["success_rate"]),
        "mean_steps": float(planner_results["mean_steps"])
    }