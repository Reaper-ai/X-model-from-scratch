"""
Train JEPA, evaluate probes, save all metrics/params to disk.
Visualizations are a separate step: run visualize.py after this.
"""

import os
import torch
import numpy as np

from data import build_transition_dataset, split_dataset
from encoder import init_encoder_params, init_target_encoder
from predictor import init_predictor_params
from training import train_jepa
from evaluation import probe_state_recovery
from planning import evaluate_planner


CONFIG = {
    "room_size": 6,
    "latent_dim": 16,
    "hidden_dim": 32,
    "max_horizon": 20,
    "eval_horizons": [5, 10, 20],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_transitions": 512,       # cached on disk; big enough to converge, tiny in RAM
    "max_epochs": 100,           # fixed length (no early stopping)
    "val_every_epochs": 1,       # validate once per epoch
    "horizon_every_epochs": 30,  # advance curriculum horizon every 30 epochs
    "batch_size": 32,
    "lr": 0.1,
    "tau": 0.99,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 0,
    "results_dir": "results",
}


def to_cpu(d: dict) -> dict:
    return {k: v.detach().cpu() for k, v in d.items()}


def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = cfg["device"]
    os.makedirs(cfg["results_dir"], exist_ok=True)

    print(f"Using device: {device}")
    room_size = cfg["room_size"]
    print(f"Room: {room_size}x{room_size}, transitions: {cfg['num_transitions']}")

    print("=== Building dataset (disk cache) ===")
    full_dataset = build_transition_dataset(
        num_transitions=cfg["num_transitions"],
        room_size=room_size,
        diverse=True,
        cache_dir="data_cache",
        device=device,
    )

    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        device=device,
    )
    print(f"Train: {train_dataset['observations'].shape[0]}, "
          f"Val: {val_dataset['observations'].shape[0]}, "
          f"Test: {test_dataset['observations'].shape[0]}")

    print("=== Initializing models ===")
    encoder_params = init_encoder_params(
        obs_channels=1, room_size=room_size,
        latent_dim=cfg["latent_dim"], seed=cfg["seed"], device=device,
    )
    target_params = init_target_encoder(encoder_params)
    predictor_params = init_predictor_params(
        latent_dim=cfg["latent_dim"], action_dim=4,
        hidden_dim=cfg["hidden_dim"], seed=cfg["seed"], device=device,
    )

    # 1 epoch = one pass over the train set (ceil of N/batch_size updates)
    train_n = train_dataset["observations"].shape[0]
    steps_per_epoch = max(1, -(-train_n // cfg["batch_size"]))  # ceil division
    max_steps = cfg["max_epochs"] * steps_per_epoch
    val_every = cfg["val_every_epochs"] * steps_per_epoch
    print(f"Epochs: {cfg['max_epochs']} x {steps_per_epoch} steps = {max_steps} steps total; "
          f"no early stopping; "
          f"val every {val_every} steps; "
          f"horizon every {cfg['horizon_every_epochs']} epochs")

    print("=== Training JEPA ===")
    encoder_params, target_params, predictor_params, history = train_jepa(
        dataset=train_dataset,
        encoder_params=encoder_params,
        target_params=target_params,
        predictor_params=predictor_params,
        num_steps=max_steps,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        tau=cfg["tau"],
        seed=cfg["seed"],
        curriculum=True,
        max_horizon=cfg["max_horizon"],
        steps_per_epoch=steps_per_epoch,
        horizon_every_epochs=cfg["horizon_every_epochs"],
        val_dataset=val_dataset,
        val_every=val_every,
        patience=None,  # early stopping disabled
    )
    print(f"Steps run: {len(history)} ({len(history) / steps_per_epoch:.1f} epochs)")
    print(f"Train loss: {history[0]['train_loss']:.4f} -> {history[-1]['train_loss']:.4f}")
    print(f"Collapse: {history[0]['collapse']:.4f} -> {history[-1]['collapse']:.4f}")
    val_losses = [h["val_loss"] for h in history if "val_loss" in h]
    if val_losses:
        print(f"Val loss: {val_losses[0]:.4f} -> {min(val_losses):.4f} (best)")

    print("=== Linear probe (test set) ===")
    probe_results = probe_state_recovery(
        dataset=test_dataset, encoder_params=encoder_params,
        probe_params=None, num_probe_steps=200,
    )
    print(f"Test Probe MAE: {probe_results['mean_abs_error']:.3f}")

    print("=== Probe train/val ===")
    probe_metrics = {"test": probe_results["mean_abs_error"]}
    for name, ds in [("train", train_dataset), ("val", val_dataset)]:
        pr = probe_state_recovery(ds, encoder_params, probe_params=None, num_probe_steps=200)
        probe_metrics[name] = pr["mean_abs_error"]
        print(f"  {name} MAE: {pr['mean_abs_error']:.3f}")

    print("=== MPC planner eval ===")
    mpc_metrics = {}
    for h in cfg["eval_horizons"]:
        m = evaluate_planner(
            encoder_params, predictor_params, room_size=room_size,
            agent_size=1, n_episodes=10, max_steps=20,
            n_sequences=64, horizon=h, n_actions=4,
        )
        mpc_metrics[h] = m
        print(f"  horizon={h}: success={m['success_rate']:.0%}, "
              f"steps={m['mean_steps']:.1f}, dist={m['mean_final_distance']:.2f}")

    # Save everything to disk (CPU tensors)
    out = {
        "config": cfg,
        "history": history,
        "probe_metrics": probe_metrics,
        "mpc_metrics": mpc_metrics,
        "encoder_params": to_cpu(encoder_params),
        "target_params": to_cpu(target_params),
        "predictor_params": to_cpu(predictor_params),
        "probe_params": {k: v.cpu() for k, v in probe_results["probe_params"].items()},
    }
    path = os.path.join(cfg["results_dir"], "jepa_results.pt")
    torch.save(out, path)
    print(f"\nSaved results to {path}")
    print("Run visualize.py next to generate plots/GIFs.")


if __name__ == "__main__":
    main()
