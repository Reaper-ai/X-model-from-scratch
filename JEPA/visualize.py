"""
Load saved results from train.py and generate visualizations.
Run train.py first. All plots/GIFs go to images/.
"""

import os
import torch

from data import build_transition_dataset
from analysis import generate_all_visualizations


RESULTS_PATH = "results/jepa_results.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Loading results from {RESULTS_PATH} (device={DEVICE})")
    out = torch.load(RESULTS_PATH, map_location=DEVICE, weights_only=False)
    cfg = out["config"]

    encoder_params = {k: v.to(DEVICE) for k, v in out["encoder_params"].items()}
    target_params = {k: v.to(DEVICE) for k, v in out["target_params"].items()}
    predictor_params = {k: v.to(DEVICE) for k, v in out["predictor_params"].items()}
    probe_results = {
        "mean_abs_error": out["probe_metrics"]["test"],
        "probe_params": {k: v.to(DEVICE) for k, v in out["probe_params"].items()},
    }
    history = out["history"]
    room_size = cfg["room_size"]

    # Load full dataset from disk cache (small: 512 x 1 x 6 x 6)
    print("Loading dataset from cache...")
    full_dataset = build_transition_dataset(
        num_transitions=cfg["num_transitions"],
        room_size=room_size,
        diverse=True,
        cache_dir="data_cache",
        device=DEVICE,
    )

    # 1 epoch = ceil(train_n / batch_size) steps (same formula as train.py)
    train_n = int(len(full_dataset["observations"]) * cfg["train_ratio"])
    steps_per_epoch = max(1, -(-train_n // cfg["batch_size"]))

    generate_all_visualizations(
        dataset=full_dataset,
        encoder_params=encoder_params,
        target_params=target_params,
        predictor_params=predictor_params,
        probe_results=probe_results,
        history=history,
        room_size=room_size,
        hidden_dim=cfg.get("hidden_dim", 32),
        steps_per_epoch=steps_per_epoch,
        output_dir="images",
    )
    print("=== Done! ===")


if __name__ == "__main__":
    main()
