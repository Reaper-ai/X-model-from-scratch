"""
JEPA World Model from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of an action-conditioned JEPA world model (LeCun-style).

Story: an untrained encoder gives COLLAPSED representations (tiny std) --
prediction in latent space is trivially easy but useless. Training with the
JEPA objective (prediction + variance/covariance regularization) produces
informative latents: a linear probe recovers the agent position, and
random-shooting MPC over the learned latent dynamics reaches goals far more
often than a random policy.
"""
import numpy as np
import torch


def random_policy_baseline(room_size, n_episodes, max_steps, seed=123):
    """Success rate of uniformly random actions (the bar MPC must beat)."""
    rng = np.random.default_rng(seed)
    successes = 0
    for ep in range(n_episodes):
        goal = rng.integers(0, room_size, size=2)
        state, obs = env_reset(room_size=room_size, seed=1000 + ep)
        for _ in range(max_steps):
            if int(state[0].item()) == int(goal[0]) and int(state[1].item()) == int(goal[1]):
                break
            state, obs = env_step(state, int(rng.integers(0, 4)), room_size=room_size)
        if int(state[0].item()) == int(goal[0]) and int(state[1].item()) == int(goal[1]):
            successes += 1
    return successes / n_episodes


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    room_size = 6
    latent_dim = 16
    seed = 0

    # ---- 1. Data ----
    dataset = build_transition_dataset(num_transitions=512, room_size=room_size, seed=seed)
    print("transitions:", int(dataset["actions"].shape[0]), "| obs shape:", tuple(dataset["observations"].shape))

    enc = init_encoder_params(obs_channels=1, room_size=room_size, latent_dim=latent_dim, seed=seed)
    tgt = init_target_encoder(enc)
    pred = init_predictor_params(latent_dim=latent_dim, action_dim=4, hidden_dim=32, seed=seed)

    # ---- 2. Before training: the collapse problem ----
    emb0 = encode_batch(dataset["observations"][:64].float(), enc)
    print(f"collapse metric BEFORE training: {float(collapse_metric(emb0)):.3f}  (near 0 = collapsed)")

    # ---- 3. Train the JEPA ----
    enc, tgt, pred, hist = train_jepa(
        dataset, enc, tgt, pred,
        num_steps=1500, batch_size=32, lr=0.1, tau=0.99, seed=seed,
    )
    print(f"JEPA loss: {hist[0]['loss']:.3f} -> {hist[-1]['loss']:.3f}")

    emb1 = encode_batch(dataset["observations"][:64].float(), enc)
    print(f"collapse metric AFTER training:  {float(collapse_metric(emb1)):.3f}  (target std is 1.0)")

    # ---- 4. The world model: multi-step latent prediction ----
    err = multi_step_prediction_error(dataset, enc, tgt, pred, horizon=5, num_samples=32)
    print(f"5-step latent prediction MSE (trained): {err:.3f}")
    print("  (an untrained, collapsed encoder scores ~0 here -- trivially easy and useless)")

    # ---- 5. No decoder needed: a linear probe recovers the state ----
    probe = probe_state_recovery(dataset, enc, probe_params=None, num_probe_steps=300)
    print(f"linear probe on agent position: mean abs error = {probe['mean_abs_error']:.2f} cells (room is {room_size}x{room_size})")

    # ---- 6. Acting = planning in latent space (no learned policy) ----
    stats = evaluate_planner(
        enc, pred, room_size=room_size, agent_size=1,
        n_episodes=10, max_steps=20, n_sequences=64, horizon=5, n_actions=4,
    )
    rnd = random_policy_baseline(room_size, n_episodes=10, max_steps=20)
    print(f"MPC planner:   success {stats['success_rate']:.0%}, mean final distance {stats['mean_final_distance']:.2f}")
    print(f"random policy: success {rnd:.0%}")


if __name__ == "__main__":
    main()

