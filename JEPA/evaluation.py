"""
Evaluation: Latent Rollouts, Multi-step Prediction, Linear Probe.
Steps 25-29 from the JEPA implementation.
"""

import torch
import torch.nn.functional as F

from encoder import encode_batch
from predictor import predict_next_embedding


def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    flag = initial_embedding.dim() == 1

    if flag:
        initial_embedding = initial_embedding.unsqueeze(0)

    if actions.dim() == 1:
        actions = actions.unsqueeze(0).expand(
            initial_embedding.shape[0], -1
        )
    current = initial_embedding
    rollout = [current]
    for t in range(actions.shape[1]):
        current = predict_next_embedding(current, actions[:, t], predictor_params)
        rollout.append(current)

    rollout = torch.stack(rollout, dim=0)
    if flag:
        rollout = rollout.squeeze(1)
    return rollout


def multi_step_prediction_error(
    dataset: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    horizon: int = 5,
    num_samples: int = 32
) -> float:

    N = len(dataset["actions"])
    num_samples = min(num_samples, N - horizon + 1)

    indices = torch.arange(num_samples)

    obs = dataset["observations"][indices]
    actions = dataset["actions"][indices[:, None] + torch.arange(horizon)]
    future_obs = dataset["next_observations"][
        indices[:, None] + torch.arange(horizon)
    ]

    embeddings = encode_batch(obs, encoder_params)

    rollout = rollout_latent_dynamics(
        embeddings, actions, predictor_params
    )[1:]

    B, H, C, W, W2 = future_obs.shape
    with torch.no_grad():
        target = encode_batch(
            future_obs.reshape(B * H, C, W, W2),
            target_params
        )

    target = target.reshape(B, H, -1).transpose(0, 1)

    error = (rollout - target).pow(2).sum(dim=-1).mean()

    return error.item()


def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0, device: str = 'cpu') -> dict:
    torch.manual_seed(seed)
    probe = {}
    probe["w"] = torch.randn(state_dim, latent_dim, device=device) * 0.01
    probe["b"] = torch.zeros(state_dim, device=device)

    return probe


def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    embeddings = embeddings.detach()
    device = embeddings.device

    w = probe_params['w'].clone().detach().requires_grad_(True)
    b = probe_params['b'].clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        pred = embeddings @ w.T + b
        loss = torch.mean((pred - states) ** 2)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None
        b.grad = None

    probe_params = {'w': w.detach().clone(), 'b': b.detach().clone()}
    return probe_params


def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    obs = dataset["observations"]
    states = dataset["states"]

    embedding = encode_batch(obs, encoder_params)
    D = embedding.shape[1]
    device = embedding.device
    if probe_params is None:
        probe_params = init_linear_probe(latent_dim=D, state_dim=states.shape[-1], seed=0, device=device)

    probe_params = train_linear_probe(embedding, states, probe_params, num_probe_steps)

    predicted_states = embedding @ probe_params["w"].T + probe_params["b"]
    mse = F.mse_loss(predicted_states, states).item()
    mae = F.l1_loss(predicted_states, states).item()

    results = {}
    results["mse"] = mse
    results["mean_abs_error"] = mae
    results["probe_params"] = probe_params

    return results