"""
Training Step and Full Training Loop.
Steps 23-24 from the JEPA implementation.
"""

import torch

from encoder import encode_batch
from predictor import predict_next_embedding
from losses import jepa_loss, collapse_metric
from encoder import ema_update


def jepa_training_step(
    batch: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    lr: float = 1e-3,
    tau: float = 0.99
) -> tuple[dict, dict, dict, float, float]:

    obs = batch["observations"]
    actions = batch["actions"]
    next_obs = batch["next_observations"]

    # 1. Enable gradients and clear previous gradients
    for params in (encoder_params, predictor_params):
        for param in params.values():
            param.requires_grad_(True)
            param.grad = None

    # 2. Online encoder
    embeddings = encode_batch(obs, encoder_params)

    # 3. Target encoder (no gradient)
    with torch.no_grad():
        target = encode_batch(next_obs, target_params)

    # 4. Predictor
    predicted = predict_next_embedding(
        embeddings, actions, predictor_params
    )

    # 5. JEPA loss
    loss = jepa_loss(predicted, target, embeddings)

    # 6. Backpropagation
    loss.backward(retain_graph=True)

    # 7. Update online encoder
    updated_encoder = {}

    for name, param in encoder_params.items():
        if param.grad is None:
            updated_encoder[name] = param.detach().requires_grad_(True)
        else:
            updated_encoder[name] = (
                param - lr * param.grad
            ).detach().requires_grad_(True)

    # 8. Update predictor
    updated_predictor = {}

    for name, param in predictor_params.items():
        if param.grad is None:
            updated_predictor[name] = param.detach().requires_grad_(True)
        else:
            updated_predictor[name] = (
                param - lr * param.grad
            ).detach().requires_grad_(True)

    # 9. EMA target update
    updated_target = ema_update(
        target_params,
        updated_encoder,
        tau=tau
    )

    # 10. Collapse metric
    collapse = collapse_metric(embeddings.detach())

    return (
        updated_encoder,
        updated_target,
        updated_predictor,
        loss.item(),
        collapse.item()
    )


def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    torch.manual_seed(seed)
    N = dataset["observations"].shape[0]

    enc = encoder_params
    tgt = target_params
    pred = predictor_params
    history = []
    for i in range(num_steps):
        idx = torch.randint(0, N, (batch_size,))
        batch = {"observations": dataset["observations"][idx],
            "actions": dataset["actions"][idx],
            "next_observations": dataset["next_observations"][idx]}
        enc, tgt, pred, loss, collapse = jepa_training_step(batch, enc, tgt, pred, lr, tau)
        history.append({
        "loss": float(loss),
        "collapse": float(collapse)
    })

    return enc, tgt, pred, history