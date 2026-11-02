"""
Training Step and Full Training Loop.
Steps 23-24 from the JEPA implementation.
Enhanced with curriculum learning, validation tracking, and early stopping.
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
    loss.backward()

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


@torch.no_grad()
def evaluate_val_loss(
    val_dataset: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    batch_size: int = 32
) -> float:
    """Evaluate validation loss on a subset of val data."""
    N = val_dataset["observations"].shape[0]
    if N == 0:
        return float('inf')
    
    # Use a fixed subset for consistent evaluation
    eval_size = min(batch_size * 4, N)
    idx = torch.randperm(N)[:eval_size]
    
    batch = {
        "observations": val_dataset["observations"][idx],
        "actions": val_dataset["actions"][idx],
        "next_observations": val_dataset["next_observations"][idx]
    }
    
    embeddings = encode_batch(batch["observations"], encoder_params)
    target = encode_batch(batch["next_observations"], target_params)
    predicted = predict_next_embedding(embeddings, batch["actions"], predictor_params)
    
    loss = jepa_loss(predicted, target, embeddings)
    return loss.item()


def train_jepa(
    dataset: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    num_steps: int = 2000,
    batch_size: int = 32,
    lr: float = 0.1,
    tau: float = 0.99,
    seed: int = 0,
    curriculum: bool = True,
    max_horizon: int = 20,
    steps_per_epoch: int = 12,
    horizon_every_epochs: int = 30,
    val_dataset: dict = None,
    val_every: int = 50,
    patience: int | None = None
) -> tuple[dict, dict, dict, list]:
    """
    Train JEPA with curriculum learning, validation tracking, and early stopping.
    
    Args:
        dataset: Training dataset (on device)
        encoder_params: Online encoder parameters
        target_params: Target encoder parameters
        predictor_params: Predictor parameters
        num_steps: Total training steps
        batch_size: Batch size
        lr: Learning rate
        tau: EMA decay
        seed: Random seed
        curriculum: If True, use progressive horizon curriculum
        max_horizon: Maximum horizon for curriculum (target)
        steps_per_epoch: Steps in one epoch (for horizon schedule)
        horizon_every_epochs: Advance horizon every N epochs
        val_dataset: Validation dataset (on device) for early stopping
        val_every: Evaluate val loss every N steps
        patience: Stop if no val improvement for this many training steps (None = no early stop)
    
    Returns:
        Updated encoder, target, predictor, and history with metrics
    """
    torch.manual_seed(seed)
    N = dataset["observations"].shape[0]

    enc = encoder_params
    tgt = target_params
    pred = predictor_params
    history = []
    
    # Curriculum: advance horizon every `horizon_every_epochs` epochs
    if curriculum:
        # 4 phases starting at 5: 5 -> 10 -> 15 -> 20 (no h=1)
        phases = [5, 10, 15, 20]
        phase_len = max(1, horizon_every_epochs * steps_per_epoch)
        horizon_schedule = []
        for phase in phases:
            horizon_schedule.extend([phase] * phase_len)
        # Pad or trim to num_steps
        if len(horizon_schedule) < num_steps:
            horizon_schedule.extend([phases[-1]] * (num_steps - len(horizon_schedule)))
        horizon_schedule = horizon_schedule[:num_steps]
        print(f"Horizon curriculum: phases {phases}, every {horizon_every_epochs} epochs "
              f"({phase_len} steps); shifts at steps "
              f"{[phase_len * k for k in range(1, len(phases))]}")
    else:
        horizon_schedule = [max_horizon] * num_steps

    # Early stopping state (patience measured in raw training steps)
    best_val_loss = float('inf')
    last_improve_step = 0
    early_stopped = False

    for i in range(num_steps):
        idx = torch.randint(0, N, (batch_size,))
        batch = {
            "observations": dataset["observations"][idx],
            "actions": dataset["actions"][idx],
            "next_observations": dataset["next_observations"][idx]
        }
        
        # Use curriculum horizon for this step (currently unused in step, but tracked)
        current_horizon = horizon_schedule[i]
        
        enc, tgt, pred, loss, collapse = jepa_training_step(batch, enc, tgt, pred, lr, tau)
        
        entry = {
            "step": i,
            "train_loss": float(loss),
            "collapse": float(collapse),
            "horizon": current_horizon
        }
        
        # Validation evaluation
        if val_dataset is not None and i % val_every == 0 and i > 0:
            val_loss = evaluate_val_loss(val_dataset, enc, tgt, pred, batch_size)
            entry["val_loss"] = val_loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                last_improve_step = i
            
            if patience is not None and i - last_improve_step >= patience:
                early_stopped = True
                entry["early_stopped"] = True
                print(f"Early stopping at step {i} (no val improvement for {patience} steps; "
                      f"best {best_val_loss:.4f} at step {last_improve_step})")
        
        history.append(entry)
        
        if early_stopped:
            break

    return enc, tgt, pred, history