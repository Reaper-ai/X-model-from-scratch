"""
Action-conditioned predictor as nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """Predicts next latent embedding given current embedding and action."""

    def __init__(self, latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.action_embedding = nn.Embedding(action_dim, latent_dim)
        self.fc1 = nn.Linear(2 * latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, latent_dim) or (latent_dim,)
            actions: (B,) or scalar
        Returns:
            predicted next embedding: (B, latent_dim) or (latent_dim,)
        """
        # Handle single sample
        squeeze = embeddings.dim() == 1
        if squeeze:
            embeddings = embeddings.unsqueeze(0)
            actions = torch.as_tensor(actions).unsqueeze(0) if isinstance(actions, int) else actions.unsqueeze(0)

        action_embeds = self.action_embedding(actions)
        x = torch.cat([embeddings, action_embeds], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if squeeze:
            x = x.squeeze(0)
        return x


def rollout_latent_dynamics(predictor: Predictor, initial_embedding: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Roll out latent dynamics for a sequence of actions.

    Args:
        predictor: Predictor module
        initial_embedding: (latent_dim,) or (B, latent_dim)
        actions: (horizon,) or (B, horizon)
    Returns:
        rollout: (horizon+1, B, latent_dim) including initial state
    """
    squeeze = initial_embedding.dim() == 1
    if squeeze:
        initial_embedding = initial_embedding.unsqueeze(0)

    if actions.dim() == 1:
        actions = actions.unsqueeze(0).expand(initial_embedding.shape[0], -1)

    current = initial_embedding
    rollout = [current]

    for t in range(actions.shape[1]):
        current = predictor(current, actions[:, t])
        rollout.append(current)

    rollout = torch.stack(rollout, dim=0)
    if squeeze:
        rollout = rollout.squeeze(1)
    return rollout