"""
Full JEPA World Model as nn.Module.
Combines encoder, EMA target encoder, and predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, EMAEncoder
from .predictor import Predictor, rollout_latent_dynamics


class JEPAModel(nn.Module):
    """
    Complete JEPA World Model.

    Components:
    - Online encoder: maps observations to latents
    - EMA target encoder: slow-moving target for stable predictions
    - Predictor: action-conditioned latent dynamics
    """

    def __init__(
        self,
        obs_channels: int = 1,
        room_size: int = 8,
        latent_dim: int = 32,
        action_dim: int = 4,
        hidden_dim: int = 64,
        tau: float = 0.99
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.room_size = room_size
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.tau = tau

        # Online encoder
        self.encoder = Encoder(obs_channels, room_size, latent_dim)

        # EMA target encoder
        self.target_encoder = EMAEncoder(self.encoder, tau)

        # Predictor
        self.predictor = Predictor(latent_dim, action_dim, hidden_dim)

    def encode(self, obs: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Encode observations to latents."""
        if use_target:
            return self.target_encoder(obs)
        return self.encoder(obs)

    def predict(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next latent given current latent and action."""
        return self.predictor(embeddings, actions)

    def rollout(self, initial_embedding: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out latent dynamics for a sequence of actions."""
        return rollout_latent_dynamics(self.predictor, initial_embedding, actions)

    @torch.no_grad()
    def update_target(self):
        """Update EMA target encoder."""
        self.target_encoder.update(self.encoder)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor) -> dict:
        """
        Full forward pass for training.

        Returns dict with:
        - embeddings: online encoder output
        - target: target encoder output for next_obs
        - predicted: predictor output
        """
        embeddings = self.encode(obs, use_target=False)
        with torch.no_grad():
            target = self.encode(next_obs, use_target=True)
        predicted = self.predict(embeddings, actions)
        return {
            'embeddings': embeddings,
            'target': target,
            'predicted': predicted
        }

    def jepa_loss(self, predicted: torch.Tensor, target: torch.Tensor, embeddings: torch.Tensor,
                  pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04,
                  gamma: float = 1.0) -> torch.Tensor:
        """Compute JEPA loss with VICReg regularization."""
        # Prediction loss
        pred_loss = F.mse_loss(predicted, target)

        # Variance loss
        var = torch.var(embeddings, dim=0)
        std = torch.sqrt(var + 1e-4)
        var_loss = F.relu(gamma - std).mean()

        # Covariance loss
        mean = embeddings.mean(dim=0)
        z = embeddings - mean
        B = embeddings.shape[0]
        cov = torch.mm(z.t(), z) / (B - 1)
        D = embeddings.shape[1]
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D

        total_loss = pred_weight * pred_loss + var_weight * var_loss + cov_weight * cov_loss
        return total_loss

    def collapse_metric(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Mean std across latent dimensions."""
        return embeddings.std(dim=0).mean()


class LinearProbe(nn.Module):
    """Linear probe to decode state from latents."""

    def __init__(self, latent_dim: int = 32, state_dim: int = 2):
        super().__init__()
        self.probe = nn.Linear(latent_dim, state_dim, bias=True)
        nn.init.normal_(self.probe.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.probe.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.probe(embeddings)