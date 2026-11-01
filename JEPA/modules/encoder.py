"""
Encoder as nn.Module with EMA target support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional encoder mapping observations to latent embeddings."""

    def __init__(self, obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32):
        super().__init__()
        self.obs_channels = obs_channels
        self.room_size = room_size
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(obs_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        conv2_out_size = room_size // 2
        flattened_size = 32 * conv2_out_size * conv2_out_size

        self.fc = nn.Linear(flattened_size, latent_dim)

        # Initialize with same scheme as functional version
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.fc(x)
        return x


class EMAEncoder(nn.Module):
    """EMA target encoder that tracks an online encoder."""

    def __init__(self, online_encoder: Encoder, tau: float = 0.99):
        super().__init__()
        self.tau = tau
        # Create target encoder with same architecture
        self.target = Encoder(
            obs_channels=online_encoder.obs_channels,
            room_size=online_encoder.room_size,
            latent_dim=online_encoder.latent_dim
        )
        # Initialize target as copy of online
        self.target.load_state_dict(online_encoder.state_dict())
        # Freeze target parameters
        for param in self.target.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: Encoder):
        """EMA update: target = tau * target + (1 - tau) * online"""
        for target_param, online_param in zip(self.target.parameters(), online_encoder.parameters()):
            target_param.data.mul_(self.tau).add_(online_param.data, alpha=1 - self.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x)