"""
Encoder and EMA Target Network.
Steps 8-12 from the JEPA implementation.
"""

import torch
import torch.nn.functional as F


def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    params = {}

    params['conv1_w'] = torch.randn(16, obs_channels, 3, 3) * 0.1
    params['conv1_b'] = torch.zeros(16)

    params['conv2_w'] = torch.randn(32, 16, 3, 3) * 0.1
    params['conv2_b'] = torch.zeros(32)

    conv2_out_size = room_size // 2
    flattened_size = 32 * conv2_out_size * conv2_out_size

    params['fc_w'] = torch.randn(latent_dim, flattened_size) * 0.1
    params['fc_b'] = torch.zeros(latent_dim)

    for k in params:
        params[k].requires_grad = True

    return params


def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    c1 = F.conv2d(input=obs, weight=encoder_params['conv1_w'], stride=(1, 1), padding=(1, 1), bias=encoder_params['conv1_b'])
    f1 = F.relu(c1)

    c2 = F.conv2d(input=f1, weight=encoder_params['conv2_w'], stride=(2, 2), padding=(1, 1), bias=encoder_params['conv2_b'])
    f2 = F.relu(c2)
    f2 = f2.flatten(1)

    fc = F.linear(input=f2, weight=encoder_params['fc_w'], bias=encoder_params['fc_b'])

    return fc


def init_target_encoder(encoder_params: dict) -> dict:
    target = {}
    for key in encoder_params:
        value = encoder_params[key]
        target[key] = value.detach().clone()
        target[key].requires_grad = False
    return target


def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    t = {}
    for key in encoder_params:
        t[key] = tau * target_params[key]
        t[key] += (1 - tau) * encoder_params[key]
    return t


def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    return encoder_forward(obs, encoder_params)