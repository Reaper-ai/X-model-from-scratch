"""
Action-Conditioned Predictor.
Steps 13-16 from the JEPA implementation.
"""

import torch
import torch.nn.functional as F


def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    t = lambda x: (torch.randn(x, requires_grad=True) * 0.02).detach().requires_grad_(True)

    predictor_params = {}
    predictor_params['action_embed_w'] = t([action_dim, latent_dim])
    predictor_params['fc1_w'] = t([hidden_dim, 2 * latent_dim])
    predictor_params['fc2_w'] = t([latent_dim, hidden_dim])
    predictor_params['fc1_b'] = torch.zeros(hidden_dim, requires_grad=True)
    predictor_params['fc2_b'] = torch.zeros(latent_dim, requires_grad=True)

    return predictor_params


def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_embed_w = predictor_params['action_embed_w']
    action_vec = action_embed_w[actions]
    return action_vec


def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_embeds = embed_action(actions, predictor_params)
    pred_vec = torch.concat([embeddings, action_embeds], dim=-1)
    fc1 = F.linear(input=pred_vec, weight=predictor_params['fc1_w'], bias=predictor_params['fc1_b'])
    r1 = F.relu(fc1)
    fc2 = F.linear(input=r1, weight=predictor_params['fc2_w'], bias=predictor_params['fc2_b'])

    return fc2


def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    next_z = predictor_forward(embeddings, actions, predictor_params)
    return next_z