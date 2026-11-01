"""
Loss Functions: Prediction, Variance, Covariance, VICReg, JEPA, Collapse Metric.
Steps 17-22 from the JEPA implementation.
"""

import torch
import torch.nn.functional as F


def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = predicted - target
    error2 = torch.square(error)
    return torch.mean(error2)


def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    var = torch.var(embeddings, dim=0)
    std = torch.sqrt(var + eps)
    res = F.relu(gamma - std)
    loss = torch.mean(res)
    return loss


def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    mean = embeddings.mean(dim=0)
    z = embeddings - mean
    B = embeddings.shape[0]
    cov = torch.mm(z.t(), z) / (B - 1)
    D = embeddings.shape[1]
    loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    return loss


def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    var = variance_loss(embeddings, gamma)
    cov = covariance_loss(embeddings)
    vic = var_weight * var + cov_weight * cov

    return vic


def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    pred_loss = prediction_loss(predicted, target)
    reg = vicreg_regularizer(online_embeddings, var_weight, cov_weight)
    return pred_weight * pred_loss + reg


def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    stds = torch.std(embeddings, dim=0)
    return torch.mean(stds)