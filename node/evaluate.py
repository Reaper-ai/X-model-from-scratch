"""
Evaluation utilities: rollout metrics, test-set loops, arbitrary-time queries.

Protocol note: a model trained on irregular timestamps must be evaluated either
on the regular grid (regular loaders) or at the loaders' own timestamps
(`evaluate_queries_on_loader`). Integrating over a regular grid while comparing
against irregularly-timed states produces wildly wrong numbers.
"""
from collections.abc import Callable
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader

from model import integrate_neural_ode


def compute_lyapunov_exponent(pred_traj: Tensor, true_traj: Tensor, t_eval: Tensor) -> float:
    """Slope of log trajectory distance vs time (early phase for damped systems)."""
    dist = torch.norm(pred_traj - true_traj, dim=-1).cpu().numpy()
    t = t_eval.cpu().numpy()

    t_cutoff = t[int(len(t) * 0.3)]
    valid = (dist > 1e-12) & (dist < 10) & (t <= t_cutoff)
    if np.sum(valid) < 5:
        valid = (dist > 1e-12) & (dist < 10)
        if np.sum(valid) < 5:
            return 0.0

    coeffs = np.polyfit(t[valid], np.log(dist[valid]), 1)
    return float(coeffs[0])


@torch.no_grad()
def evaluate_trajectory(model: nn.Module, x0: Tensor, t_eval: Tensor, true_traj: Tensor, is_neural_ode: bool = True,
                        is_discrete: bool = False, integration_method: str = 'dopri5',
                        rtol: float = 1e-4, atol: float = 1e-6) -> dict[str, float]:
    """Evaluate one rollout from x0 (discrete: iterate the map; ODE: integrate)."""
    model.eval()
    if is_discrete:
        pred_traj = model.rollout(x0, max(int(t_eval.numel()) - 1, 0))
    elif is_neural_ode:
        pred_traj = integrate_neural_ode(model, x0, t_eval, method=integration_method,
                                         use_adjoint=False, rtol=rtol, atol=atol)
    else:
        raise ValueError("Either is_discrete or is_neural_ode must be True.")

    true_traj = true_traj.to(x0.device)
    return {
        'mse': torch.mean((pred_traj - true_traj) ** 2).item(),
        'max_error': torch.max(torch.norm(pred_traj - true_traj, dim=-1)).item(),
        'lyapunov_exp': compute_lyapunov_exponent(pred_traj, true_traj, t_eval),
    }


def evaluate_on_test_set(model: nn.Module, test_trajectories: Tensor, t_eval: Tensor, is_neural_ode: bool = True,
                         is_discrete: bool = False, integration_method: str = 'dopri5',
                         rtol: float = 1e-4, atol: float = 1e-6,
                         device: torch.device = torch.device('cpu')) -> dict[str, ndarray]:
    """Run `evaluate_trajectory` over a stacked (N, T, 2) test tensor."""
    all_metrics = [
        evaluate_trajectory(model, traj[0].to(device), t_eval.to(device), traj.to(device), is_neural_ode,
                            is_discrete, integration_method, rtol, atol)
        for traj in test_trajectories
    ]
    return {key: np.array([m[key] for m in all_metrics]) for key in all_metrics[0]}


def evaluate_all_tests(model: nn.Module, test_A_loader: DataLoader, test_B_loader: DataLoader,
                       test_C_loader: DataLoader, t_eval: ndarray, is_neural_ode: bool = True,
                       is_discrete: bool = False, integration_method: str = 'dopri5',
                       rtol: float = 1e-4, atol: float = 1e-6, device: torch.device = torch.device('cpu'),
                       use_loader_times: bool = False) -> dict[str, dict[str, ndarray]]:
    """Evaluate on all 3 test splits; prints a one-line summary per split.

    Forward-only inference (no adjoint). With `use_loader_times=True` the
    timestamps stored in each loader are used (irregular for the ODE view);
    otherwise a regular grid is reconstructed from `t_eval`.
    """
    loaders = {'test_A': test_A_loader, 'test_B': test_B_loader, 'test_C': test_C_loader}
    results = {}

    for name, loader in loaders.items():
        all_metrics = []
        for batch in loader:
            x = batch['x'].to(device)
            lengths = batch['lengths']
            t_batch = batch['t'].to(device) if 't' in batch else None

            for i in range(x.shape[0]):
                L_i = lengths[i].item()
                true_traj = x[i, :L_i]
                if use_loader_times and t_batch is not None:
                    t_i = t_batch[i, :L_i]
                else:
                    t_i = torch.arange(L_i, device=device) * (t_eval[1] - t_eval[0])

                all_metrics.append(evaluate_trajectory(
                    model, true_traj[0], t_i, true_traj, is_neural_ode, is_discrete,
                    integration_method, rtol, atol))

        results[name] = {key: np.array([m[key] for m in all_metrics]) for key in all_metrics[0]}
        print(f"  {name}: MSE={results[name]['mse'].mean():.6e}, "
              f"max_err={results[name]['max_error'].mean():.6e}, "
              f"lyap={results[name]['lyapunov_exp'].mean():.6e}")

    return results


@torch.no_grad()
def evaluate_on_queries(model: nn.Module, x0: Tensor, query_t: Tensor, true_traj: Tensor, is_discrete: bool = False,
                        dt: float = 0.05, integration_method: str = 'dopri5',
                        rtol: float = 1e-4, atol: float = 1e-6,
                        device: torch.device = torch.device('cpu')) -> dict[str, float]:
    """Evaluate at arbitrary query times.

    The ODE integrates directly to the query times. The discrete map can only
    be stepped at its own dt, so it rolls out on its native grid and is
    linearly interpolated to the query times (its documented handicap).
    """
    model.eval()
    x0 = x0.to(device)

    if is_discrete:
        q = query_t.detach().cpu().numpy().astype(float)
        n_steps = int(np.ceil(q[-1] / dt))
        grid_t = np.arange(n_steps + 1, dtype=float) * dt
        pred_grid = model.rollout(x0, n_steps).detach().cpu().numpy()
        if pred_grid.ndim == 1:
            pred_grid = pred_grid[None, :]
        pred = np.stack([np.interp(q, grid_t, pred_grid[:, c]) for c in range(pred_grid.shape[1])], axis=1)
        pred_traj = torch.from_numpy(pred).to(device).float()
    else:
        pred_traj = integrate_neural_ode(model, x0, query_t.to(device), method=integration_method,
                                         use_adjoint=False, rtol=rtol, atol=atol)

    true_traj = true_traj.to(device)
    return {
        'mse': torch.mean((pred_traj - true_traj) ** 2).item(),
        'max_error': torch.max(torch.norm(pred_traj - true_traj, dim=-1)).item(),
    }


def evaluate_queries_on_loader(model: nn.Module, loader: DataLoader, is_discrete: bool = False, dt: float = 0.05,
                               integration_method: str = 'dopri5', rtol: float = 1e-4, atol: float = 1e-6,
                               device: torch.device = torch.device('cpu')) -> dict[str, ndarray]:
    """Run `evaluate_on_queries` over every trajectory in a loader."""
    all_metrics = []
    for batch in loader:
        x = batch['x'].to(device)
        t = batch['t'].to(device)
        lengths = batch['lengths']
        for i in range(x.shape[0]):
            L_i = lengths[i].item()
            all_metrics.append(evaluate_on_queries(model, x[i, 0], t[i, :L_i], x[i, :L_i],
                                                   is_discrete, dt, integration_method, rtol, atol, device))

    if not all_metrics:
        return {'mse': np.array([]), 'max_error': np.array([])}

    results = {key: np.array([m[key] for m in all_metrics]) for key in all_metrics[0]}
    print(f"  [queries] MSE={results['mse'].mean():.6e}, max_err={results['max_error'].mean():.6e}")
    return results


class ClassicalEvaluator:
    """Evaluator for classical integrators (scipy RK45, etc.)."""

    def __init__(self, integrator: Any, device: torch.device):
        self.integrator, self.device = integrator, device

    @torch.no_grad()
    def evaluate(self, true_trajectories: Tensor, t_eval: Tensor, true_func: Callable) -> dict[str, ndarray]:
        """Integrate each trajectory's IC and report MSE / max error / Lyapunov slope."""

        def wrapped_func(t, x):
            t_np = t.cpu().numpy() if isinstance(t, Tensor) else t
            x_np = x.cpu().numpy() if isinstance(x, Tensor) else x
            return torch.from_numpy(true_func(t_np, x_np)).to(x.device).to(x.dtype)

        mse, max_error, lyapunov_exp = [], [], []
        for i in range(true_trajectories.shape[0]):
            true_traj = true_trajectories[i].to(self.device)
            pred_traj = self.integrator.integrate(wrapped_func, true_traj[0], t_eval.to(self.device))
            mse.append(torch.mean((pred_traj - true_traj) ** 2).item())
            max_error.append(torch.max(torch.norm(pred_traj - true_traj, dim=-1)).item())
            lyapunov_exp.append(compute_lyapunov_exponent(pred_traj, true_traj, t_eval.to(self.device)))

        return {'mse': np.array(mse), 'max_error': np.array(max_error), 'lyapunov_exp': np.array(lyapunov_exp)}
