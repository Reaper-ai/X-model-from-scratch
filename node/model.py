"""
Model architectures: Classical integrators, ResNet MLP, Neural ODE.
"""
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


# STEP 1: Residual Block (shared by ResNet and Neural ODE)
class ResidualBlock(nn.Module):
    """Residual block: x + Linear(ReLU(Linear(x)))."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# STEP 2: ResNet MLP (Baseline - Derivative Matching)
class ResNetMLP(nn.Module):
    """
    Time-invariant MLP with residual blocks for derivative prediction.
    Learns f_θ(x) → dx/dt. Used as baseline: predicts vector field at each point.
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, n_layers: int = 4, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.net = nn.Sequential(self.layer, self.res_blocks, self.output_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# STEP 3: Neural ODE (Time-Conditioned Vector Field)
class NeuralODE(nn.Module):
    """
    Neural ODE: time-conditioned vector field f_θ(t, x) → dx/dt.
    Uses torchdiffeq for integration with adjoint method for memory-efficient gradients.
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, n_layers: int = 4, output_dim: int = 3, time_embed_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.time_embed_dim = time_embed_dim

        self.time_embed = nn.Sequential(nn.Linear(1, time_embed_dim), nn.ReLU(), nn.Linear(time_embed_dim, time_embed_dim))
        self.input_layer = nn.Sequential(nn.Linear(input_dim + time_embed_dim, hidden_dim), nn.ReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.net = nn.Sequential(self.input_layer, self.res_blocks, self.final_layer)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Handle both batched (from training) and non-batched (from torchdiffeq) inputs
        # torchdiffeq calls with t as scalar (0-d) and x as (dim,) or (batch, dim)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (dim,) -> (1, dim)
        
        batch_size = x.shape[0]
        t_flat = t.reshape(-1, 1).expand(batch_size, 1)
        t_emb = self.time_embed(t_flat)
        tx = torch.cat([x, t_emb], dim=-1)
        out = self.net(tx)
        
        # Return same shape as input x (without batch dim if input was 1D)
        return out.squeeze(0) if out.shape[0] == 1 and x.shape[0] == 1 else out


# STEP 4: Classical Integrator Wrapper (Not Learned)
class ClassicalIntegrator:
    """
    Wrapper for classical numerical integrators (not learned).
    """
    def __init__(self, method: str = 'RK45', rtol: float = 1e-7, atol: float = 1e-10):
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def integrate(self, func: callable, x0: torch.Tensor, t_eval: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Integrate ODE from initial condition.

        Args:
            func: Callable f(t, x) → dx/dt.
            x0: Initial state of shape (3,) or (batch, 3).
            t_eval: Time points of shape (T,).

        Returns:
            Trajectory of shape (T, 3) or (T, batch, 3).
        """
        if self.method in ('Euler', 'RK4'):
            return self._fixed_step_integrate(func, x0, t_eval)
        else:
            return self._adaptive_integrate(func, x0, t_eval, **kwargs)

    def _fixed_step_integrate(
        self,
        func: callable,
        x0: torch.Tensor,
        t_eval: torch.Tensor
    ) -> torch.Tensor:
        """Fixed-step Euler or RK4 integration."""
        device = x0.device
        traj = [x0]
        x = x0

        for i in range(len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            t = t_eval[i]

            if self.method == 'Euler':
                x = x + func(t, x) * dt
            elif self.method == 'RK4':
                k1 = func(t, x)
                k2 = func(t + dt / 2, x + dt * k1 / 2)
                k3 = func(t + dt / 2, x + dt * k2 / 2)
                k4 = func(t + dt, x + dt * k3)
                x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            else:
                raise ValueError(f"Unsupported method: {self.method}")

            traj.append(x)

        return torch.stack(traj)

    def _adaptive_integrate(
        self,
        func: callable,
        x0: torch.Tensor,
        t_eval: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Adaptive-step integration via scipy (requires CPU)."""
        from scipy.integrate import solve_ivp
        import numpy as np

        def scipy_func(t, y):
            t_t = torch.tensor(t, dtype=x0.dtype, device=x0.device)
            y_t = torch.tensor(y, dtype=x0.dtype, device=x0.device)
            with torch.no_grad():
                dydt = func(t_t, y_t)
            return dydt.cpu().numpy()

        x0_np = x0.detach().cpu().numpy()
        t_eval_np = t_eval.detach().cpu().numpy()

        if x0_np.ndim == 1:
            sol = solve_ivp(scipy_func, (t_eval_np[0], t_eval_np[-1]), x0_np,
                          t_eval=t_eval_np, method=self.method, rtol=self.rtol, atol=self.atol)
            return torch.from_numpy(sol.y.T).to(x0.device).to(x0.dtype)
        else:
            trajs = []
            for i in range(x0_np.shape[0]):
                sol = solve_ivp(scipy_func, (t_eval_np[0], t_eval_np[-1]), x0_np[i],
                              t_eval=t_eval_np, method=self.method, rtol=self.rtol, atol=self.atol)
                trajs.append(sol.y.T)
            # trajs: list of (T, 3) -> stack to (batch, T, 3) -> transpose to (T, batch, 3)
            trajs_np = np.stack(trajs)  # (batch, T, 3)
            trajs_np = np.transpose(trajs_np, (1, 0, 2))  # (T, batch, 3)
            return torch.from_numpy(trajs_np).to(x0.device).to(x0.dtype)


# STEP 5: Loss Functions
def vector_field_loss(pred_deriv: torch.Tensor, true_deriv: torch.Tensor) -> torch.Tensor:
    """MSE loss on predicted vs true derivatives."""
    return torch.mean((pred_deriv - true_deriv) ** 2)


def trajectory_loss(pred_traj: torch.Tensor, true_traj: torch.Tensor) -> torch.Tensor:
    """MSE loss on predicted vs true trajectories."""
    return torch.mean((pred_traj - true_traj) ** 2)


# STEP 6: Utilities
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def integrate_neural_ode(
    model: NeuralODE,
    x0: torch.Tensor,
    t_eval: torch.Tensor,
    method: str = 'dopri5',
    use_adjoint: bool = True,
    **kwargs
) -> torch.Tensor:
    """Integrate Neural ODE using torchdiffeq."""
    if use_adjoint:
        return odeint_adjoint(model, x0, t_eval, method=method, **kwargs)
    else:
        return odeint(model, x0, t_eval, method=method, **kwargs)


def integrate_resnet(
    model: ResNetMLP,
    x0: torch.Tensor,
    t_eval: torch.Tensor,
    method: str = 'RK4'
) -> torch.Tensor:
    """Integrate ResNet MLP using fixed-step integrator."""
    integrator = ClassicalIntegrator(method=method)
    return integrator.integrate(lambda t, x: model(x), x0, t_eval)