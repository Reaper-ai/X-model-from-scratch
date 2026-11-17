"""
Model architectures: Classical integrators, Discrete MLP, Neural ODE.
Tensor-first design. Numpy only at scipy boundary.

The two learned models are deliberately *different hypothesis classes*:

  DiscreteMLP : x_{k+1} = MLP(x_k)   -- a one-step flow map at a fixed dt.
                 No notion of continuous time; can only be stepped at its dt.
  NeuralODE   : dx/dt = MLP(x)        -- a continuous-time vector field.
                 Can be integrated at arbitrary query times and arbitrary dt.

Sharing the exact same MLP stack between the two keeps the parameter count
and capacity identical, so any difference in accuracy comes from the
time parameterization and not from capacity.
"""
from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

def _mlp_stack(input_dim: int, hidden_dim: int, n_layers: int, output_dim: int) -> nn.Sequential:
    """Shared MLP trunk: Linear + SiLU stack, identical for both models."""
    layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

# STEP 1: Discrete MLP (one-step flow map — the honest discrete baseline)
class DiscreteMLP(nn.Module):
    """
    Discrete one-step flow map: x_{k+1} = MLP(x_k), with a fixed step dt.

    This is the real "no continuous-time structure" baseline. It never sees a
    timestamp, so it cannot be queried at arbitrary times and it is tied to
    the dt it was trained on.

    Two parameterizations:
      residual=False (default): x_{k+1} = MLP(x_k)            — direct next state.
      residual=True : x_{k+1} = x_k + MLP(x_k)                — delta form, with the
      last layer zero-initialised so the map starts exactly at the identity.
      The delta form is the standard discrete flow map (a.k.a. "ResNet flow");
      it is still a purely discrete map with no time variable.

    `dt` is stored because the Euler-equivalent continuous field implied by
    the map is  f(x) = (MLP(x) - x) / dt  (used for the vector-field panel).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 4,
        output_dim: int = 2,
        dt: float = 0.05,
        residual: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dt = dt
        self.residual = residual

        self.net = _mlp_stack(input_dim, hidden_dim, n_layers, output_dim)
        _init_weights(self)

        if residual:
            # Zero the output layer so the map starts as the identity map:
            # a randomly initialised one-step map is unstable to roll out.
            last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One-step map: x_k -> x_{k+1}. Shape (..., 2) -> (..., 2)."""
        out = self.net(x)
        return x + out if self.residual else out

    def rollout(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Iterate the map n_steps times.

        x0: (2,) or (B, 2)
        returns: (n_steps + 1, 2) or (B, n_steps + 1, 2)
        """
        x = x0
        states = [x]
        for _ in range(n_steps):
            x = self.forward(x)
            states.append(x)
        if x0.dim() == 1:
            return torch.stack(states, dim=0)
        return torch.stack(states, dim=1)

    def implied_field(self, x: torch.Tensor) -> torch.Tensor:
        """Euler-equivalent continuous field implied by the one-step map."""
        return (self.forward(x) - x) / self.dt

# STEP 2: Neural ODE (Autonomous Vector Field)
class NeuralODE(nn.Module):
    """
    Neural ODE: autonomous vector field f_θ(x) → dx/dt.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, n_layers: int = 4, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.net = _mlp_stack(input_dim, hidden_dim, n_layers, output_dim)
        _init_weights(self)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# STEP 3: Classical Integrator Wrapper (Not Learned)
class ClassicalIntegrator:
    """Wrapper for classical numerical integrators (not learned)."""
    def __init__(self, method: str = 'RK45', rtol: float = 1e-7, atol: float = 1e-10):
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def integrate(
        self,
        func: callable,
        x0: torch.Tensor,
        t_eval: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
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
        device = x0.device
        dtype = x0.dtype
        traj = [x0]
        x = x0

        def to_tensor(fx):
            if isinstance(fx, np.ndarray):
                return torch.from_numpy(fx).to(device).to(dtype)
            return fx

        for i in range(len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            t = t_eval[i]

            if self.method == 'Euler':
                x = x + to_tensor(func(t, x)) * dt
            elif self.method == 'RK4':
                k1 = to_tensor(func(t, x))
                k2 = to_tensor(func(t + dt / 2, x + dt * k1 / 2))
                k3 = to_tensor(func(t + dt / 2, x + dt * k2 / 2))
                k4 = to_tensor(func(t + dt, x + dt * k3))
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
        from scipy.integrate import solve_ivp
        import numpy as np

        def scipy_func(t, y):
            t_t = torch.tensor(t, dtype=x0.dtype, device=x0.device)
            y_t = torch.tensor(y, dtype=x0.dtype, device=x0.device)
            with torch.no_grad():
                dydt = func(t_t, y_t)
            if isinstance(dydt, torch.Tensor):
                return dydt.detach().cpu().numpy()
            else:
                return np.array(dydt, dtype=np.float64)

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
            trajs_np = np.stack(trajs)
            trajs_np = np.transpose(trajs_np, (1, 0, 2))
            return torch.from_numpy(trajs_np).to(x0.device).to(x0.dtype)

# STEP 4: Loss Functions
def vector_field_loss(pred_deriv: torch.Tensor, true_deriv: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_deriv - true_deriv) ** 2)

def trajectory_loss(pred_traj: torch.Tensor, true_traj: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_traj - true_traj) ** 2)

# STEP 5: Utilities
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def integrate_neural_ode(
    model: NeuralODE,
    x0: torch.Tensor,
    t_eval: torch.Tensor,
    method: str = 'dopri5',
    use_adjoint: bool = False,
    **kwargs
) -> torch.Tensor:
    if use_adjoint:
        return odeint_adjoint(model, x0, t_eval, method=method, **kwargs)
    else:
        return odeint(model, x0, t_eval, method=method, **kwargs)