"""Shared helpers for the figure modules."""
import numpy as np
from numpy import ndarray
from torch import Tensor

from data_env import pendulum_system


def wrap_angle(theta: ndarray) -> ndarray:
    """Wrap angle to [-pi, pi]."""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def noiseless_reference(traj: Tensor, t_grid: ndarray) -> ndarray:
    """Noise-free reference trajectory re-integrated from the observed IC.

    The dataset carries additive sampling noise, so the raw 'True' curve looks
    jagged; this re-integrates the exact pendulum from the same IC so plotted
    truth is smooth.
    """
    from scipy.integrate import solve_ivp

    x0 = traj[0].detach().cpu().numpy().astype(float)
    sol = solve_ivp(lambda t, s: pendulum_system(t, s), (float(t_grid[0]), float(t_grid[-1])), x0,
                    t_eval=t_grid, method='RK45', rtol=1e-10, atol=1e-12)
    return sol.y.T
