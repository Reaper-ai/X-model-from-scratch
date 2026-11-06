"""
Lorenz system data generation, irregular sampling, and Dataset classes.
STUBS ONLY — implement each function step by step.
"""
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp


# =============================================================================
# STEP 1: Ground Truth ODE
# =============================================================================

def lorenz_system(t: float, state: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    """
    Lorenz system ODE: dx/dt = f(x).
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


# STEP 2: Generate Trajectories (Ground Truth)
def generate_trajectories(
    n_trajectories: int = 100,
    t_span: Tuple[float, float] = (0.0, 25.0),
    dt: float = 0.01,
    noise_std: float = 0.0,
    seed: int = 42,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lorenz trajectories using high-order integration (RK45).
    """
    rng = np.random.default_rng(seed=seed)
    t_eval = np.arange(t_span[0], t_span[1], dt)
    trajectories = []

    for _ in range(n_trajectories):
        x0 = rng.uniform(-15, 15, size=3)
        sol = solve_ivp(
            fun=lambda t, y: lorenz_system(t, y, sigma, rho, beta),
            t_span=t_span,
            y0=x0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-12
        )
        traj = sol.y.T  # Shape: (T, 3)
        if noise_std > 0:
            traj += rng.normal(0, noise_std, traj.shape)
        trajectories.append(traj)

    return t_eval, np.stack(trajectories)  # Shape: (n_trajectories, T, 3)


# STEP 3: Irregular Time Sampling
def sample_irregular_timepoints(t_eval: np.ndarray, n_samples: int, strategy: str = 'random', seed: int = 42) -> np.ndarray:
    """
    Sample irregular time indices from a regular time grid.
    """
    rng = np.random.default_rng(seed=seed)
    T = len(t_eval)
    n_samples = min(n_samples, T)

    if strategy == 'random':
        indices = rng.choice(T, size=n_samples, replace=False)
        indices.sort()
        return indices
    elif strategy == 'poisson':
        indices = [0]
        while len(indices) < n_samples and indices[-1] < T - 1:
            gap = rng.exponential(scale=T / n_samples)
            next_idx = min(indices[-1] + int(gap) + 1, T - 1)
            if next_idx > indices[-1]:
                indices.append(next_idx)
        return np.array(indices[:n_samples])
    elif strategy == 'log':
        log_space = np.logspace(0, np.log10(T - 1), num=n_samples)
        indices = np.unique(np.floor(log_space).astype(int))
        return np.sort(indices)[:n_samples]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# STEP 4: PyTorch Dataset
class LorenzDataset(Dataset):
    """
    PyTorch Dataset for Lorenz trajectories.
    """
    def __init__(
        self,
        trajectories: np.ndarray,
        t_eval: np.ndarray,
        indices: Optional[np.ndarray] = None,
        compute_deriv: bool = True
    ):
        """
        Args:
            trajectories: Array of shape (N, T, 3).
            t_eval: Time grid of shape (T,).
            indices: Optional time indices to subsample (for irregular sampling).
            compute_deriv: If True, compute derivatives using lorenz_system.
        """
        self.trajectories = trajectories
        self.t_eval = t_eval
        self.indices = indices if indices is not None else np.arange(len(t_eval))
        self.compute_deriv = compute_deriv

        if compute_deriv:
            # Precompute derivatives for all trajectories at selected time points
            self.derivatives = np.zeros((len(trajectories), len(self.indices), 3))
            for i, traj in enumerate(trajectories):
                for j, idx in enumerate(self.indices):
                    self.derivatives[i, j] = lorenz_system(t_eval[idx], traj[idx])
        else:
            self.derivatives = None

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys: 't' (Tensor), 'x' (Tensor), 'dx' (Tensor, optional).
        """
        traj = self.trajectories[idx]
        t = self.t_eval[self.indices]
        x = traj[self.indices]

        sample = {
            't': torch.from_numpy(t).float(),
            'x': torch.from_numpy(x).float()
        }

        if self.compute_deriv and self.derivatives is not None:
            sample['dx'] = torch.from_numpy(self.derivatives[idx]).float()

        return sample


# STEP 5: DataLoaders
def create_dataloaders(
    trajectories: np.ndarray,
    t_eval: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    irregular: bool = False,
    n_irregular_samples: int = 50,
    irregular_strategy: str = 'random',
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with optional irregular time sampling.
    """
    rng = np.random.default_rng(seed=seed)
    N = len(trajectories)
    indices = rng.permutation(N)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if irregular:
        sampled_indices = sample_irregular_timepoints(t_eval, n_irregular_samples, irregular_strategy, seed)
    else:
        sampled_indices = None

    train_dataset = LorenzDataset(trajectories[train_idx], t_eval, indices=sampled_indices, compute_deriv=True)
    val_dataset = LorenzDataset(trajectories[val_idx], t_eval, indices=sampled_indices, compute_deriv=True)
    test_dataset = LorenzDataset(trajectories[test_idx], t_eval, indices=sampled_indices, compute_deriv=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# STEP 6: Save/Load
def save_trajectories(t_eval: np.ndarray, trajectories: np.ndarray, path: str) -> None:
    """Save trajectories to .npz file."""
    np.savez_compressed(path, t_eval=t_eval, trajectories=trajectories)


def load_trajectories(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load trajectories from .npz file."""
    data = np.load(path)
    return data['t_eval'], data['trajectories']