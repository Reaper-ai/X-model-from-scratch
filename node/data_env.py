"""
Damped Simple Pendulum data generation, IC sampling, and Dataset classes.
"""
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp


# Damped Pendulum System
def pendulum_system(t: float, state: np.ndarray, g: float = 9.81, L: float = 1.0, b: float = 0.1) -> np.ndarray:
    """
    Damped simple pendulum ODE: d/dt [θ, ω] = [ω, -(g/L)*sin(θ) - b*ω]

    Args:
        t: Time (unused, system is autonomous).
        state: Current state [θ, ω] of shape (2,).
        g: Gravitational acceleration (default 9.81).
        L: Pendulum length (default 1.0).
        b: Damping coefficient (default 0.1).

    Returns:
        Derivatives [dθ/dt, dω/dt] of shape (2,).
    """
    import torch
    
    # Handle both numpy arrays and torch tensors
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    
    theta, omega = state
    dtheta = omega
    domega = -(g / L) * np.sin(theta) - b * omega
    return np.array([dtheta, domega])


def pendulum_system_batch(states: np.ndarray, g: float = 9.81, L: float = 1.0, b: float = 0.1) -> np.ndarray:
    """
    Vectorized pendulum derivative for batch of states.

    Args:
        states: (N, 2) array of [θ, ω].

    Returns:
        (N, 2) array of [dθ/dt, dω/dt].
    """
    theta = states[:, 0]
    omega = states[:, 1]
    dtheta = omega
    domega = -(g / L) * np.sin(theta) - b * omega
    return np.stack([dtheta, domega], axis=1)


def pendulum_energy( state: np.ndarray, g: float = 9.81, L: float = 1.0) -> float:
    """
    Compute total mechanical energy: E = 0.5 * ω² + g/L * (1 - cos(θ))

    Args:
        state: State [θ, ω] of shape (2,) or (N, 2).
        g: Gravitational acceleration.
        L: Pendulum length.

    Returns:
        Energy value(s).
    """
    theta = state[..., 0]
    omega = state[..., 1]
    return 0.5 * omega**2 + (g / L) * (1 - np.cos(theta))


# Generate Trajectories (Ground Truth)
def generate_trajectories(n_trajectories: int = 300, k_periods_train: int = 3,
    test_B_max_periods: int = 5,
    test_C_ic_range_factor: float = 1.5,
    dt: float = 0.05,
    noise_std: float = 0.0,
    seed: int = 42,
    g: float = 9.81,
    L: float = 1.0,
    b: float = 0.1,
    train_theta_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
    train_omega_range: Tuple[float, float] = (-2.0, 2.0),
    symmetric_augment: bool = True
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Generate damped pendulum trajectories with physics-based lengths and 4-way labels.

    Trajectory length t is a random variable sampled per trajectory.
    Each trajectory may have a different length (t1, t2, ...).

    Splits:
        Train:  ICs in train_theta_range x train_omega_range (default ±π/2 × ±2),
                length up to k*T
        Test A: unseen ICs in same range, length up to k*T (in-distribution)
        Test B: ICs same as train, length up to m*T (time extrapolation, m > k)
        Test C: ICs at test_C_ic_range_factor x train range, length up to k*T
                (state extrapolation)

    Args:
        n_trajectories: Total number of trajectories across all splits.
        k_periods_train: Training/test-A/test-C max periods (k).
        test_B_max_periods: Test B max periods (m). Must be > k_periods_train.
        test_C_ic_range_factor: Multiplier on training IC range for Test C.
        dt: Sampling rate.
        noise_std: Gaussian noise std (0 = no noise).
        seed: Random seed.
        g, L, b: Pendulum parameters.
        train_theta_range: IC sampling range for θ0 of train/test_A/test_B.
        train_omega_range: IC sampling range for ω0 of train/test_A/test_B.
        symmetric_augment: If True, add (−θ,−ω) mirror of each TRAIN trajectory.
            The pendulum is symmetric under (θ,ω)→(−θ,−ω); this forces the
            learned vector field to respect that symmetry.

    Returns:
        t_eval: Common fine time grid covering longest trajectory, shape (T_max,).
        trajectories: List of (T_i, 2) arrays, variable length per trajectory.
        traj_lengths: Array of trajectory lengths in time steps, shape (N,).
        ic_labels: Array of split labels 'train'|'test_A'|'test_B'|'test_C', shape (N,).
    """
    rng = np.random.default_rng(seed=seed)

    # Pendulum period: T = 2π√(L/g)
    period = 2.0 * np.pi * np.sqrt(L / g)  # ≈ 2.007s for L=1.0, g=9.81

    # Max times per split
    train_max_t = k_periods_train * period
    test_B_max_t = test_B_max_periods * period

    # IC ranges (train box configurable; test_C scales it)
    train_theta = train_theta_range
    train_omega = train_omega_range
    test_C_theta = (-test_C_ic_range_factor * train_theta[1], test_C_ic_range_factor * train_theta[1])
    test_C_omega = (-test_C_ic_range_factor * train_omega[1], test_C_ic_range_factor * train_omega[1])

    # Split counts (70/10/10/10)
    n_train = int(n_trajectories * 0.70)
    n_test_A = int(n_trajectories * 0.10)
    n_test_B = int(n_trajectories * 0.10)
    n_test_C = n_trajectories - n_train - n_test_A - n_test_B

    def _simulate(theta_range, omega_range, max_t, min_frac=0.5):
        theta0 = rng.uniform(*theta_range)
        omega0 = rng.uniform(*omega_range)
        x0 = np.array([theta0, omega0])

        # Random trajectory length: uniform in [min_frac * max_t, max_t]
        traj_t = rng.uniform(min_frac * max_t, max_t)
        n_steps = int(round(traj_t / dt)) + 1
        t_i = np.arange(n_steps) * dt

        sol = solve_ivp(fun=lambda t, s: pendulum_system(t, s, g, L, b), t_span=(0.0, t_i[-1]), y0=x0, t_eval=t_i, method='RK45', rtol=1e-9, atol=1e-12)
        traj = sol.y.T  # (n_steps, 2)
        if noise_std > 0:
            traj += rng.normal(0, noise_std, traj.shape)
        return traj

    trajectories: List[np.ndarray] = []
    ic_labels: List[str] = []

    for _ in range(n_train):
        traj = _simulate(train_theta, train_omega, train_max_t)
        trajectories.append(traj)
        ic_labels.append('train')
        if symmetric_augment:
            # (θ,ω)→(−θ,−ω) mirror: exact symmetry of the damped pendulum
            trajectories.append(-traj)
            ic_labels.append('train')

    for _ in range(n_test_A):
        trajectories.append(_simulate(train_theta, train_omega, train_max_t))
        ic_labels.append('test_A')

    for _ in range(n_test_B):
        # Same ICs as train, longer trajectory (extrapolation)
        trajectories.append(_simulate(train_theta, train_omega, test_B_max_t))
        ic_labels.append('test_B')

    for _ in range(n_test_C):
        trajectories.append(_simulate(test_C_theta, test_C_omega, train_max_t))
        ic_labels.append('test_C')

    traj_lengths = np.array([len(t) for t in trajectories], dtype=np.int64)

    # Common fine time grid covering the longest trajectory
    t_eval = np.arange(traj_lengths.max()) * dt

    return t_eval, trajectories, traj_lengths, np.array(ic_labels)


# Irregular Time Sampling
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


def make_irregular_view( trajectories: List[np.ndarray], t_eval: np.ndarray, traj_lengths: np.ndarray, frac: float = 0.5, strategy: str = 'random', seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Build an irregularly-sampled view of the *same* trajectories.

    Every trajectory keeps its identity (same IC, same underlying physics) but
    is subsampled onto its own irregular timestamps. The first sample is always
    kept (index 0) so the initial condition is always available.

    Args:
        trajectories: List of (T_i, 2) arrays (regular grid).
        t_eval: Common regular time grid of shape (T_max,).
        traj_lengths: Actual length of each trajectory.
        frac: Fraction of regular samples kept (0 < frac <= 1).
        strategy: Sampling strategy ('random' | 'poisson' | 'log').
        seed: Base seed (per-trajectory offset added).

    Returns:
        times: List of (T'_i,) irregular time arrays.
        xs: List of (T'_i, 2) irregular state arrays.
        lengths: (N,) array of irregular lengths.
    """
    times: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    lengths: List[int] = []

    for i, traj in enumerate(trajectories):
        L = int(traj_lengths[i])
        n = max(2, int(round(frac * L)))
        idx = sample_irregular_timepoints(t_eval[:L], n, strategy, seed + i)
        # Always keep the initial condition
        idx = np.unique(np.concatenate([np.array([0]), idx.astype(np.int64)]))
        times.append(t_eval[idx].copy())
        xs.append(traj[idx].copy())
        lengths.append(len(idx))

    return times, xs, np.array(lengths, dtype=np.int64)

class PendulumDataset(Dataset):
    """
    PyTorch Dataset for variable-length pendulum trajectories.
    Returns (t, x, length, label) per sample.

    If `times` is provided, each trajectory carries its own (possibly
    irregular) timestamps instead of a shared regular grid.
    """
    def __init__( self, trajectories: List[np.ndarray], t_eval: np.ndarray, traj_lengths: np.ndarray, ic_labels: np.ndarray, mask: Optional[np.ndarray] = None,
        compute_deriv: bool = True,
        g: float = 9.81,
        L: float = 1.0,
        b: float = 0.1,
        times: Optional[List[np.ndarray]] = None
    ):
        """
        Args:
            trajectories: List of (T_i, 2) arrays (variable length).
            t_eval: Common time grid of shape (T_max,).
            traj_lengths: Array of actual lengths, shape (N,).
            ic_labels: Array of split labels, shape (N,).
            mask: Boolean mask selecting which trajectories to include.
            compute_deriv: If True, precompute true derivatives.
            g, L, b: Pendulum parameters.
            times: Optional list of per-trajectory time arrays (irregular view).
        """
        if mask is not None:
            sel = np.where(mask)[0]
            self.trajectories = [trajectories[i] for i in sel]
            self.traj_lengths = traj_lengths[mask]
            self.ic_labels = ic_labels[mask]
            self.times = [times[i] for i in sel] if times is not None else None
        else:
            self.trajectories = trajectories
            self.traj_lengths = traj_lengths
            self.ic_labels = ic_labels
            self.times = times

        self.t_eval = t_eval
        self.compute_deriv = compute_deriv
        self.g = g
        self.L = L
        self.b = b

        if compute_deriv:
            self.derivatives = [pendulum_system_batch(traj, g, L, b) for traj in self.trajectories]
        else:
            self.derivatives = None

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        length = self.traj_lengths[idx]
        
        if self.times is not None:
            t = self.times[idx][:length]
        else:
            t = self.t_eval[:length]
        x = traj[:length]

        sample = {
            't': torch.from_numpy(np.ascontiguousarray(t)).float(),
            'x': torch.from_numpy(np.ascontiguousarray(x)).float(),
            'length': torch.tensor(length, dtype=torch.long),
            'label': self.ic_labels[idx],
        }

        if self.compute_deriv and self.derivatives is not None:
            sample['dx'] = torch.from_numpy(self.derivatives[idx]).float()

        return sample


def collate_variable_length(batch: List[Dict]) -> Dict[str, Any]:
    """Pad variable-length trajectories to the max length in the batch."""
    lengths = [b['x'].shape[0] for b in batch]
    max_len = max(lengths)
    B = len(batch)
    dim = batch[0]['x'].shape[1]

    x_pad = torch.zeros(B, max_len, dim)
    t_pad = torch.zeros(B, max_len)
    dx_pad = torch.zeros(B, max_len, dim) if 'dx' in batch[0] else None

    for i, b in enumerate(batch):
        L_i = b['x'].shape[0]
        x_pad[i, :L_i] = b['x']
        t_pad[i, :L_i] = b['t']
        if dx_pad is not None:
            dx_pad[i, :L_i] = b['dx']

    out = {'x': x_pad, 't': t_pad, 'lengths': torch.tensor(lengths, dtype=torch.long), 'labels': [b['label'] for b in batch]}
    
    if dx_pad is not None: out['dx'] = dx_pad
    
    return out


# STEP 5: DataLoaders with Temporal Split
def get_split_masks( ic_labels: np.ndarray, val_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic train/validation split of the 'train' label group.

    Both the regular view (discrete MLP) and the irregular view (Neural ODE)
    call this with the same ic_labels, so they share *identical* membership.
    """
    train_indices = np.where(ic_labels == 'train')[0]
    rng = np.random.default_rng(0)
    rng.shuffle(train_indices)
    n_val = max(1, int(len(train_indices) * val_fraction))
    val_idx, train_idx = train_indices[:n_val], train_indices[n_val:]

    train_mask = np.zeros(len(ic_labels), dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(len(ic_labels), dtype=bool)
    val_mask[val_idx] = True

    return train_mask, val_mask


def create_dataloaders( trajectories: List[np.ndarray], t_eval: np.ndarray,
    traj_lengths: np.ndarray,
    ic_labels: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    g: float = 9.81,
    L: float = 1.0,
    b: float = 0.1,
    val_fraction: float = 0.1,
    times: Optional[List[np.ndarray]] = None,
    compute_deriv: bool = False,
    split_masks: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create 5 DataLoaders for the physics-based split.

    Returns:
        train_loader, val_loader, test_A_loader, test_B_loader, test_C_loader

    Splits by ic_labels: 'train', 'test_A', 'test_B', 'test_C'.
    A validation set is held out from 'train' via val_fraction.

    If `times` is given, each sample carries its own (irregular) timestamps.
    """
    def _make_loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        ds = PendulumDataset(
            trajectories, t_eval, traj_lengths, ic_labels,
            mask=mask, compute_deriv=compute_deriv, g=g, L=L, b=b, times=times
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=collate_variable_length
        )

    if split_masks is None:
        split_masks = get_split_masks(ic_labels, val_fraction)
    train_mask, val_mask = split_masks

    train_loader = _make_loader(train_mask, shuffle=True)
    val_loader = _make_loader(val_mask, shuffle=True)
    test_A_loader = _make_loader(ic_labels == 'test_A', shuffle=False)
    test_B_loader = _make_loader(ic_labels == 'test_B', shuffle=False)
    test_C_loader = _make_loader(ic_labels == 'test_C', shuffle=False)

    print(
        f"DataLoaders: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test_A={len(test_A_loader.dataset)}, "
        f"test_B={len(test_B_loader.dataset)}, "
        f"test_C={len(test_C_loader.dataset)}"
    )

    return train_loader, val_loader, test_A_loader, test_B_loader, test_C_loader


# STEP 6: Save/Load
def save_trajectories(t_eval: np.ndarray, trajectories: List[np.ndarray], traj_lengths: np.ndarray, ic_labels: np.ndarray, path: str) -> None:
    """Save variable-length trajectories to .npz file."""
    np.savez_compressed(path, t_eval=t_eval, lengths=traj_lengths, labels=ic_labels, **{f'traj_{i}': t for i, t in enumerate(trajectories)})


def load_trajectories(path: str) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Load variable-length trajectories from .npz file."""
    data = np.load(path, allow_pickle=True)
    t_eval = data['t_eval']
    traj_lengths = data['lengths']
    ic_labels = data['labels']
    trajectories = [data[f'traj_{i}'] for i in range(len(traj_lengths))]
    return t_eval, trajectories, traj_lengths, ic_labels