"""
Trainers for the two learned models (window-based trajectory matching).

  DiscreteTrainer   : one-step MSE + multi-step rollout MSE of the learned map.
  TrajectoryTrainer : window trajectory matching through the ODE solver (NODE).

Both are supervised by observed states only; the Neural ODE's optional
analytic-derivative term is off by default so the supervision stays fair.
"""
from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from model import integrate_neural_ode
from data_env import pendulum_system_batch


class EarlyStopping:
    """Track a validation score; flag `early_stop` when it stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-5, mode: str = 'min', relative: bool = True):
        self.patience, self.min_delta = patience, min_delta
        self.mode, self.relative = mode, relative
        self.counter, self.best_score, self.best_state = 0, None, None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module | None = None) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.relative and self.best_score != 0:
            improved = abs(score - self.best_score) / abs(self.best_score) > self.min_delta
        elif self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score, self.counter = score, 0
            if model is not None:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

    def restore_best(self, model: nn.Module) -> None:
        """Roll the model back to the best validation state seen so far."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class TrajectoryTrainer:
    """Neural ODE trainer: match observed states over short integrated windows.

    Loss = MSE between the solved trajectory and the observed states inside each
    window. Windows use the *actual* timestamps (relative to their first sample),
    so irregular sampling needs no special handling in the solver. The optional
    analytic-derivative term is off by default (`use_derivative_loss=False`) so
    the ODE is supervised by observed states only — same information as the
    discrete MLP.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: Optimizer, device: torch.device, scheduler: Any = None,
                 window_size: int = 50, window_size_start: int = 15, n_windows_per_batch: int = 128,
                 integration_method: str = 'dopri5', rtol: float = 1e-4, atol: float = 1e-6,
                 use_adjoint: bool = True, adjoint_method: str = 'rk4', curriculum_epochs: int = 50,
                 use_derivative_loss: bool = False, time_scale: float = 1.0, seed: int = 42):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.optimizer, self.device, self.scheduler = optimizer, device, scheduler
        self.window_size, self.window_size_start = window_size, window_size_start
        self.n_windows_per_batch, self.curriculum_epochs = n_windows_per_batch, curriculum_epochs
        self.integration_method, self.rtol, self.atol = integration_method, rtol, atol
        self.use_adjoint, self.adjoint_method = use_adjoint, adjoint_method
        self.use_derivative_loss, self.seed = use_derivative_loss, seed
        # time_scale = mean sampling gap / dt (>1 for the irregular view). It
        # keeps a window's *time span* equal across data views, so both models
        # face the same integration horizon.
        self.time_scale = max(time_scale, 1e-6)
        self.history = {'train_loss': [], 'val_loss': []}

    def current_window_size(self, epoch: int) -> int:
        """Window length in samples at `epoch` (curriculum ramp, scaled by time_scale)."""
        if self.curriculum_epochs <= 0 or epoch >= self.curriculum_epochs:
            base = self.window_size
        else:
            frac = epoch / self.curriculum_epochs
            base = int(round(self.window_size_start + frac * (self.window_size - self.window_size_start)))
        return max(2, int(round(base / self.time_scale)))

    def sample_windows(self, x: Tensor, t: Tensor, lengths: Tensor, n_windows: int,
                       window_size: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Random windows from padded trajectories -> (x_windows, relative times, valid mask)."""
        B, T_max, _ = x.shape
        win = min(window_size if window_size is not None else self.window_size, T_max - 1)

        # Only sample from trajectories long enough: a window running past the
        # end would pick up padded (zero) timestamps, which breaks the
        # strictly-increasing time requirement of the ODE solver.
        valid_b = torch.nonzero(lengths >= win).flatten()
        if valid_b.numel() == 0:
            win = max(int(torch.min(lengths).item()), 2)
            valid_b = torch.nonzero(lengths >= win).flatten()

        b_idx = valid_b[torch.randint(0, valid_b.numel(), (n_windows,), device=self.device)]
        max_start = lengths[b_idx] - win
        s_idx = (torch.rand(n_windows, device=self.device) * (max_start + 1)).long()
        s_idx = torch.minimum(s_idx, max_start)

        idx = s_idx[:, None] + torch.arange(win, device=self.device)  # (n, win)
        b_exp = b_idx[:, None].expand(-1, win)
        x_windows = x[b_exp, idx]  # (n, win, 2)

        end_idx = s_idx[:, None] + win
        valid_mask = torch.arange(win, device=self.device)[None, :] < end_idx

        # Relative time from the ACTUAL timestamps (regular grid -> arange(win)*dt).
        t_windows = t[b_exp, idx]
        t_relative = t_windows - t_windows[:, :1]

        return x_windows, t_relative, valid_mask

    def integrate_model(self, x0: Tensor, t_eval: Tensor) -> Tensor:
        """Integrate the model from x0 over t_eval (adaptive solver)."""
        return integrate_neural_ode(self.model, x0, t_eval, method=self.integration_method,
                                    use_adjoint=self.use_adjoint, adjoint_method=self.adjoint_method,
                                    rtol=self.rtol, atol=self.atol)

    def train_step(self, x: Tensor, t: Tensor, lengths: Tensor, epoch: int = 0) -> float:
        self.optimizer.zero_grad()

        # Optional derivative matching (OFF by default for fairness)
        deriv_loss = torch.tensor(0.0, device=self.device)
        if self.use_derivative_loss:
            n_curriculum = self.curriculum_epochs
            deriv_weight = 1.0 - 0.9 * (epoch / n_curriculum) if epoch < n_curriculum else 0.1

            B, T_max, _ = x.shape
            b_idx = torch.randint(0, B, (self.n_windows_per_batch,), device=self.device)
            t_idx = torch.randint(0, T_max, (self.n_windows_per_batch,), device=self.device)
            valid = t_idx < lengths[b_idx]
            b_idx, t_idx = b_idx[valid], t_idx[valid]
            if len(b_idx) > 0:
                states = x[b_idx, t_idx]
                zero_t = torch.zeros(states.shape[0], device=states.device)
                dx_pred = self.model(zero_t, states)
                with torch.no_grad():
                    dx_true = torch.from_numpy(pendulum_system_batch(states.cpu().numpy())).to(
                        device=states.device, dtype=states.dtype)
                deriv_loss = torch.mean((dx_pred - dx_true) ** 2) * deriv_weight

        # Trajectory matching over windows
        win = self.current_window_size(epoch)
        x_windows, t_rel, valid_mask = self.sample_windows(x, t, lengths, self.n_windows_per_batch, window_size=win)
        total_traj_loss, n_valid_total = 0, 0
        for i in range(x_windows.shape[0]):
            mask_i = valid_mask[i]
            if mask_i.sum() < 2:
                continue
            pred = self.integrate_model(x_windows[i, 0], t_rel[i])
            mask_loss = mask_i[1:]
            diff = (pred[1:] - x_windows[i, 1:]) ** 2
            total_traj_loss += (diff.mean(dim=-1) * mask_loss).sum() / mask_loss.sum()
            n_valid_total += 1

        traj_loss = total_traj_loss / n_valid_total if n_valid_total > 0 else torch.tensor(0.0, device=self.device)
        total_loss = traj_loss + deriv_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return total_loss.item()

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0, 0
        for batch in self.val_loader:
            x = batch['x'].to(self.device)
            t = batch['t'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            x_windows, t_rel, valid_mask = self.sample_windows(
                x, t, lengths, min(self.n_windows_per_batch, 16),
                window_size=max(2, int(round(self.window_size / self.time_scale))))

            batch_loss, n_valid = 0, 0
            for i in range(x_windows.shape[0]):
                mask_i = valid_mask[i]
                if mask_i.sum() < 2:
                    continue
                pred = self.integrate_model(x_windows[i, 0], t_rel[i])
                mask_loss = mask_i[1:]
                diff = (pred[1:] - x_windows[i, 1:]) ** 2
                batch_loss += ((diff.mean(dim=-1) * mask_loss).sum() / mask_loss.sum()).item()
                n_valid += 1

            if n_valid > 0:
                total_loss += batch_loss / n_valid
                n_batches += 1

        self.model.train()
        return total_loss / n_batches if n_batches > 0 else 0.0

    def train(self, epochs: int, early_stopping: EarlyStopping | None = None) -> dict[str, list[float]]:
        torch.manual_seed(self.seed)
        for epoch in range(epochs):
            self.model.train()
            train_loss, n_batches = 0, 0
            for batch in self.train_loader:
                loss = self.train_step(batch['x'].to(self.device), batch['t'].to(self.device),
                                       batch['lengths'].to(self.device), epoch=epoch)
                train_loss, n_batches = train_loss + loss, n_batches + 1

            train_loss = train_loss / max(n_batches, 1)
            val_loss = self.validate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if self.scheduler:
                self.scheduler.step()

            print(f"epoch: {epoch}, train loss : {train_loss:.6f}, val_loss: {val_loss:.6f}")

            if early_stopping and early_stopping(val_loss, self.model):
                print(f"Early stop at {epoch}")
                early_stopping.restore_best(self.model)
                break

        return self.history


class DiscreteTrainer:
    """Trainer for the one-step flow map x_{k+1} = x_k + MLP(x_k).

    Loss = w_one * MSE(MLP(x_k), x_{k+1}) + w_multi * MSE(rollout(x_0), x_{1:K}).
    The rollout term is what makes the map usable over long horizons; the
    curriculum grows the rollout window from `window_size_start` to `window_size`
    over `curriculum_epochs`.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: Optimizer, device: torch.device, scheduler: Any = None,
                 window_size: int = 50, window_size_start: int = 15, n_windows_per_batch: int = 64,
                 one_step_weight: float = 0.5, multi_step_weight: float = 0.5,
                 curriculum_epochs: int = 50, seed: int = 42):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.optimizer, self.device, self.scheduler = optimizer, device, scheduler
        self.window_size, self.window_size_start = window_size, window_size_start
        self.n_windows_per_batch, self.curriculum_epochs = n_windows_per_batch, curriculum_epochs
        self.one_step_weight, self.multi_step_weight = one_step_weight, multi_step_weight
        self.seed, self.history = seed, {'train_loss': [], 'val_loss': []}

    def current_window_size(self, epoch: int) -> int:
        """Window length in samples at `epoch` (linear curriculum ramp)."""
        if self.curriculum_epochs <= 0 or epoch >= self.curriculum_epochs:
            return self.window_size
        frac = epoch / self.curriculum_epochs
        return max(2, int(round(self.window_size_start + frac * (self.window_size - self.window_size_start))))

    def sample_windows(self, x: Tensor, lengths: Tensor, n_windows: int,
                       window_size: int) -> tuple[Tensor, Tensor]:
        """Random windows from padded trajectories -> (x_windows, valid mask)."""
        B, T_max, _ = x.shape
        win = min(window_size, T_max - 1)

        # Only sample from trajectories long enough to contain a full window
        valid_b = torch.nonzero(lengths >= win).flatten()
        if valid_b.numel() == 0:
            win = max(int(torch.min(lengths).item()), 2)
            valid_b = torch.nonzero(lengths >= win).flatten()

        b_idx = valid_b[torch.randint(0, valid_b.numel(), (n_windows,), device=self.device)]
        max_start = lengths[b_idx] - win
        s_idx = (torch.rand(n_windows, device=self.device) * (max_start + 1)).long()
        s_idx = torch.minimum(s_idx, max_start)

        idx = s_idx[:, None] + torch.arange(win, device=self.device)
        x_windows = x[b_idx[:, None].expand(-1, win), idx]

        end_idx = s_idx[:, None] + win
        valid_mask = torch.arange(win, device=self.device)[None, :] < end_idx
        return x_windows, valid_mask

    def _losses(self, x_windows: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """One-step and multi-step MSE for a batch of windows."""
        mask_next = valid_mask[:, 1:].to(x_windows.dtype)  # (n, win-1)

        # One-step (teacher forced)
        pred_one = self.model(x_windows[:, :-1])
        one_step = ((pred_one - x_windows[:, 1:]) ** 2).mean(dim=-1).mul(mask_next).sum() / mask_next.sum().clamp(min=1.0)

        # Multi-step (free running rollout)
        roll = self.model.rollout(x_windows[:, 0], x_windows.shape[1] - 1)
        multi_step = ((roll[:, 1:] - x_windows[:, 1:]) ** 2).mean(dim=-1).mul(mask_next).sum() / mask_next.sum().clamp(min=1.0)

        return one_step, multi_step

    def train_step(self, x: Tensor, lengths: Tensor, epoch: int = 0) -> float:
        self.optimizer.zero_grad()
        x_windows, valid_mask = self.sample_windows(x, lengths, self.n_windows_per_batch,
                                                    window_size=self.current_window_size(epoch))
        one_step, multi_step = self._losses(x_windows, valid_mask)
        total_loss = self.one_step_weight * one_step + self.multi_step_weight * multi_step

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return total_loss.item()

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0, 0
        for batch in self.val_loader:
            x_windows, valid_mask = self.sample_windows(batch['x'].to(self.device), batch['lengths'].to(self.device),
                                                        min(self.n_windows_per_batch, 32), self.window_size)
            one_step, multi_step = self._losses(x_windows, valid_mask)
            total_loss += (self.one_step_weight * one_step + self.multi_step_weight * multi_step).item()
            n_batches += 1

        self.model.train()
        return total_loss / n_batches if n_batches > 0 else 0.0

    def train(self, epochs: int, early_stopping: EarlyStopping | None = None) -> dict[str, list[float]]:
        torch.manual_seed(self.seed)
        for epoch in range(epochs):
            self.model.train()
            train_loss, n_batches = 0, 0
            for batch in self.train_loader:
                loss = self.train_step(batch['x'].to(self.device), batch['lengths'].to(self.device), epoch=epoch)
                train_loss, n_batches = train_loss + loss, n_batches + 1

            train_loss = train_loss / max(n_batches, 1)
            val_loss = self.validate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if self.scheduler:
                self.scheduler.step()

            print(f"epoch: {epoch}, train loss : {train_loss:.6f}, val_loss: {val_loss:.6f}")

            if early_stopping and early_stopping(val_loss, self.model):
                print(f"Early stop at {epoch}")
                early_stopping.restore_best(self.model)
                break

        return self.history
