"""
JEPA World Model - Training and Evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from model import (
    Encoder,
    TargetEncoder,
    Predictor,
    LinearProbe,
    jepa_loss,
    collapse_metric,
    prediction_loss,
    variance_loss,
    covariance_loss,
    rollout_latent_dynamics,
    probe_state_recovery,
    evaluate_planner,
    JEPALossConfig,
)
from jepa.data_env import build_transition_dataset, load_dataset, split_dataset


@dataclass
class TrainingConfig:
    num_steps: int = 1500
    batch_size: int = 32
    lr: float = 1e-3
    tau: float = 0.99
    seed: int = 0
    loss_config: JEPALossConfig = None
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    def __post_init__(self):
        if self.loss_config is None:
            self.loss_config = JEPALossConfig()


@dataclass
class EvalConfig:
    n_probe_samples: int = 512
    num_probe_steps: int = 300
    n_eval_episodes: int = 10
    max_steps: int = 20
    n_sequences: int = 64
    horizon: int = 5
    n_actions: int = 4
    agent_size: int = 1


class JEPATrainer:
    """JEPA trainer with train/val logging."""

    def __init__(
        self,
        encoder: Encoder,
        target_encoder: TargetEncoder,
        predictor: Predictor,
        config: TrainingConfig,
        device: str = 'cpu'
    ):
        self.encoder = encoder.to(device)
        self.target_encoder = target_encoder.to(device)
        self.predictor = predictor.to(device)
        self.config = config
        self.device = device

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=config.lr)
        self.predictor_opt = torch.optim.Adam(self.predictor.parameters(), lr=config.lr)

        self.history = {
            'train_loss': [],
            'train_collapse': [],
            'val_loss': [],
            'val_collapse': [],
        }

    def train_step(self, batch: dict) -> tuple[float, float]:
        """Single training step."""
        obs = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_obs = batch['next_observations'].to(self.device)

        self.encoder_opt.zero_grad()
        self.predictor_opt.zero_grad()

        # Online encoder
        embeddings = self.encoder(obs)

        # Target encoder (no grad)
        with torch.no_grad():
            target = self.target_encoder(next_obs)

        # Predictor
        predicted = self.predictor(embeddings, actions)

        # Loss
        loss = jepa_loss(predicted, target, embeddings, self.config.loss_config)

        loss.backward()
        self.encoder_opt.step()
        self.predictor_opt.step()

        # EMA update
        self.target_encoder.update(self.encoder, self.config.tau)

        # Collapse metric
        collapse = collapse_metric(embeddings.detach()).item()

        return loss.item(), collapse

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Compute validation loss and collapse."""
        self.encoder.eval()
        self.target_encoder.eval()
        self.predictor.eval()

        total_loss = 0.0
        total_collapse = 0.0
        n_batches = 0

        for batch in val_loader:
            obs = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            next_obs = batch['next_observations'].to(self.device)

            embeddings = self.encoder(obs)
            target = self.target_encoder(next_obs)
            predicted = self.predictor(embeddings, actions)

            loss = jepa_loss(predicted, target, embeddings, self.config.loss_config)
            collapse = collapse_metric(embeddings).item()

            total_loss += loss.item()
            total_collapse += collapse
            n_batches += 1

        self.encoder.train()
        self.target_encoder.train()
        self.predictor.train()

        return total_loss / n_batches, total_collapse / n_batches

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Full training loop with validation and early stopping."""
        torch.manual_seed(self.config.seed)

        best_val_loss = float('inf')
        patience_counter = 0

        for step in range(self.config.num_steps):
            for batch in train_loader:
                loss, collapse = self.train_step(batch)
                self.history['train_loss'].append(loss)
                self.history['train_collapse'].append(collapse)
                break  # One batch per step

            if val_loader is not None and (step + 1) % 50 == 0:
                val_loss, val_collapse = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_collapse'].append(val_collapse)
                print(f"Step {step+1}: train_loss={loss:.4f}, train_collapse={collapse:.4f}, "
                      f"val_loss={val_loss:.4f}, val_collapse={val_collapse:.4f}")

                # Early stopping check
                if val_loss < best_val_loss - self.config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  Early stopping patience: {patience_counter}/{self.config.early_stopping_patience}")

                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered at step {step+1}!")
                    break
            elif (step + 1) % 50 == 0:
                print(f"Step {step+1}: train_loss={loss:.4f}, train_collapse={collapse:.4f}")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'predictor': self.predictor.state_dict(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'predictor_opt': self.predictor_opt.state_dict(),
            'history': self.history,
            'config': asdict(self.config),
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.encoder_opt.load_state_dict(checkpoint['encoder_opt'])
        self.predictor_opt.load_state_dict(checkpoint['predictor_opt'])
        self.history = checkpoint['history']


def create_dataloader(dataset: dict, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create DataLoader from dataset dict."""
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return self.data['observations'].shape[0]

        def __getitem__(self, idx):
            return {
                'observations': self.data['observations'][idx],
                'actions': self.data['actions'][idx],
                'next_observations': self.data['next_observations'][idx],
            }

    ds = DictDataset(dataset)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def multi_step_prediction_error(
    dataset: dict,
    encoder: Encoder,
    target_encoder: TargetEncoder,
    predictor: Predictor,
    horizon: int = 5,
    num_samples: int = 32,
    device: str = 'cpu'
) -> float:
    """Compute multi-step prediction error in latent space."""
    encoder.eval()
    target_encoder.eval()
    predictor.eval()

    N = len(dataset['actions'])
    num_samples = min(num_samples, N - horizon + 1)

    indices = torch.arange(num_samples)

    obs = dataset['observations'][indices].to(device)
    actions = dataset['actions'][indices[:, None] + torch.arange(horizon)].to(device)
    future_obs = dataset['next_observations'][indices[:, None] + torch.arange(horizon)].to(device)

    embeddings = encoder(obs)

    rollout = rollout_latent_dynamics(embeddings, actions, predictor)[1:]

    B, H, C, W, W2 = future_obs.shape
    with torch.no_grad():
        target = target_encoder(future_obs.reshape(B * H, C, W, W2))

    target = target.reshape(B, H, -1).transpose(0, 1)

    error = (rollout - target).pow(2).sum(dim=-1).mean()

    encoder.train()
    target_encoder.train()
    predictor.train()

    return error.item()


@torch.no_grad()
def collect_test_predictions(
    test_dataset: dict,
    encoder: Encoder,
    predictor: Predictor,
    room_size: int,
    n_sequences: int,
    horizon: int,
    n_actions: int,
    max_steps: int,
    n_success: int = 5,
    n_fail: int = 5,
    device: str = 'cpu'
) -> dict:
    """Collect successful and unsuccessful test predictions."""
    from jepa.model import encode_goal, sample_action_sequences, score_action_sequences, select_best_plan
    from jepa.data_env import env_reset, env_step

    encoder.eval()
    predictor.eval()

    successful = []
    unsuccessful = []

    # First try goals from test dataset
    for ep in range(len(test_dataset['actions'])):
        if len(successful) >= n_success and len(unsuccessful) >= n_fail:
            break

        goal_pos = test_dataset['states'][ep, :].float()
        g_x, g_y = int(goal_pos[0].item()), int(goal_pos[1].item())

        state, obs = env_reset(room_size=room_size, seed=ep + 10000)
        goal_state = torch.tensor([g_x, g_y], dtype=torch.float32)

        goal_embedding = encode_goal(goal_state, encoder, room_size)

        trajectory = []
        success = False

        for step in range(max_steps):
            if int(state[0].item()) == g_x and int(state[1].item()) == g_y:
                success = True
                break

            start_embedding = encoder(obs.unsqueeze(0).to(device)).squeeze(0)

            action_seq = sample_action_sequences(n_sequences, horizon, n_actions)
            costs = score_action_sequences(start_embedding, action_seq, goal_embedding, predictor)
            best_plan = select_best_plan(action_seq, costs)
            action = best_plan[0].item()

            state, obs = env_step(state, action, room_size=room_size)
            trajectory.append({
                'state': (float(state[0].item()), float(state[1].item())),
                'action': int(action)
            })

        success = bool(int(state[0].item()) == g_x and int(state[1].item()) == g_y)

        result = {
            'goal': (g_x, g_y),
            'trajectory': trajectory,
            'success': success,
            'steps': len(trajectory)
        }

        if success and len(successful) < n_success:
            successful.append(result)
        elif not success and len(unsuccessful) < n_fail:
            unsuccessful.append(result)

    # If still need more, sample random goals
    ep_offset = 20000
    while (len(successful) < n_success or len(unsuccessful) < n_fail) and ep_offset < 30000:
        g_x, g_y = torch.randint(0, room_size, (2,)).tolist()
        goal_state = torch.tensor([g_x, g_y], dtype=torch.float32)

        state, obs = env_reset(room_size=room_size, seed=ep_offset)
        ep_offset += 1

        goal_embedding = encode_goal(goal_state, encoder, room_size)

        trajectory = []
        success = False

        for step in range(max_steps):
            if int(state[0].item()) == g_x and int(state[1].item()) == g_y:
                success = True
                break

            start_embedding = encoder(obs.unsqueeze(0).to(device)).squeeze(0)

            action_seq = sample_action_sequences(n_sequences, horizon, n_actions)
            costs = score_action_sequences(start_embedding, action_seq, goal_embedding, predictor)
            best_plan = select_best_plan(action_seq, costs)
            action = best_plan[0].item()

            state, obs = env_step(state, action, room_size=room_size)
            trajectory.append({
                'state': (float(state[0].item()), float(state[1].item())),
                'action': int(action)
            })

        success = bool(int(state[0].item()) == g_x and int(state[1].item()) == g_y)

        result = {
            'goal': (g_x, g_y),
            'trajectory': trajectory,
            'success': success,
            'steps': len(trajectory)
        }

        if success and len(successful) < n_success:
            successful.append(result)
        elif not success and len(unsuccessful) < n_fail:
            unsuccessful.append(result)

    encoder.train()
    predictor.train()

    return {
        'successful': successful[:n_success],
        'unsuccessful': unsuccessful[:n_fail]
    }


def jepa_world_model_experiment(
    room_size: int = 6,
    latent_dim: int = 16,
    n_train_transitions: int = 512,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_steps: int = 1500,
    lr: float = 1e-3,
    tau: float = 0.99,
    seed: int = 0,
    n_probe_samples: int = 512,
    num_probe_steps: int = 300,
    n_eval_episodes: int = 10,
    max_steps: int = 20,
    n_sequences: int = 64,
    horizon: int = 5,
    save_dir: str = './jepa_output',
    device: str = 'cpu',
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4
) -> dict:
    """
    Full JEPA experiment with train/val/test split, metrics logging, and test predictions.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # Build or load dataset
    dataset_path = save_path / 'data.pt'
    if dataset_path.exists():
        print(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
    else:
        print("Building dataset...")
        dataset = build_transition_dataset(
            num_transitions=n_train_transitions,
            room_size=room_size,
            seed=seed,
            save_path=str(dataset_path)
        )

    # Split dataset
    train_data, val_data, test_data = split_dataset(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    print(f"Train: {train_data['observations'].shape[0]}, "
          f"Val: {val_data['observations'].shape[0]}, "
          f"Test: {test_data['observations'].shape[0]}")

    # Create models
    encoder = Encoder(obs_channels=1, room_size=room_size, latent_dim=latent_dim)
    target_encoder = TargetEncoder(encoder)
    predictor = Predictor(latent_dim=latent_dim, action_dim=4, hidden_dim=64)

    # Train
    config = TrainingConfig(
        num_steps=num_steps,
        batch_size=batch_size,
        lr=lr,
        tau=tau,
        seed=seed,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta
    )

    trainer = JEPATrainer(encoder, target_encoder, predictor, config, device)

    train_loader = create_dataloader(train_data, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size, shuffle=False)

    print("Training...")
    trainer.train(train_loader, val_loader)

    # Save checkpoint
    trainer.save_checkpoint(str(save_path / 'checkpoint.pt'))

    # Save history
    with open(save_path / 'history.json', 'w') as f:
        json.dump(trainer.history, f, indent=2)

    # Probe on test set
    print("Evaluating linear probe...")
    probe_data = {k: v[:n_probe_samples] for k, v in test_data.items()}
    probe_results = probe_state_recovery(probe_data, encoder, predictor, num_probe_steps=num_probe_steps)

    # R² score
    states = probe_data['states']
    state_variance = torch.mean((states - states.mean(dim=0, keepdim=True)) ** 2)
    probe_mse = torch.tensor(probe_results['mse'], dtype=state_variance.dtype)
    probe_r2 = 1.0 - (probe_mse / (state_variance + 1e-8))

    # Multi-step prediction error
    print("Computing multi-step prediction error...")
    mse_error = multi_step_prediction_error(
        test_data, encoder, target_encoder, predictor,
        horizon=horizon, num_samples=32, device=device
    )

    # Planner evaluation
    print("Evaluating planner...")
    planner_results = evaluate_planner(
        encoder, predictor, room_size=room_size,
        agent_size=1, n_episodes=n_eval_episodes,
        max_steps=max_steps, n_sequences=n_sequences,
        horizon=horizon, n_actions=4
    )

    # Collect test predictions
    print("Collecting test predictions...")
    test_predictions = collect_test_predictions(
        test_data, encoder, predictor, room_size,
        n_sequences, horizon, 4, max_steps,
        n_success=5, n_fail=5, device=device
    )

    # Save test predictions
    with open(save_path / 'test_predictions.json', 'w') as f:
        json.dump(test_predictions, f, indent=2)

    # Final results
    results = {
        'train_losses': trainer.history['train_loss'],
        'train_collapse': trainer.history['train_collapse'],
        'val_losses': trainer.history['val_loss'],
        'val_collapse': trainer.history['val_collapse'],
        'probe_r2': float(probe_r2.item()),
        'probe_mse': probe_results['mse'],
        'probe_mae': probe_results['mean_abs_error'],
        'multi_step_mse': mse_error,
        'success_rate': planner_results['success_rate'],
        'mean_steps': planner_results['mean_steps'],
        'mean_final_distance': planner_results['mean_final_distance'],
    }

    with open(save_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k}: [{v[0]:.4f} -> {v[-1]:.4f}] ({len(v)} steps)")
        else:
            print(f"  {k}: {v}")

    return results