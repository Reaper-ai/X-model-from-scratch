"""
JEPA World Model - Models and Loss Functions (Class-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


class Encoder(nn.Module):
    """Online encoder network."""

    def __init__(
        self,
        obs_channels: int = 1,
        room_size: int = 8,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.room_size = room_size
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(obs_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        conv2_out_size = room_size // 2
        flattened_size = 32 * conv2_out_size * conv2_out_size

        self.fc = nn.Linear(flattened_size, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.fc(x)
        return x


class TargetEncoder(nn.Module):
    """EMA target encoder network."""

    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = Encoder(
            obs_channels=encoder.obs_channels,
            room_size=encoder.room_size,
            latent_dim=encoder.latent_dim,
        )
        self.encoder.load_state_dict(encoder.state_dict())
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: Encoder, tau: float = 0.99):
        for target_param, online_param in zip(self.encoder.parameters(), online_encoder.parameters()):
            target_param.data.mul_(tau).add_(online_param.data, alpha=1 - tau)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)


class Predictor(nn.Module):
    """Action-conditioned predictor network."""

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.action_embed = nn.Embedding(action_dim, latent_dim)
        self.fc1 = nn.Linear(2 * latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.action_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_embeds = self.action_embed(actions)
        x = torch.cat([embeddings, action_embeds], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearProbe(nn.Module):
    """Linear probe for state recovery from latents."""

    def __init__(self, latent_dim: int = 32, state_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(latent_dim, state_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


@dataclass
class JEPALossConfig:
    pred_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04
    gamma: float = 1.0
    eps: float = 1e-4


def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss between predicted and target embeddings."""
    return F.mse_loss(predicted, target)


def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Variance regularization to prevent collapse."""
    std = torch.sqrt(torch.var(embeddings, dim=0) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Covariance regularization to decorrelate dimensions."""
    mean = embeddings.mean(dim=0)
    z = embeddings - mean
    B = embeddings.shape[0]
    cov = torch.mm(z.t(), z) / (B - 1)
    D = embeddings.shape[1]
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


def vicreg_regularizer(
    embeddings: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    gamma: float = 1.0,
    eps: float = 1e-4
) -> torch.Tensor:
    """VICReg regularization combining variance and covariance losses."""
    var = variance_loss(embeddings, gamma, eps)
    cov = covariance_loss(embeddings)
    return var_weight * var + cov_weight * cov


def jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    online_embeddings: torch.Tensor,
    config: Optional[JEPALossConfig] = None
) -> torch.Tensor:
    """Full JEPA loss: prediction + VICReg regularization."""
    if config is None:
        config = JEPALossConfig()

    pred_loss = prediction_loss(predicted, target)
    reg_loss = vicreg_regularizer(
        online_embeddings,
        var_weight=config.var_weight,
        cov_weight=config.cov_weight,
        gamma=config.gamma
    )
    return config.pred_weight * pred_loss + reg_loss


def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    """Average std across latent dimensions (higher = less collapsed)."""
    return embeddings.std(dim=0).mean()


def train_linear_probe(
    probe: LinearProbe,
    embeddings: torch.Tensor,
    states: torch.Tensor,
    num_steps: int = 100,
    lr: float = 1e-2
) -> LinearProbe:
    """Train linear probe on frozen embeddings."""
    embeddings = embeddings.detach()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(num_steps):
        pred = probe(embeddings)
        loss = F.mse_loss(pred, states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return probe


def probe_state_recovery(
    dataset: dict,
    encoder: Encoder,
    predictor: Predictor,
    probe: Optional[LinearProbe] = None,
    num_probe_steps: int = 100
) -> dict:
    """Evaluate linear probe on state recovery using predictor's next-step predictions."""
    obs = dataset['observations']
    actions = dataset['actions']
    states = dataset['states']

    embeddings = encoder(obs)
    D = embeddings.shape[1]

    if probe is None:
        probe = LinearProbe(latent_dim=D, state_dim=states.shape[-1])

    # Use predictor's single-step predictions as probe input
    with torch.no_grad():
        predicted_latents = predictor(embeddings, actions)

    probe = train_linear_probe(probe, predicted_latents, states, num_probe_steps)

    predicted_states = probe(predicted_latents)
    mse = F.mse_loss(predicted_states, states).item()
    mae = F.l1_loss(predicted_states, states).item()

    return {
        'mse': mse,
        'mean_abs_error': mae,
        'probe': probe
    }


@torch.no_grad()
def encode_goal(goal_state: torch.Tensor, encoder: Encoder, room_size: int = 8) -> torch.Tensor:
    """Encode goal state to latent representation."""
    screen = torch.zeros([1, room_size, room_size])
    x, y = goal_state.to(torch.int32)
    screen[0][y][x] = 1
    image = screen.unsqueeze(0)
    embedding = encoder(image)
    return embedding.squeeze(0)


def latent_cost(latents: torch.Tensor, goal_embedding: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance to goal in latent space."""
    return ((latents - goal_embedding) ** 2).sum(dim=-1)


def sample_action_sequences(n_sequences: int, horizon: int, n_actions: int) -> torch.Tensor:
    """Sample random action sequences for planning."""
    return torch.randint(0, n_actions, (n_sequences, horizon))


def rollout_latent_dynamics(
    initial_embedding: torch.Tensor,
    actions: torch.Tensor,
    predictor: Predictor
) -> torch.Tensor:
    """Rollout latent dynamics given action sequences."""
    if initial_embedding.dim() == 1:
        initial_embedding = initial_embedding.unsqueeze(0)

    if actions.dim() == 1:
        actions = actions.unsqueeze(0).expand(initial_embedding.shape[0], -1)

    current = initial_embedding
    rollout = [current]

    for t in range(actions.shape[1]):
        current = predictor(current, actions[:, t])
        rollout.append(current)

    rollout = torch.stack(rollout, dim=0)
    if initial_embedding.shape[0] == 1 and rollout.shape[1] == 1:
        rollout = rollout.squeeze(1)
    return rollout


def score_action_sequences(
    start_embedding: torch.Tensor,
    action_sequences: torch.Tensor,
    goal_embedding: torch.Tensor,
    predictor: Predictor
) -> torch.Tensor:
    """Score action sequences by total cost to goal."""
    N = action_sequences.shape[0]
    initial_embeddings = start_embedding.unsqueeze(0).expand(N, -1)

    rollout = rollout_latent_dynamics(initial_embeddings, action_sequences, predictor)
    future_latents = rollout[1:]

    step_costs = latent_cost(future_latents, goal_embedding)
    total_costs = step_costs.sum(dim=0)
    return total_costs


def select_best_plan(action_sequences: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    """Select the action sequence with minimum cost."""
    best_idx = torch.argmin(costs)
    return action_sequences[best_idx]


def mpc_step(
    start_embedding: torch.Tensor,
    goal_embedding: torch.Tensor,
    predictor: Predictor,
    n_sequences: int,
    horizon: int,
    n_actions: int
) -> int:
    """Single MPC step: sample, score, select best action."""
    action_seq = sample_action_sequences(n_sequences, horizon, n_actions)
    costs = score_action_sequences(start_embedding, action_seq, goal_embedding, predictor)
    best_plan = select_best_plan(action_seq, costs)
    return best_plan[0].item()


@torch.no_grad()
def run_mpc_episode(
    encoder: Encoder,
    predictor: Predictor,
    goal_pos: torch.Tensor | tuple,
    room_size: int,
    agent_size: int,
    max_steps: int,
    n_sequences: int,
    horizon: int,
    n_actions: int
) -> dict:
    """Run a single MPC episode."""
    if isinstance(goal_pos, torch.Tensor):
        g_x, g_y = int(goal_pos[0].item()), int(goal_pos[1].item())
        goal_state = goal_pos.float()
    else:
        g_x, g_y = int(goal_pos[0]), int(goal_pos[1])
        goal_state = torch.tensor(goal_pos, dtype=torch.float32)

    goal_embedding = encode_goal(goal_state, encoder, room_size=room_size)
    state, obs = env_reset(room_size=room_size)

    trajectory = [(float(state[0].item()), float(state[1].item()))]
    success = False

    for step in range(max_steps):
        if int(state[0].item()) == g_x and int(state[1].item()) == g_y:
            success = True
            break

        start_embedding = encoder(obs.unsqueeze(0)).squeeze(0)
        action = mpc_step(
            start_embedding, goal_embedding, predictor,
            n_sequences, horizon, n_actions
        )

        state, obs = env_step(state, action, room_size=room_size)
        trajectory.append((float(state[0].item()), float(state[1].item())))

    success = bool(int(state[0].item()) == g_x and int(state[1].item()) == g_y)

    dx = float(state[0].item()) - float(g_x)
    dy = float(state[1].item()) - float(g_y)
    final_distance = float((dx ** 2 + dy ** 2) ** 0.5)

    return {
        'success': success,
        'steps': int(len(trajectory) - 1),
        'final_distance': final_distance,
        'trajectory': trajectory
    }


@torch.no_grad()
def evaluate_planner(
    encoder: Encoder,
    predictor: Predictor,
    room_size: int,
    agent_size: int,
    n_episodes: int,
    max_steps: int,
    n_sequences: int,
    horizon: int,
    n_actions: int
) -> dict:
    """Evaluate planner over multiple episodes."""
    total_success = 0.0
    total_steps = 0.0
    total_distance = 0.0

    for ep in range(n_episodes):
        goal_pos = torch.randint(0, room_size, (2,)).float()

        result = run_mpc_episode(
            encoder, predictor, goal_pos,
            room_size, agent_size, max_steps,
            n_sequences, horizon, n_actions
        )

        total_success += float(result['success'])
        total_steps += float(result['steps'])
        total_distance += float(result['final_distance'])

    return {
        'success_rate': total_success / n_episodes,
        'mean_steps': total_steps / n_episodes,
        'mean_final_distance': total_distance / n_episodes
    }


