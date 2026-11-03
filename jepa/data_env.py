"""
JEPA World Model - Data Generation and Environment
"""

import os
import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    if seed:
        set_seed(seed)
    v = torch.randint(low=0, high=room_size, size=(2,))
    return v.to(torch.float32)


def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    get_dir = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    state = state + torch.tensor(get_dir[action])
    return torch.clamp(state, min=0, max=room_size - 1)


def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    screen = torch.zeros([1, room_size, room_size])
    x, y = state.to(torch.int32)
    screen[0][y][x] = 1
    return screen


def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    new_state = init_env_state(room_size, seed)
    screen = render_observation(new_state, room_size)
    return new_state, screen


def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    new_state = apply_action(state, action, room_size)
    new_obs = render_observation(new_state, room_size)
    return new_state, new_obs


def collect_random_transitions(
    num_transitions: int,
    room_size: int = 8,
    seed: int = 0,
    chunk_size: int = 10000
) -> dict:
    """
    Collect transitions in chunks to avoid OOM.
    """
    random.seed(seed)

    if num_transitions == 0:
        return {
            'observations': torch.empty((0, 1, room_size, room_size), dtype=torch.float32),
            'actions': torch.empty((0,), dtype=torch.long),
            'next_observations': torch.empty((0, 1, room_size, room_size), dtype=torch.float32),
            'states': torch.empty((0, 2), dtype=torch.float32),
            'next_states': torch.empty((0, 2), dtype=torch.float32)
        }

    all_observations = []
    all_actions = []
    all_next_observations = []
    all_states = []
    all_next_states = []

    state, obs = env_reset(room_size, seed)

    for i in range(num_transitions):
        action = random.randint(0, 3)
        next_state, next_obs = env_step(state, action, room_size)

        all_observations.append(obs)
        all_actions.append(action)
        all_next_observations.append(next_obs)
        all_states.append(state)
        all_next_states.append(next_state)

        state = next_state
        obs = next_obs

        if (i + 1) % chunk_size == 0:
            yield {
                'observations': torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in all_observations]),
                'actions': torch.tensor(all_actions, dtype=torch.long),
                'next_observations': torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in all_next_observations]),
                'states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in all_states]),
                'next_states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in all_next_states])
            }
            all_observations.clear()
            all_actions.clear()
            all_next_observations.clear()
            all_states.clear()
            all_next_states.clear()

    if all_observations:
        yield {
            'observations': torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in all_observations]),
            'actions': torch.tensor(all_actions, dtype=torch.long),
            'next_observations': torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in all_next_observations]),
            'states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in all_states]),
            'next_states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in all_next_states])
        }


def build_transition_dataset(
    num_transitions: int = 512,
    room_size: int = 8,
    seed: int = 0,
    save_path: str | None = None,
    chunk_size: int = 10000
) -> dict:
    """
    Build transition dataset with optional disk saving to avoid OOM.
    """
    chunks = list(collect_random_transitions(num_transitions, room_size, seed, chunk_size))

    if len(chunks) == 1:
        dataset = chunks[0]
    else:
        dataset = {
            'observations': torch.cat([c['observations'] for c in chunks], dim=0),
            'actions': torch.cat([c['actions'] for c in chunks], dim=0),
            'next_observations': torch.cat([c['next_observations'] for c in chunks], dim=0),
            'states': torch.cat([c['states'] for c in chunks], dim=0),
            'next_states': torch.cat([c['next_states'] for c in chunks], dim=0)
        }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, save_path)
        print(f"Dataset saved to {save_path}")

    return dataset


def load_dataset(path: str) -> dict:
    """Load dataset from disk."""
    return torch.load(path, map_location='cpu')


def split_dataset(dataset: dict, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> tuple[dict, dict, dict]:
    """
    Split dataset into train, validation, and test sets.
    """
    torch.manual_seed(seed)
    N = dataset['observations'].shape[0]
    indices = torch.randperm(N)

    train_size = int(N * train_ratio)
    val_size = int(N * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    def make_split(idx):
        return {
            'observations': dataset['observations'][idx],
            'actions': dataset['actions'][idx],
            'next_observations': dataset['next_observations'][idx],
            'states': dataset['states'][idx],
            'next_states': dataset['next_states'][idx]
        }

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


def random_policy_baseline(room_size, n_episodes, max_steps, seed=123):
    rng = np.random.default_rng(seed)
    successes = 0
    for ep in range(n_episodes):
        goal = rng.integers(0, room_size, size=2)
        state, obs = env_reset(room_size=room_size, seed=1000 + ep)
        for _ in range(max_steps):
            if int(state[0].item()) == int(goal[0]) and int(state[1].item()) == int(goal[1]):
                break
            state, obs = env_step(state, int(rng.integers(0, 4)), room_size=room_size)
        if int(state[0].item()) == int(goal[0]) and int(state[1].item()) == int(goal[1]):
            successes += 1
    return successes / n_episodes