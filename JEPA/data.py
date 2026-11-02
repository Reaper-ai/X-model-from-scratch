"""
Data Collection: Random transitions in the 2D room.
Steps 6-7 from the JEPA implementation.
Enhanced with multi-seed collection for better diversity and disk caching.
"""

import random
import torch
import numpy as np
import os
import pickle
import math

from env import env_reset, env_step


def optimal_transitions(room_size: int) -> int:
    """Calculate optimal number of transitions: nearest power of 2 > room_size^2."""
    cells = room_size * room_size
    return 2 ** math.ceil(math.log2(cells + 1))


def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    if num_transitions == 0:
        return {
            'observations': torch.empty((0, 1, room_size, room_size), dtype=torch.float32),
            'actions': torch.empty((0,), dtype=torch.long),
            'next_observations': torch.empty((0, 1, room_size, room_size), dtype=torch.float32),
            'states': torch.empty((0, 2), dtype=torch.float32),
            'next_states': torch.empty((0, 2), dtype=torch.float32)
        }

    observations = []
    actions = []
    next_observations = []
    states = []
    next_states = []

    state, obs = env_reset(room_size, seed)

    for _ in range(num_transitions):
        action = random.randint(0, 3)
        next_state, next_obs = env_step(state, action, room_size)

        observations.append(obs)
        actions.append(action)
        next_observations.append(next_obs)
        states.append(state)
        next_states.append(next_state)

        state = next_state
        obs = next_obs

    return {
        'observations': torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in observations]).view(num_transitions, 1, room_size, room_size),
        'actions': torch.tensor(actions, dtype=torch.long).view(num_transitions),
        'next_observations': torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in next_observations]).view(num_transitions, 1, room_size, room_size),
        'states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states]).view(num_transitions, 2),
        'next_states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in next_states]).view(num_transitions, 2)
    }


def collect_diverse_transitions(num_transitions: int, room_size: int = 8, num_seeds: int = 4) -> dict:
    """
    Collect transitions from multiple random seeds for better diversity.
    Ensures coverage across the entire room.
    """
    transitions_per_seed = num_transitions // num_seeds
    all_observations = []
    all_actions = []
    all_next_observations = []
    all_states = []
    all_next_states = []

    for seed_idx in range(num_seeds):
        seed = seed_idx * 1000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        state, obs = env_reset(room_size, seed)

        for _ in range(transitions_per_seed):
            action = random.randint(0, 3)
            next_state, next_obs = env_step(state, action, room_size)

            all_observations.append(obs)
            all_actions.append(action)
            all_next_observations.append(next_obs)
            all_states.append(state)
            all_next_states.append(next_state)

            state = next_state
            obs = next_obs

    total = len(all_observations)
    return {
        'observations': torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in all_observations]).view(total, 1, room_size, room_size),
        'actions': torch.tensor(all_actions, dtype=torch.long).view(total),
        'next_observations': torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in all_next_observations]).view(total, 1, room_size, room_size),
        'states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in all_states]).view(total, 2),
        'next_states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in all_next_states]).view(total, 2)
    }


def save_dataset(dataset: dict, filepath: str) -> None:
    """Save dataset to disk using pickle for efficient loading."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert tensors to numpy for smaller pickle size
    numpy_dataset = {k: v.cpu().numpy() for k, v in dataset.items()}
    with open(filepath, 'wb') as f:
        pickle.dump(numpy_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset(filepath: str, device: str = 'cpu') -> dict:
    """Load dataset from disk."""
    with open(filepath, 'rb') as f:
        numpy_dataset = pickle.load(f)
    return {k: torch.from_numpy(v).to(device) for k, v in numpy_dataset.items()}


def split_dataset(dataset: dict, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, device: str = 'cpu') -> tuple[dict, dict, dict]:
    """
    Split dataset into train/val/test sets.
    Returns (train_dataset, val_dataset, test_dataset) with tensors on specified device.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    N = dataset['observations'].shape[0]
    indices = torch.randperm(N)
    
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    def subset(idxs):
        return {k: v[idxs].to(device) for k, v in dataset.items()}
    
    return subset(train_idx), subset(val_idx), subset(test_idx)


def build_transition_dataset(
    num_transitions: int = None,
    room_size: int = 8,
    seed: int = 0,
    diverse: bool = True,
    cache_dir: str = "data_cache",
    device: str = 'cpu'
) -> dict:
    """
    Build transition dataset with optional disk caching.
    If num_transitions is None, uses optimal_transitions(room_size).
    If cached dataset exists with same params, loads from disk.
    """
    if num_transitions is None:
        num_transitions = optimal_transitions(room_size)
    
    cache_key = f"transitions_{num_transitions}_room{room_size}_diverse{diverse}.pkl"
    cache_path = os.path.join(cache_dir, cache_key)
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading dataset from cache: {cache_path}")
        return load_dataset(cache_path, device=device)
    
    print(f"Generating new dataset ({num_transitions} transitions)...")
    if diverse:
        dataset = collect_diverse_transitions(num_transitions, room_size)
    else:
        dataset = collect_random_transitions(num_transitions, room_size, seed)
    
    # Save to cache
    save_dataset(dataset, cache_path)
    print(f"Dataset cached to: {cache_path}")
    
    return {k: v.to(device) for k, v in dataset.items()}