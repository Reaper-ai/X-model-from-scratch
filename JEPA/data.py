"""
Data Collection: Random transitions in the 2D room.
Steps 6-7 from the JEPA implementation.
"""

import random
import torch

from env import env_reset, env_step


def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    random.seed(seed)

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


def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    return collect_random_transitions(num_transitions, room_size, seed)