"""
Environment: 2D Room with discrete actions and pixel observations.
Steps 1-5 from the JEPA implementation.
"""

import random
import torch
import numpy as np


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