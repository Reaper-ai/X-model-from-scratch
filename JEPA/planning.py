"""
MPC Planning in Latent Space.
Steps 30-37 from the JEPA implementation.
"""

import torch

from env import render_observation, env_reset, env_step
from encoder import encode_batch
from predictor import predict_next_embedding
from evaluation import rollout_latent_dynamics


def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    image = render_observation(goal_state, room_size)
    image = image.unsqueeze(0).to(next(iter(encoder_params.values())).device)
    embedding = encode_batch(image, encoder_params)
    embedding = embedding.squeeze(0)
    return embedding


def latent_cost(latents, goal_embedding):
    diff = torch.sum((latents - goal_embedding) ** 2, dim=-1)
    return diff


def sample_action_sequences(n_sequences, horizon, n_actions, device='cpu'):
    return torch.randint(0, n_actions, (n_sequences, horizon), device=device)


def score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params):
    N = action_sequences.shape[0]

    initial_embeddings = start_embedding.unsqueeze(0).expand(N, -1)

    rollout = rollout_latent_dynamics(
        initial_embeddings,
        action_sequences,
        predictor_params
    )

    future_latents = rollout[1:]

    step_costs = latent_cost(future_latents, goal_embedding)

    total_costs = step_costs.sum(dim=0)

    return total_costs


def select_best_plan(action_sequences, costs):
    minmin = torch.argmin(costs)
    action = action_sequences[minmin]
    return action


def mpc_step(start_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions):
    device = start_embedding.device
    action_seq = sample_action_sequences(n_sequences, horizon, n_actions, device=device)
    costs = score_action_sequences(start_embedding, action_seq, goal_embedding, predictor_params)
    best_plan = select_best_plan(action_seq, costs)
    return best_plan[0].item()


def run_mpc_episode(encoder_params, predictor_params, goal_pos, room_size, agent_size, max_steps, n_sequences, horizon, n_actions):
    if isinstance(goal_pos, torch.Tensor):
        g_x, g_y = int(goal_pos[0].item()), int(goal_pos[1].item())
        goal_state = goal_pos.float()
    else:
        g_x, g_y = int(goal_pos[0]), int(goal_pos[1])
        goal_state = torch.tensor(goal_pos, dtype=torch.float32)

    goal_embedding = encode_goal(goal_state, encoder_params, room_size=room_size)
    state, obs = env_reset(room_size=room_size)

    trajectory = [(float(state[0].item()), float(state[1].item()))]
    success = False

    for step in range(max_steps):
        if int(state[0].item()) == g_x and int(state[1].item()) == g_y:
            success = True
            break

        start_embedding = encode_batch(obs.unsqueeze(0).to(next(iter(encoder_params.values())).device), encoder_params).squeeze(0)
        action = mpc_step(start_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions)

        state, obs = env_step(state, action, room_size=room_size)
        trajectory.append((float(state[0].item()), float(state[1].item())))

    success = bool(int(state[0].item()) == g_x and int(state[1].item()) == g_y)

    dx = float(state[0].item()) - float(g_x)
    dy = float(state[1].item()) - float(g_y)
    final_distance = float((dx ** 2 + dy ** 2) ** 0.5)

    return {
        "success": success,
        "steps": int(len(trajectory) - 1),
        "final_distance": final_distance,
        "trajectory": trajectory
    }


def evaluate_planner(encoder_params, predictor_params, room_size, agent_size, n_episodes, max_steps, n_sequences, horizon, n_actions):
    total_success = 0.0
    total_steps = 0.0
    total_distance = 0.0

    for ep in range(n_episodes):
        goal_pos = torch.randint(0, room_size, (2,)).float()

        result = run_mpc_episode(
            encoder_params,
            predictor_params,
            goal_pos,
            room_size,
            agent_size,
            max_steps,
            n_sequences,
            horizon,
            n_actions
        )

        total_success += float(result['success'])
        total_steps += float(result['steps'])
        total_distance += float(result['final_distance'])

    return {
        "success_rate": total_success / n_episodes,
        "mean_steps": total_steps / n_episodes,
        "mean_final_distance": total_distance / n_episodes
    }