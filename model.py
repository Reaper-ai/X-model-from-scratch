"""
JEPA World Model from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_env_state
import torch
import random
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
    if seed: set_seed(seed)

    v = torch.randint(low=0, high=room_size, size=(2,))
    
    return v.to(torch.float32)

# Step 2 - apply_action
def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    
    get_dir = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    state = state + torch.tensor(get_dir[action])
    return torch.clamp(state, min=0, max=room_size - 1)

# Step 3 - render_observation
def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    screen = torch.zeros([1,room_size,room_size])
    
    x,y = state.to(torch.int32)
    screen[0][y][x] = 1
    return screen

# Step 4 - env_reset
def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    new_state = init_env_state(room_size, seed)
    screen = render_observation(new_state, room_size)
    return new_state, screen

# Step 5 - env_step
def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    new_state =  apply_action(state, action, room_size)
    new_obs = render_observation(new_state,room_size)
    return new_state, new_obs

# Step 6 - collect_random_transitions
import torch
import random

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

# Step 7 - build_transition_dataset
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    return  collect_random_transitions(num_transitions, room_size, seed)

# Step 8 - init_encoder_params
import torch

def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    
    params = {}
    
    params['conv1_w'] = torch.randn(16, obs_channels, 3, 3) * 0.1
    params['conv1_b'] = torch.zeros(16)
    
    params['conv2_w'] = torch.randn(32, 16, 3, 3) * 0.1
    params['conv2_b'] = torch.zeros(32)

    conv2_out_size = room_size // 2
    flattened_size = 32 * conv2_out_size * conv2_out_size
    
    params['fc_w'] = torch.randn(latent_dim, flattened_size) * 0.1
    params['fc_b'] = torch.zeros(latent_dim)
    
    for k in params:
        params[k].requires_grad = True
        
    return params

# Step 9 - encoder_forward
import torch
import torch.nn.functional as F

def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    c1 = F.conv2d(input=obs, weight=encoder_params['conv1_w'], stride=(1,1), padding=(1,1), bias=encoder_params['conv1_b'])
    f1 = F.relu(c1)
    
    c2 = F.conv2d(input=f1, weight=encoder_params['conv2_w'], stride=(2,2), padding=(1,1), bias=encoder_params['conv2_b'])
    f2 = F.relu(c2)
    f2 = f2.flatten(1)
    
    fc = F.linear(input=f2, weight=encoder_params['fc_w'], bias=encoder_params['fc_b'])
    
    return fc

# Step 10 - init_target_encoder
def init_target_encoder(encoder_params: dict) -> dict:
    # TODO: Create the EMA target encoder by deep-copying the online encoder params.
    target = {}
    for key in encoder_params:
        value = encoder_params[key]
        target[key] = value.detach().clone()
        target[key].requires_grad = False
    return target

# Step 11 - ema_update
def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    # TODO: Refresh target encoder params via EMA of the online encoder.
    
    t = {}
    for key in encoder_params:
        t[key] = tau*target_params[key]
        t[key] += (1-tau)*encoder_params[key]
    
    return t

# Step 12 - encode_batch
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params
def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    t = lambda x: (torch.randn(x, requires_grad=True) * 0.02).detach().requires_grad_(True)
    
    predictor_params = {}
    predictor_params['action_embed_w'] =  t([action_dim,latent_dim])
    predictor_params['fc1_w'] = t([hidden_dim,2*latent_dim])
    predictor_params['fc2_w'] = t([latent_dim, hidden_dim])
    predictor_params['fc1_b'] = torch.zeros(hidden_dim, requires_grad=True)
    predictor_params['fc2_b'] = torch.zeros(latent_dim, requires_grad=True)

    return predictor_params

# Step 14 - embed_action
def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_embed_w = predictor_params['action_embed_w']
    action_vec = action_embed_w[actions]
    return action_vec

# Step 15 - predictor_forward
def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_embeds =  embed_action(actions, predictor_params)
    pred_vec = torch.concat([embeddings, action_embeds], dim=-1)
    fc1 = F.linear(input=pred_vec, weight=predictor_params['fc1_w'], bias=predictor_params['fc1_b'])
    r1 = F.relu(fc1)
    fc2 = F.linear(input=r1, weight=predictor_params['fc2_w'], bias=predictor_params['fc2_b'])
    
    return fc2

# Step 16 - predict_next_embedding
def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    next_z = predictor_forward(embeddings, actions, predictor_params)
    return next_z

# Step 17 - prediction_loss
def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = predicted - target
    error2 = torch.square(error)
    return torch.mean(error2)

# Step 18 - variance_loss
def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    var = torch.var(embeddings, dim=0)
    std = torch.sqrt(var + eps) 
    res = F.relu(gamma - std)
    loss = torch.mean(res)
    return loss

# Step 19 - covariance_loss
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    mean = embeddings.mean(dim=0)
    z = embeddings - mean
    B = embeddings.shape[0]
    cov = torch.mm(z.t(),z) / (B-1)
    D = embeddings.shape[1]
    loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    return loss

# Step 20 - vicreg_regularizer
def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    var = variance_loss(embeddings, gamma)
    cov = covariance_loss(embeddings)
    vic = var_weight * var + cov_weight * cov

    return vic

# Step 21 - jepa_loss
def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    pred_loss = prediction_loss(predicted, target)
    reg = vicreg_regularizer(online_embeddings, var_weight, cov_weight)
    return pred_weight * pred_loss + reg

# Step 22 - collapse_metric
def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    stds = torch.std(embeddings, dim=0)
    return torch.mean(stds)

# Step 23 - jepa_training_step
def jepa_training_step(
    batch: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    lr: float = 1e-3,
    tau: float = 0.99
) -> tuple[dict, dict, dict, float, float]:

    obs = batch["observations"]
    actions = batch["actions"]
    next_obs = batch["next_observations"]

    # 1. Enable gradients and clear previous gradients
    for params in (encoder_params, predictor_params):
        for param in params.values():
            param.requires_grad_(True)
            param.grad = None

    # 2. Online encoder
    embeddings = encode_batch(obs, encoder_params)

    # 3. Target encoder (no gradient)
    with torch.no_grad():
        target = encode_batch(next_obs, target_params)

    # 4. Predictor
    predicted = predict_next_embedding(
        embeddings, actions, predictor_params
    )

    # 5. JEPA loss
    loss = jepa_loss(predicted, target, embeddings)

    # 6. Backpropagation
    loss.backward(retain_graph=True)

    # 7. Update online encoder
    updated_encoder = {}

    for name, param in encoder_params.items():
        if param.grad is None:
            updated_encoder[name] = param.detach().requires_grad_(True)
        else:
            updated_encoder[name] = (
                param - lr * param.grad
            ).detach().requires_grad_(True)

    # 8. Update predictor
    updated_predictor = {}

    for name, param in predictor_params.items():
        if param.grad is None:
            updated_predictor[name] = param.detach().requires_grad_(True)
        else:
            updated_predictor[name] = (
                param - lr * param.grad
            ).detach().requires_grad_(True)

    # 9. EMA target update
    updated_target = ema_update(
        target_params,
        updated_encoder,
        tau=tau
    )

    # 10. Collapse metric
    collapse = collapse_metric(embeddings.detach())

    return (
        updated_encoder,
        updated_target,
        updated_predictor,
        loss.item(),
        collapse.item()
    )

# Step 24 - train_jepa
def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    torch.manual_seed(seed)
    N = dataset["observations"].shape[0]
 
    enc = encoder_params
    tgt = target_params
    pred = predictor_params
    history = []
    for i in range(num_steps):
        idx = torch.randint(0,N, (batch_size,))
        batch = {"observations": dataset["observations"][idx],
            "actions" : dataset["actions"][idx],
            "next_observations" : dataset["next_observations"][idx]}
        enc, tgt, pred, loss, collapse = jepa_training_step(batch, enc, tgt, pred, lr, tau)
        history.append({
        "loss": float(loss),
        "collapse": float(collapse)
    })
    
    return enc, tgt, pred, history

# Step 25 - rollout_latent_dynamics
def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    flag =  initial_embedding.dim() == 1
    
    if flag:
        initial_embedding = initial_embedding.unsqueeze(0)    

    if actions.dim() == 1:
        actions = actions.unsqueeze(0).expand(
            initial_embedding.shape[0], -1
        )
    current = initial_embedding
    rollout = [current]
    for t in range(actions.shape[1]):
        current = predict_next_embedding(current, actions[: , t], predictor_params)
        rollout.append(current)
    
    rollout =  torch.stack(rollout, dim=0)
    if flag: rollout = rollout.squeeze(1)
    return rollout

# Step 26 - multi_step_prediction_error
def multi_step_prediction_error(
    dataset: dict,
    encoder_params: dict,
    target_params: dict,
    predictor_params: dict,
    horizon: int = 5,
    num_samples: int = 32
) -> float:

    N = len(dataset["actions"])
    num_samples = min(num_samples, N - horizon + 1)

    indices = torch.arange(num_samples)

    obs = dataset["observations"][indices]
    actions = dataset["actions"][indices[:, None] + torch.arange(horizon)]
    future_obs = dataset["next_observations"][
        indices[:, None] + torch.arange(horizon)
    ]

    embeddings = encode_batch(obs, encoder_params)

    rollout = rollout_latent_dynamics(
        embeddings, actions, predictor_params
    )[1:]

    B, H, C, W, W2 = future_obs.shape
    with torch.no_grad():
        target = encode_batch(
            future_obs.reshape(B * H, C, W, W2),
            target_params
        )

    target = target.reshape(B, H, -1).transpose(0, 1)

    error = (rollout - target).pow(2).sum(dim=-1).mean()

    return error.item()

# Step 27 - init_linear_probe
def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    probe = {}
    probe["w"] = torch.randn(state_dim, latent_dim) * 0.01
    probe["b"] = torch.zeros(state_dim)

    return probe

# Step 28 - train_linear_probe
def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    embeddings = embeddings.detach()

    w = probe_params['w'].clone().detach().requires_grad_(True)
    b = probe_params['b'].clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        pred = embeddings @ w + b 
        loss = torch.mean((pred -states ) ** 2)
        loss.backward()
        with torch.no_grad():
            w -= lr*w.grad
            b -= lr*b.grad
        w.grad = None
        b.grad = None
    
    probe_params = {'w': w.detach().clone(), 'b': b.detach().clone()}
    return probe_params

# Step 29 - probe_state_recovery
def train_linear_probe_correct(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    embeddings = embeddings.detach()

    w = probe_params['w'].clone().detach().requires_grad_(True)
    b = probe_params['b'].clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        pred = embeddings @ w.T + b 
        loss = torch.mean((pred -states ) ** 2)
        loss.backward()
        with torch.no_grad():
            w -= lr*w.grad
            b -= lr*b.grad
        w.grad = None
        b.grad = None
    
    probe_params = {'w': w.detach().clone(), 'b': b.detach().clone()}
    return probe_params

def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    obs = dataset["observations"]
    states = dataset["states"]

    embedding = encode_batch(obs, encoder_params)
    D = embedding.shape[1]
    if probe_params is None:
        probe_params = init_linear_probe(latent_dim=D, state_dim=states.shape[-1], seed=0)
    
    probe_params = train_linear_probe_correct(embedding, states, probe_params, num_probe_steps)
    
    predicted_states = embedding @ probe_params["w"].T + probe_params["b"]
    mse = F.mse_loss(predicted_states, states).item()
    mae = F.l1_loss(predicted_states, states).item()
    
    results = {}
    results["mse"] = mse
    results["mean_abs_error"] = mae 
    results["probe_params"] = probe_params

    return results

# Step 30 - encode_goal
def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    image = render_observation(goal_state, room_size)
    image = image.unsqueeze(0)
    embedding = encoder_forward(image, encoder_params)
    embedding = embedding.squeeze(0)
    return embedding

# Step 31 - latent_cost
def latent_cost(latents, goal_embedding):
    diff = torch.sum((latents - goal_embedding)**2, dim=-1)
    return diff

# Step 32 - sample_action_sequences
def sample_action_sequences(n_sequences, horizon, n_actions):
    return torch.randint(0, n_actions, (n_sequences, horizon))

# Step 33 - score_action_sequences
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

# Step 34 - select_best_plan
def select_best_plan(action_sequences, costs):
    minmin = torch.argmin(costs)
    action = action_sequences[minmin]
    return action

# Step 35 - mpc_step
def mpc_step(start_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions):
    action_seq = sample_action_sequences(n_sequences, horizon, n_actions)
    costs = score_action_sequences(start_embedding, action_seq, goal_embedding, predictor_params)
    best_plan = select_best_plan(action_seq, costs)
    return best_plan[0].item()

# Step 36 - run_mpc_episode
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

        start_embedding = encode_batch(obs.unsqueeze(0), encoder_params).squeeze(0)
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

# Step 37 - evaluate_planner
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

# Step 38 - jepa_world_model_experiment
def jepa_world_model_experiment(room_size, agent_size, embed_dim, n_train_transitions, n_epochs, batch_size, n_probe_samples, n_eval_episodes, max_steps, n_sequences, horizon):
    
    torch.manual_seed(0)

    dataset = build_transition_dataset(n_train_transitions,room_size=room_size, seed=0)
    encoder_params = init_encoder_params(room_size=room_size, latent_dim=embed_dim, seed=0)

    target_params = init_target_encoder(encoder_params)

    predictor_params = init_predictor_params(latent_dim=embed_dim, action_dim=4, hidden_dim=64, seed=0)

    encoder_params, target_params, predictor_params, history = train_jepa(
        dataset=dataset,
        encoder_params=encoder_params,
        target_params=target_params,
        predictor_params=predictor_params,
        num_steps=n_epochs,
        batch_size=batch_size,
        lr=1e-3, tau=0.99, seed=0
    )

    train_losses = [ float(entry["loss"]) for entry in history]
    collapse_metrics = [ float(entry["collapse"]) for entry in history]

    probe_dataset = { key: value[:n_probe_samples] for key, value in dataset.items()}

    probe_results = probe_state_recovery(
        dataset=probe_dataset,
        encoder_params=encoder_params,
        probe_params=None,
        num_probe_steps=100
    )

    # R² = 1 - MSE / (Var(states) + eps)
    states = probe_dataset["states"]

    state_variance = torch.mean(
        (states - states.mean(dim=0, keepdim=True)) ** 2
    )

    probe_mse = torch.as_tensor(
        probe_results["mse"],
        dtype=state_variance.dtype,
        device=state_variance.device
    )

    probe_r2 = 1.0 - (
        probe_mse / (state_variance + 1e-8)
    )

    planner_results = evaluate_planner(
        encoder_params=encoder_params,
        predictor_params=predictor_params,
        room_size=room_size,
        agent_size=agent_size,
        n_episodes=n_eval_episodes,
        max_steps=max_steps,
        n_sequences=n_sequences,
        horizon=horizon,
        n_actions=4
    )

    return {
        "train_losses": train_losses,
        "collapse_metrics": collapse_metrics,
        "probe_r2": float(probe_r2.item()),
        "success_rate": float(planner_results["success_rate"]),
        "mean_steps": float(planner_results["mean_steps"])
    }

