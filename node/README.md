# Neural ODE (Damped Pendulum)

A Neural ODE built from scratch: a network that learns a damped pendulum's velocity field purely from noisy state observations, integrated by an adaptive ODE solver. A discrete-time MLP with the identical trunk serves as a fixed-step reference baseline, so the comparison isolates the *time parameterization* — a chained step map vs a learned continuous field — not capacity.

## The Comparison

```
Discrete MLP (fixed step map)              Neural ODE (continuous field)

x_k ──► x_k + MLP(x_k) ──► x_{k+1}         x(t) ──► dx/dt = MLP(x(t)) ──► ODE solver ──► x(t')
        chained k times for horizon k               integrates to ANY query time
```

- **Discrete MLP**: predicts the state *delta* per fixed Δt = 0.05 s; rollouts chain the map.
- **Neural ODE**: the same MLP is interpreted as a velocity field and integrated with `dopri5` (adjoint method for training).

## Architecture Overview

Both models: `2 → 64 → 64 → 64 → 64 → 2`, SiLU activations, Kaiming init, 12,802 parameters each, identical seed (42).

| Model | Form | Time handling |
|-------|------|---------------|
| **DiscreteMLP** | residual: `x_{k+1} = x_k + MLP(x_k)` (zero-init output layer) | fixed Δt steps only |
| **NeuralODE** | field: `dx/dt = MLP(x)` | any query time via solver |

The residual form is essential: a direct `x_{k+1} = MLP(x_k)` map is unstable when rolled out (compounding errors, ~50× worse loss). The zero-init output layer means training starts from "do nothing" and grows the correction.

## Data

- **System**: damped pendulum `dθ/dt = ω`, `dω/dt = −(g/L)·sin(θ) − b·ω` (g=9.81, L=1, b=0.1)
- **Trajectories**: 300 (+ mirrored for symmetry) with random ICs in the training box θ₀ ∈ ±π/2, ω₀ ∈ ±2 (extended experiment: ±π, ±4), dt = 0.05 s, slight measurement noise
- **Splits** (physics-based labels):
  - `train` — ICs in the training box, up to 3 periods
  - `test_A` — unseen ICs, same box, up to 3 periods (in-distribution)
  - `test_B` — same box, up to 8 periods (time extrapolation)
  - `test_C` — ICs at 1.5× the box (state extrapolation)
- **Two views of the same trajectories**:
  - regular grid → Discrete MLP
  - irregular (random 50% of time points kept, per-trajectory) → Neural ODE

## Training

| | DiscreteTrainer | TrajectoryTrainer (NODE) |
|---|---|---|
| Data | regular grid | irregular timestamps |
| Loss | one-step + multi-step rollout (0.5/0.5) | observed-state matching over windows (no derivative loss) |
| Curriculum | window length 15 → 50 over 50 epochs | same, with `time_scale` keeping window time span equal |
| Optimizer | Adam lr=1e-3, cosine schedule, wd=1e-5 | same |
| Epochs | 200 (early stop patience 20) | 200 (adjoint, `rk4` adjoint method) |

Both trainers sample training windows only from trajectories long enough to contain them.

## Evaluation

Two protocols (`run_evaluation` in `main.py`):

- **Protocol R (regular grid)** — both models queried on the dense regular grid (`evaluate_all_tests` on the regular loaders).
- **Protocol I (irregular queries)** — both models queried at the same irregular timestamps (`evaluate_queries_on_loader`); the MLP rolls on its own grid and is linearly interpolated, the ODE integrates natively.

> Note: never evaluate an irregular-trained model by integrating over a regular grid while comparing against irregularly-timed states — the time mismatch produces wildly wrong numbers. Use the query-based protocol instead.

Reference results (200 epochs, main config):

| Model | test_A | test_B | test_C |
|-------|--------|--------|--------|
| Discrete MLP | 3.9e-2 | 4.8e-2 | 0.91 |
| Neural ODE | **1.2e-2** | **1.2e-2** | 0.86 |

## Usage

```bash
# Full pipeline: data -> train both -> dual-protocol eval -> 5 figures (~2.2 h)
python main.py full-pipeline

# Quick smoke test
python main.py full-pipeline --epochs 2

# Print Protocol R / Protocol I comparison tables from metrics.json
python main.py summarize

# Regenerate figures only (loads saved models)
python main.py visualize

# Extended-range variant: train IC box θ0 ∈ ±π, ω0 ∈ ±4 (test at 1.5x)
python extended_experiment.py
```

CLI options: `--epochs`, `--irregular-frac`, `--irregular-strategy`, `--n-trajectories`, `--seed`, `--regenerate-data`, and more (see `main.py`).

## Output Files

### `node/data/`
- `pendulum_trajectories.npz` — main dataset (variable-length, 4-way labels)
- `discrete_mlp_model.pt` / `node_model.pt` — trained weights
- `metrics.json` — training histories + dual-protocol evaluation
- `extended_trajectories.npz`, `extended_metrics.json` — extended-range experiment
- `extended/` — extended-experiment checkpoints

### `docs/images/node/`
- `phase_portrait.png` — true vs each model's orbit (1×3 panels)
- `vector_field.png` — true field vs learned fields (MLP's implied field included)
- `timeseries_seen_unseen.png` — seen/unseen split with NODE training-sample markers
- `error_growth.png` — rollout error vs time (log scale)
- `ic_sensitivity.png` — trajectory MSE vs initial condition sweeps
- `extended_vector_field.png`, `extended_ic_sensitivity.png` — extended-range variants

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 64 | Hidden width (4 layers) |
| `dt` | 0.05 | Sampling time step |
| `n_trajectories` | 300 | Base trajectories (+ symmetric mirrors) |
| `k_periods_train` | 3 | Max training horizon in periods |
| `test_B_max_periods` | 8 | Long-horizon test length |
| `test_C_ic_range_factor` | 1.5 | Extrapolation IC range multiplier |
| `mlp_form` | `residual` | `x + MLP(x)` map (vs `direct`) |
| `one_step_weight` / `multi_step_weight` | 0.5 / 0.5 | Discrete loss mix |
| `irregular_frac` | 0.5 | Fraction of points kept for the NODE view |
| `use_derivative_loss` | False | Fair supervision (states only) |
| `integration_method` | `dopri5` | ODE solver |
| `use_adjoint` | True | Adjoint backprop through the solver |
| `window_size` / `window_size_start` | 50 / 15 | Rollout curriculum |
| `n_windows_per_batch` | 64 | Windows sampled per batch |
| `curriculum_epochs` | 50 | Ramp-up epochs for window length |
