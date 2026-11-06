# Neural ODE from Scratch: Lorenz Attractor

**Scaffolding only** — implement each function step by step.

## Overview

This project implements Neural Ordinary Differential Equations (Neural ODEs) for learning the Lorenz attractor dynamics, with comparison against ResNet MLP (derivative matching) and classical numerical integrators.

## System: Lorenz Attractor

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

σ = 10, ρ = 28, β = 8/3
```

## Methods to Implement

| Method | Training Objective | Integration |
|--------|-------------------|-------------|
| **ResNet MLP** | `MSE(f_θ(x), dx/dt_true)` | Fixed-step RK4 |
| **Neural ODE** | `MSE(∫f_θ dt, x_true)` | Adaptive DOPRI5 (adjoint) |
| **Classical RK45** | None (ground truth) | SciPy RK45 |
| **Classical DOP853** | None (ground truth) | SciPy DOP853 |
| **Classical Euler** | None (ground truth) | Fixed-step Euler |
| **Classical RK4** | None (ground truth) | Fixed-step RK4 |

## Implementation Order

Follow the step numbers in each file:

### 1. `node/data_env.py` — Data Generation
- [ ] `lorenz_system()` — Ground truth ODE
- [ ] `generate_trajectories()` — Simulate with scipy.integrate.solve_ivp
- [ ] `sample_irregular_timepoints()` — Random/Poisson/log sampling
- [ ] `LorenzDataset` — PyTorch Dataset
- [ ] `create_dataloaders()` — Train/val/test splits
- [ ] `save_trajectories()` / `load_trajectories()` — .npz I/O

### 2. `node/model.py` — Model Architectures
- [ ] `ResidualBlock` — x + Linear(ReLU(Linear(x)))
- [ ] `ResNetMLP` — Time-invariant f_θ(x) → dx/dt
- [ ] `NeuralODE` — Time-conditioned f_θ(t, x) → dx/dt
- [ ] `ClassicalIntegrator` — Wrapper for scipy/fixed-step integrators
- [ ] `vector_field_loss()` / `trajectory_loss()` — MSE losses
- [ ] `count_parameters()` — Parameter counting
- [ ] `integrate_neural_ode()` — torchdiffeq integration
- [ ] `integrate_resnet()` — Fixed-step RK4 integration

### 3. `node/training_eval.py` — Training & Evaluation
- [ ] `EarlyStopping` — Callback
- [ ] `ResNetTrainer` — Derivative matching training
- [ ] `NeuralODETrainer` — Trajectory matching with adjoint
- [ ] `ClassicalEvaluator` — Scipy integrator evaluation
- [ ] `evaluate_trajectory()` — Single trajectory metrics
- [ ] `compute_lyapunov_exponent()` — Chaos divergence metric
- [ ] `evaluate_on_test_set()` — Full test set evaluation

### 4. `node/main.py` — Orchestration
- [ ] `setup_device()` — CUDA/MPS/CPU
- [ ] `setup_models()` — Instantiate all models
- [ ] `setup_optimizers()` — Adam + ReduceLROnPlateau
- [ ] `run_data_generation()` — Generate/load data
- [ ] `run_resnet_training()` — Train ResNet
- [ ] `run_node_training()` — Train Neural ODE
- [ ] `run_evaluation()` — Evaluate all methods
- [ ] `run_visualization()` — Generate figures
- [ ] `run_full_pipeline()` — End-to-end
- [ ] `main()` — CLI with argparse

### 5. `node/visualize.py` — Visualization
- [ ] `plot_lorenz_ground_truth()` — 3D + time series
- [ ] `plot_vector_field_comparison()` — Quiver plots
- [ ] `plot_3d_trajectories_comparison()` — 3D overlay
- [ ] `plot_trajectory_timeseries()` — x(t), y(t), z(t)
- [ ] `plot_error_growth()` — MSE + Lyapunov bars
- [ ] `plot_loss_curves()` — Training curves
- [ ] `plot_irregular_sampling_demo()` — Time sampling viz
- [ ] `plot_method_comparison_table()` — Summary table
- [ ] `save_all_figures()` — Orchestrates all plots

## Quick Start

```bash
# Install deps
uv pip install torchdiffeq scipy matplotlib seaborn tqdm

# Run full pipeline (after implementing)
python -m node.main full-pipeline

# Or individual steps
python -m node.main generate-data
python -m node.main train-resnet
python -m node.main train-node
python -m node.main evaluate
python -m node.main visualize
```

## Outputs

- **Models**: `node/data/resnet_model.pt`, `node/data/node_model.pt`
- **Metrics**: `node/data/metrics.json`
- **Figures**: `docs/images/node/`
  - `lorenz_ground_truth.png`
  - `vector_field_comparison.png`
  - `trajectories_3d_comparison.png`
  - `trajectory_timeseries.png`
  - `error_growth_log.png`
  - `loss_curves.png`
  - `irregular_sampling_demo.png`
  - `method_comparison_table.png`

## Key Concepts to Implement

1. **Derivative matching** (ResNet): Direct MSE on `f_θ(x)` vs true `dx/dt`
2. **Trajectory matching** (Neural ODE): Integrate `f_θ` then MSE on states
3. **Adjoint method**: `torchdiffeq.odeint_adjoint` for memory-efficient gradients
4. **Irregular time sampling**: Neural ODE handles arbitrary `t`; ResNet needs regular grid
5. **Lyapunov exponent**: Quantifies chaos preservation (true λ ≈ 0.9)

## References

- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
- torchdiffeq: https://github.com/rtqichen/torchdiffeq
- Lorenz, "Deterministic Nonperiodic Flow" (1963)