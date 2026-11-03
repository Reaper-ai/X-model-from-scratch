# JEPA World Model

Joint Embedding Predictive Architecture (JEPA) world model for a 2D grid navigation task. The model learns a latent dynamics model from random exploration and uses it for model-predictive control (MPC) planning.

## Architecture Overview

```
Observation o_t ──► Online Encoder ──► z_t ──┐
                                              ├──► Predictor ──► z_{t+1} ──► Linear Probe ──► (x, y)
Action a_t ───────────────────────────────────┘
                                              │
Next Obs o_{t+1} ──► Target Encoder (EMA) ────┘
                         (τ = 0.996)
```

### Data Flow
1. **Online Encoder**: CNN + FC encodes 6×6 grid observation → 16-dim latent `z_t`
2. **Target Encoder**: EMA copy of online encoder (τ=0.996), encodes `o_{t+1}` → `z_{t+1}^target`
3. **Predictor**: Takes `z_t` + action embedding `a_t` → predicts `z_{t+1}`
4. **Linear Probe**: Trained on predictor's output `z_{t+1}` to decode position `(x, y)`

## Model Components

| Component | Architecture | Parameters |
|-----------|--------------|------------|
| **Online Encoder** | Conv2d(1→16, 3×3, s=1) → ReLU → Conv2d(16→32, 3×3, s=2) → ReLU → Flatten → Linear(288→16) | ~15K |
| **Target Encoder** | Same as online (frozen, EMA updated) | ~15K |
| **Predictor** | Embedding(4→16) → Concat(z,a) → Linear(32→64) → ReLU → Linear(64→16) | ~6K |
| **Linear Probe** | Linear(16→2) | ~34 |

**Total trainable**: ~21K parameters

## Loss Function

**JEPA Loss** = Prediction Loss + VICReg Regularization

```
L = ||z_{t+1}^pred - z_{t+1}^target||²  +  λ_var · VarLoss  +  λ_cov · CovLoss
```

- **Prediction Loss**: MSE between predicted and target latents
- **Variance Loss**: `ReLU(γ - std(z))` with γ=1.0 (prevents collapse)
- **Covariance Loss**: Off-diagonal covariance penalty (decorrelates dimensions)
- **Weights**: λ_var=1.0, λ_cov=0.04

## Training Procedure

1. **Data Collection**: 512 random transitions in 6×6 grid (4 actions: up/down/left/right)
2. **Split**: 70% train / 15% val / 15% test
3. **Optimization**: Adam, lr=3×10⁻⁴, batch=32
4. **EMA Update**: Target encoder updated with τ=0.996 each step
5. **Early Stopping**: Patience=10 val evaluations (every 50 steps), min_delta=1e-4
6. **Probe Training**: Linear probe on predictor's single-step predictions (300 steps, lr=1e-2)

## Evaluation

- **Linear Probe R²**: How well predicted latents encode agent position
- **Multi-step Prediction Error**: Latent rollout vs target encoder over 5 steps
- **MPC Planner**: Random shooting (64 sequences, horizon=5) in latent space
- **Success Rate**: % episodes reaching goal within 20 steps

## Usage

```bash
# Train model (outputs to jepa/data/)
python -m jepa.main

# Generate visualizations (outputs to docs/images/JEPA/)
python -m jepa.visualize
```

## Output Files

### `jepa/data/`
- `data.pt` — Full dataset (train/val/test splits)
- `checkpoint.pt` — Model weights
- `history.json` — Train/val loss & collapse per step
- `results.json` — Final metrics
- `test_predictions.json` — 5 successful + 5 unsuccessful trajectories

### `docs/images/JEPA/`
- `architecture.png` — Architecture diagram with L-shaped arrows & LaTeX math
- `training_animation.gif` — Animated loss & collapse curves
- `trajectory_grid.png` — 2×5 grid (green=success, red=fail with X markers)

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `room_size` | 6 | Grid dimension |
| `latent_dim` | 16 | Latent embedding size |
| `n_train_transitions` | 512 | Training samples |
| `batch_size` | 32 | Mini-batch size |
| `lr` | 3×10⁻⁴ | Learning rate |
| `tau` | 0.996 | EMA decay for target encoder |
| `num_steps` | 1500 | Max training steps |
| `early_stopping_patience` | 10 | Val evaluations without improvement |
| `n_sequences` | 64 | MPC random sequences |
| `horizon` | 5 | MPC planning horizon |