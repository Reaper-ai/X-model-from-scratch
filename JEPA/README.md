# JEPA World Model from Scratch

Build an action-conditioned Joint Embedding Predictive Architecture (JEPA) world model in pure PyTorch. An encoder and EMA target learn collapse-resistant latents of a 2D room via VICReg, a predictor rolls dynamics in embedding space, and acting is random-shooting MPC toward goal embeddings. No decoder, no pixels, no learned policy.

## Configuration (current run)

- **Room**: 6×6, **transitions**: 512, **latent dim**: 16, **hidden**: 32
- **Epochs**: 100 (12 steps/epoch = 1200 steps), **no early stopping**
- **Batch**: 32, **lr**: 0.1, **EMA tau**: 0.99
- **Horizon curriculum**: 5 → 10 → 15 → 20 every 30 epochs
- **VICReg target**: collapse metric (mean latent std) → 1.0
- **Results**: Probe MAE ≈ 0.37, MPC success 60%/90%/70% (h=5/10/20)

## File Structure

| File | Steps | Description |
|------|-------|-------------|
| `env.py` | 1–5 | 2D room: state init, actions, pixel rendering, reset/step API |
| `data.py` | 6–7 | Random transition dataset collection |
| `encoder.py` | 8–12 | Encoder init, forward, EMA target, EMA update, batch encode |
| `predictor.py` | 13–16 | Predictor init, action embed, forward, next-embedding prediction |
| `losses.py` | 17–22 | Prediction, variance, covariance, VICReg, JEPA, collapse metric |
| `training.py` | 23–24 | Training step, full training loop with history |
| `evaluation.py` | 25–29 | Latent rollouts, multi-step error, linear probe |
| `planning.py` | 30–37 | Goal encoding, MPC planning, episode runner, planner evaluation |
| `experiment.py` | 38 | Full end-to-end experiment |
| `analysis.py` | — | Visualization: architecture, training curves, MPC, probes, pipeline |
| `train.py` | — | Train JEPA, evaluate, save results to `results/` |
| `visualize.py` | — | Generate all plots/GIFs from saved results |

## How to Run

```bash
# Train JEPA (100 epochs, saves to results/jepa_results.pt)
python train.py

# Generate all visualizations from saved results
python visualize.py
```

### Expected Output (train.py + visualize.py)

```
Epochs: 100 x 12 steps = 1200 steps total; no early stopping; val every 12 steps; horizon every 30 epochs
Train loss: 0.963 -> 0.771
Collapse: 0.037 -> 1.09
Val loss: 0.956 -> 0.318 (best @ epoch 18)
Test Probe MAE: 0.368
MPC planner:   success 60% (h=5), 90% (h=10), 70% (h=20)
```

Outputs in `JEPA/images/`:
- `architecture.png` — U-shaped block diagram (PNG)
- `training_curves.gif` — Animated loss + collapse (val points, horizon-shift markers at epochs 30/60/90)
- `pipeline_animation.gif` — Full pipeline: room → encoder → predictor → probe
- `mpc_trajectories.png` — 5 success + 5 failure MPC paths
- `probe_accuracy.png` — Predicted vs actual position + error histogram

See [documentation page](../docs/jepa.html) for embedded visualizations.

## Key Concepts

### The Collapse Problem
An untrained encoder maps all observations to nearly identical vectors (std ≈ 0). Prediction becomes trivial but useless. VICReg forces the encoder to:
- **Variance loss**: Keep each dimension's std ≥ 1.0
- **Covariance loss**: Decorrelate dimensions (off-diagonal cov → 0)

### Architecture
```
Observation (1,6,6) → Encoder → Latent z (16)
                          ↓
Action (0-3) → Embed → Concat → Predictor → Predicted z'
                          ↑
                    Next Obs → Target Encoder (EMA) → Target z'
                          ↓
                JEPA Loss = MSE(pred, target) + VICReg(z)
```

### Acting = Planning in Latent Space
No policy network. Instead:
1. Encode goal position → goal embedding
2. Sample random action sequences
3. Roll out each in latent dynamics (predictor)
4. Score by distance to goal embedding
5. Execute first action of best sequence (MPC)

## Visualizations

Run `python train.py` then `python visualize.py` to generate:

| File | Description |
|------|-------------|
| `images/architecture.png` | U-shaped block diagram (PNG) |
| `images/training_curves.gif` | Animated loss + collapse (val points, horizon-shift markers) |
| `images/pipeline_animation.gif` | Full pipeline: room → encoder → predictor → probe |
| `images/mpc_trajectories.png` | 5 success + 5 failure MPC paths in 2D room |
| `images/probe_accuracy.png` | Predicted vs actual position scatter + error histogram |

See [documentation page](../docs/jepa.html) for embedded visualizations.

## Dependencies

- `torch` ≥ 2.0
- `matplotlib` (visualizations)
- `seaborn` (visualizations)
- `imageio` (animated GIFs)

## References

- LeCun, Y. "A Path Towards Autonomous Machine Intelligence" (2022)
- Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization" (ICLR 2022)
- Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023)

---

Built on [Deep-ML](https://deep-ml.com)