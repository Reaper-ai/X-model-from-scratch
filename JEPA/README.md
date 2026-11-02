# JEPA World Model from Scratch

Build an action-conditioned Joint Embedding Predictive Architecture (JEPA) world model in pure PyTorch. An encoder and EMA target learn collapse-resistant latents of a 2D room via VICReg, a predictor rolls dynamics in embedding space, and acting is random-shooting MPC toward goal embeddings. No decoder, no pixels, no learned policy.

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
| `scaffold.py` | — | End-to-end demo (run this) |
| `train.py` | — | Train JEPA, evaluate, save results to `results/` |
| `visualize.py` | — | Generate all plots/GIFs from saved results |

## How to Run

```bash
# Quick demo (runs in ~30s)
python scaffold.py

# Train, then generate all visualizations
python train.py
python visualize.py
```

### Expected Output (scaffold.py)

```
transitions: 512 | obs shape: (512, 1, 6, 6)
collapse metric BEFORE training: 0.037  (near 0 = collapsed)
JEPA loss: 0.963 -> 0.705
collapse metric AFTER training:  1.073  (target std is 1.0)
5-step latent prediction MSE (trained): 15.422
linear probe on agent position: mean abs error = 0.30 cells (room is 6x6)
MPC planner:   success 90%, mean final distance 0.20
random policy: success 40%
```

## Key Concepts

### The Collapse Problem
An untrained encoder maps all observations to nearly identical vectors (std ≈ 0). Prediction becomes trivial but useless. VICReg forces the encoder to:
- **Variance loss**: Keep each dimension's std ≥ 1.0
- **Covariance loss**: Decorrelate dimensions (off-diagonal cov → 0)

### Architecture
```
Observation (1,8,8) → Encoder → Latent z (32)
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
| `images/architecture.png` | Full model diagram (U-shaped block diagram) |
| `images/training_curves.gif` | Animated loss + collapse (val points, horizon-shift markers) |
| `images/pipeline_animation.gif` | Full pipeline: room → encoder → predictor → probe |
| `images/mpc_trajectories.png` | 5 success + 5 failure MPC paths in 2D room |
| `images/probe_accuracy.png` | Predicted vs actual position scatter + error histogram |

See [documentation page](docs/jepa.html) for embedded visualizations.

## Dependencies

- `torch` ≥ 2.0
- `matplotlib` (visualizations)
- `seaborn` (visualizations)
- `imageio` (animated GIFs)
- `graphviz` (optional, for architecture diagram)

## References

- LeCun, Y. "A Path Towards Autonomous Machine Intelligence" (2022)
- Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization" (ICLR 2022)
- Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023)

---

Built on [Deep-ML](https://deep-ml.com)