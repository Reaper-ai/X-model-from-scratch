# X-Model-From-Scratch

From-scratch implementations of modern ML architectures. Built step-by-step with pure PyTorch — no high-level libraries, no hidden magic.

## Models

| Model | Description | Demo | Documentation |
|-------|-------------|------|---------------|
| **[JEPA World Model](./JEPA)** | Action-conditional Joint Embedding Predictive Architecture with VICReg regularization. Learns collapse-resistant latents in a 2D grid world, predicts multi-step dynamics in embedding space, and plans via random-shooting MPC — all without a decoder or learned policy. | `python JEPA/scaffold.py` | [View Docs](./JEPA/README.md) |
| *Model X* | *Coming soon* | — | — |
| *Model Y* | *Coming soon* | — | — |
| *Model Z* | *Coming soon* | — | — |

## Philosophy

Each model is implemented **from scratch** — every layer, loss, and training loop written explicitly in PyTorch. The code is organized as readable, modular Python functions (not classes) to make the data flow transparent.

## Project Structure

```
X-model-from-scratch/
├── JEPA/                    # JEPA World Model
│   ├── env.py              # 2D room environment
│   ├── data.py             # Transition data collection
│   ├── encoder.py          # Encoder + EMA target
│   ├── predictor.py        # Action-conditioned predictor
│   ├── losses.py           # VICReg + JEPA losses
│   ├── training.py         # Training loop
│   ├── evaluation.py       # Rollouts, probes, metrics
│   ├── planning.py         # MPC in latent space
│   ├── experiment.py       # Full experiment pipeline
│   ├── analysis.py         # Visualization & plotting
│   ├── scaffold.py         # End-to-end demo
│   ├── train.py            # Train + save metrics/params
│   └── visualize.py        # Generate all plots from saved results
├── docs/
│   ├── index.html          # Model gallery (cards)
│   └── jepa.html           # JEPA deep-dive with visualizations
└── README.md               # This file
```

## Quick Start

```bash
# Install dependencies (project venv)
uv sync

# Run JEPA demo
python JEPA/scaffold.py

# Train JEPA, then generate visualizations
python JEPA/train.py
python JEPA/visualize.py
```

## JEPA Highlights

- **No decoder** — learns purely from prediction in latent space
- **VICReg regularization** — variance + covariance losses prevent collapse
- **EMA target encoder** — stable targets for bootstrap prediction
- **Latent MPC** — plan by random-shooting in embedding space
- **Linear probe** — verifies latents encode true state (position)

---

Built on [Deep-ML](https://deep-ml.com) • [@Reaper-ai](https://github.com/Reaper-ai)