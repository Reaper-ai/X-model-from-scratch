# From Scratch Models

This repository contains from-scratch implementations of machine learning architectures built with pure PyTorch primitives — no high-level frameworks, just NumPy and PyTorch core. Each model is implemented step-by-step with full training, evaluation, and visualization pipelines.

## Models

| Model | Description | Key Features | Link |
|-------|-------------|--------------|------|
| **JEPA World Model** | Joint Embedding Predictive Architecture for 2D grid navigation | Self-supervised latent dynamics, VICReg regularization, MPC planning, linear probe evaluation | [📖 README](jepa/README.md) · [🌐 Article](docs/jepa.html) |
| **Neural ODEs** | A Neural ODE built from scratch that learns a pendulum's vector field from noisy observations — with a fixed-step MLP as reference baseline | Neural ODE + adjoint method, irregular-time training, rollout curriculum, dual-protocol evaluation, extended-range study | [📖 README](node/README.md) · [🌐 Article](docs/node.html) |
| *Coming Soon* | More from-scratch implementations in progress... | | |

## Quick Start

```bash
# JEPA World Model
cd jepa
python -m jepa.main          # Train the model
python -m jepa.visualize     # Generate visualizations

# Neural ODE vs Discrete MLP (damped pendulum)
cd node
python main.py full-pipeline       # Data -> train both -> evaluate -> figures
python main.py summarize           # Print comparison tables
python extended_experiment.py      # Extended IC-range variant
```

## Philosophy

Each implementation follows these principles:
- **From scratch**: No `nn.Sequential`, no pre-built modules — just `nn.Module` primitives
- **Modular**: Clean separation of data, model, training, evaluation, visualization
- **Documented**: Step-by-step implementation with in-code comments and external articles
- **Visualized**: Architecture diagrams, animated training curves, qualitative results
- **Reproducible**: Fixed seeds, early stopping, checkpointing, train/val/test splits
