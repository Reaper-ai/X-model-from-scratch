# From Scratch Models

This repository contains from-scratch implementations of machine learning architectures built with pure PyTorch primitives — no high-level frameworks, just NumPy and PyTorch core. Each model is implemented step-by-step with full training, evaluation, and visualization pipelines.

## Models

| Model | Description | Key Features | Link |
|-------|-------------|--------------|------|
| **JEPA World Model** | Joint Embedding Predictive Architecture for 2D grid navigation | Self-supervised latent dynamics, VICReg regularization, MPC planning, linear probe evaluation | [📖 README](jepa/README.md) · [🌐 Article](docs/JEPA.html) |
| *Coming Soon* | More from-scratch implementations in progress... | | |

## Quick Start

```bash
# JEPA World Model
cd jepa
python -m jepa.main          # Train the model
python -m jepa.visualize     # Generate visualizations
```

## Repository Structure

```
├── jepa/                    # JEPA World Model
│   ├── data_env.py          # Environment & dataset generation
│   ├── model.py             # Encoder, TargetEncoder, Predictor, LinearProbe + losses
│   ├── training_eval.py     # Trainer with early stopping, validation
│   ├── main.py              # Orchestration entry point
│   ├── visualize.py         # Architecture, training curves, trajectories
│   ├── data/                # Checkpoints, history, predictions
│   └── README.md            # Detailed documentation
├── docs/
│   ├── index.html           # Model gallery (card-based)
│   ├── JEPA.html            # JEPA deep-dive article
│   └── images/JEPA/         # Architecture, training GIF, trajectory grid
├── scaffold.py              # Original scaffold (reference)
├── model.py                 # Original monolithic file (reference)
└── README.md                # This file
```

## Philosophy

Each implementation follows these principles:
- **From scratch**: No `nn.Sequential`, no pre-built modules — just `nn.Module` primitives
- **Modular**: Clean separation of data, model, training, evaluation, visualization
- **Documented**: Step-by-step implementation with in-code comments and external articles
- **Visualized**: Architecture diagrams, animated training curves, qualitative results
- **Reproducible**: Fixed seeds, early stopping, checkpointing, train/val/test splits

## Built on

[Deep-ML](https://deep-ml.com) — Implement ML from scratch.
