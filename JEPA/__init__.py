"""
JEPA World Model from Scratch - Package Exports.

Assembled from step-by-step solutions.
"""

# Environment (Steps 1-5)
from .env import (
    set_seed,
    init_env_state,
    apply_action,
    render_observation,
    env_reset,
    env_step,
)

# Data Collection (Steps 6-7)
from .data import (
    collect_random_transitions,
    build_transition_dataset,
)

# Encoder & EMA Target (Steps 8-12)
from .encoder import (
    init_encoder_params,
    encoder_forward,
    init_target_encoder,
    ema_update,
    encode_batch,
)

# Predictor (Steps 13-16)
from .predictor import (
    init_predictor_params,
    embed_action,
    predictor_forward,
    predict_next_embedding,
)

# Losses (Steps 17-22)
from .losses import (
    prediction_loss,
    variance_loss,
    covariance_loss,
    vicreg_regularizer,
    jepa_loss,
    collapse_metric,
)

# Training (Steps 23-24)
from .training import (
    jepa_training_step,
    train_jepa,
)

# Evaluation (Steps 25-29)
from .evaluation import (
    rollout_latent_dynamics,
    multi_step_prediction_error,
    init_linear_probe,
    train_linear_probe,
    probe_state_recovery,
)

# Planning / MPC (Steps 30-37)
from .planning import (
    encode_goal,
    latent_cost,
    sample_action_sequences,
    score_action_sequences,
    select_best_plan,
    mpc_step,
    run_mpc_episode,
    evaluate_planner,
)

# Full Experiment (Step 38)
from .experiment import (
    jepa_world_model_experiment,
)

__all__ = [
    # Environment
    "set_seed",
    "init_env_state",
    "apply_action",
    "render_observation",
    "env_reset",
    "env_step",
    # Data
    "collect_random_transitions",
    "build_transition_dataset",
    # Encoder
    "init_encoder_params",
    "encoder_forward",
    "init_target_encoder",
    "ema_update",
    "encode_batch",
    # Predictor
    "init_predictor_params",
    "embed_action",
    "predictor_forward",
    "predict_next_embedding",
    # Losses
    "prediction_loss",
    "variance_loss",
    "covariance_loss",
    "vicreg_regularizer",
    "jepa_loss",
    "collapse_metric",
    # Training
    "jepa_training_step",
    "train_jepa",
    # Evaluation
    "rollout_latent_dynamics",
    "multi_step_prediction_error",
    "init_linear_probe",
    "train_linear_probe",
    "probe_state_recovery",
    # Planning
    "encode_goal",
    "latent_cost",
    "sample_action_sequences",
    "score_action_sequences",
    "select_best_plan",
    "mpc_step",
    "run_mpc_episode",
    "evaluate_planner",
    # Experiment
    "jepa_world_model_experiment",
]