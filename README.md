# JEPA World Model from Scratch

Build an action-conditioned Joint Embedding Predictive Architecture (JEPA) world model in pure PyTorch. An encoder and EMA target learn collapse-resistant latents of a 2D room via VICReg, a predictor rolls dynamics in embedding space, and acting is random-shooting MPC toward goal embeddings. No decoder, no pixels, no learned policy.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** init_env_state
- [x] **2.** apply_action
- [x] **3.** render_observation
- [x] **4.** env_reset
- [x] **5.** env_step
- [x] **6.** collect_random_transitions
- [x] **7.** build_transition_dataset
- [x] **8.** init_encoder_params
- [x] **9.** encoder_forward
- [x] **10.** init_target_encoder
- [x] **11.** ema_update
- [x] **12.** encode_batch
- [x] **13.** init_predictor_params
- [x] **14.** embed_action
- [x] **15.** predictor_forward
- [x] **16.** predict_next_embedding
- [x] **17.** prediction_loss
- [x] **18.** variance_loss
- [x] **19.** covariance_loss
- [x] **20.** vicreg_regularizer
- [x] **21.** jepa_loss
- [x] **22.** collapse_metric
- [x] **23.** jepa_training_step
- [x] **24.** train_jepa
- [x] **25.** rollout_latent_dynamics
- [x] **26.** multi_step_prediction_error
- [x] **27.** init_linear_probe
- [x] **28.** train_linear_probe
- [x] **29.** probe_state_recovery
- [x] **30.** encode_goal
- [x] **31.** latent_cost
- [x] **32.** sample_action_sequences
- [x] **33.** score_action_sequences
- [x] **34.** select_best_plan
- [x] **35.** mpc_step
- [x] **36.** run_mpc_episode
- [x] **37.** evaluate_planner
- [x] **38.** jepa_world_model_experiment

---

Built on Deep-ML.
