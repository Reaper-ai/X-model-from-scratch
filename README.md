# jepa-world-model-from-scratch
Build an action-conditioned Joint Embedding Predictive Architecture (JEPA) world model in pure PyTorch. An encoder and EMA target learn collapse-resistant latents of a 2D room via VICReg, a predictor rolls dynamics in embedding space, and acting is random-shooting MPC toward goal embeddings. No decoder, no pixels, no learned policy.
