from src.super.rl_driver_ppo_super import RlDriverPPOEnv
import tensorflow as tf
import numpy as np
from src.super.driver import build_rl_driver
import os
class RawFramePPO(RlDriverPPOEnv):

    def __init__(self, model, max_cte=5.15, timesteps=6, max_speed=6):
        tracks = [
            "donkey-generated-track-v0"
        ]
        super().__init__(
            model,
            tracks,
            max_cte=max_cte,
            timesteps=timesteps,
            max_speed=max_speed,
            track_steps=4096, # tune
        )

    def process_obs(self, obs):
        return np.array(obs/255.0, dtype=np.float32)

if __name__ == "__main__":
    path = "checkpoints/best_bc_raw_frame.weights.h5"
    model = build_rl_driver()
    if os.path.exists(path):
        model.load_weights(path)
        print(f"loaded model from {path}")
    trainer = RawFramePPO(model=model)
    trainer.train(epochs=10_000)