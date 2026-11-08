from src.super.rl_driver_ppo_super import RlDriverPPOEnv
import tensorflow as tf
import numpy as np
from src.super.driver import build_rl_driver
import os
class RawFramePPO(RlDriverPPOEnv):

    def __init__(self, model, max_cte=5.0, timesteps=8, max_speed=8):
        tracks = [
            "donkey-generated-track-v0",
            "donkey-warren-track-v0",
            "donkey-generated-track-v0",
            "donkey-warehouse-v0",
            "donkey-generated-track-v0",
        ]
        super().__init__(
            model,
            tracks,
            max_cte=max_cte,
            timesteps=timesteps,
            max_speed=max_speed,
            track_steps=700, # tune
        )

    def process_obs(self, obs):
        return np.array(obs/255.0, dtype=np.float32)

if __name__ == "__main__":
    path = "checkpoints/latest_raw_frame_ppo.weights.h5"
    model = build_rl_driver(8)
    if os.path.exists(path):
        model.load_weights(path)
        print("loaded model")
    trainer = RawFramePPO(model=model)
    trainer.train(epochs=10_000)