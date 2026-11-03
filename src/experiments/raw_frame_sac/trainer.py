from src.super.rl_driver_sac_super import RlDriverSACEnv
import tensorflow as tf
import numpy as np
import cv2 as cv
from src.super.driver import build_rl_driver
import os

class RawFrameSAC(RlDriverSACEnv):

    def __init__(self, model, max_cte=5.0, max_speed=7):
        tracks = [
            "donkey-generated-track-v0",
            "donkey-roboracingleague-track-v0",
            "donkey-warehouse-v0", 
            "donkey-warren-track-v0", 
            "donkey-mountain-track-v0", 
        ]
        super().__init__(
            model,
            tracks,
            max_cte=max_cte,
            n_latents=16,
            max_speed=max_speed,
        )

    def process_state(self):
        frames = list(self.obs_buffer)[-self.n_latents:] # most recent n frames, oldest first
        h, w, _ = self.img_size
        processed = [
            cv.resize(np.asarray(f), (w, h), interpolation=cv.INTER_AREA).astype(np.float32) / 255.0 # cv takes (width, height)
            for f in frames
        ]
        return np.stack(processed, axis=0) # (n_timesteps, height, width, channels)

if __name__ == "__main__":
    model = build_rl_driver()
    trainer = RawFrameSAC(model=model)
    trainer.load_models(method="raw_frame")
    trainer.train(epochs=10_000, method="raw_frame")