import os
from src.super.bc_trainer_super import BCTrainerSuper
import numpy as np
from src.super.driver import build_rl_driver

class BCTrainRawFrame(BCTrainerSuper):

    def __init__(self, input_shape, model): # raw frame shape
        super().__init__(input_shape, model)

    def process_obs(self, arr):
        return arr/255.0 # thats it for raw frame

if __name__ == "__main__":
    model = build_rl_driver()
    if os.path.exists("checkpoints/best_bc_raw_frame.keras"):
        model.load_weights("checkpoints/best_bc_raw_frame.keras")
        print("loading weights")
    else:
        print("loading new")
    train = BCTrainRawFrame(input_shape=(120, 160, 18), model=model)
    train.train()