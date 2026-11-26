import os
import time
import uuid
import tensorflow as tf
import numpy as np
import gym_donkeycar
import gymnasium as gym
from collections import deque
from src.super.driver import build_rl_driver

track = "donkey-generated-track-v0"
n_laps = 3
throttle_boosting_range = [0.05, 0.10] # new idea: throttle boosting
# throttle boosting will add small throttle boost to the teacher model to gradually increase speed and make the training data faster
# the model trained on this sped up data will be then used and boosted again iteratively about 2-3 times so the car is actually fast instead of driving as slow as me (product of behavior cloning)
# this helps to fix the problem of me driving as the expert (not 100%)

model_path = "checkpoints/teacher1.weights.h5"
output_dir = "bc_data"
timesteps = 6
frame_buffer = deque(maxlen=timesteps)
speed_buffer = deque(maxlen=timesteps)
max_cte = 4
max_speed = 6
img_shape = (120, 160, timesteps*3)
discard_on_crash = 20 
throttle_boost = float(np.random.uniform(*throttle_boosting_range))

# configs
conf = {
    "exe_path": "remote",
    "port": 9091,
    "cam_resolution": (160, 120, 3),
    "cam_config": {
        "img_w": 160,
        "img_h": 120,
        "img_d": 3,
        "img_enc": "JPG",
        "fov": 63, # vertical fov matches ssd camera
    },
    "max_cte": max_cte,
    "throttle_min": 0.0, # removing braking temporarily
    "throttle_max": 1.0, 

    # car appearance
    "body_style": "f1", 
    "body_rgb": (71, 71, 71),      
    "car_name": "rl_driver",
    "font_size": 50,
}

# we are keeping raw frames for the teacher model because its what i have right now and its more likely the most reliable because the inputs are consistent (lane/vae models might have some noise)
# it shouldnt matter for the sim because it's being trained on it, but each of the 3 models used for testing will use their own respective input processing
# data is going to be saved in uint8 frame format so that during training, the vae/lane/normalization needs to be applied to the frame before its passed in
# speed and action remain the same
def process_obs(obs):
    return obs/255.0

def reset_buffers(obs, info):
    first = np.asarray(obs, dtype=np.uint8)
    frame_buffer.clear()
    speed_buffer.clear()
    frame_buffer.extend([first] * timesteps)
    speed_buffer.extend([info.get("speed", 0.0) / max_speed] * timesteps)

def model_inputs():
    frames = np.concatenate(list(frame_buffer), axis=-1).astype(np.float32)
    frames = process_obs(frames).reshape((1,) + img_shape)
    speeds = np.array(speed_buffer, dtype=np.float32)[None]
    return tf.constant(frames, dtype=tf.float32), tf.constant(speeds, dtype=tf.float32)

def commit(ep, store, discard=0):
    n = max(len(ep["frames"]) - discard, 0)
    for k in ("frames", "speeds", "actions"):
        store[k].extend(ep[k][:n])
        ep[k].clear()
    return n

# load model
driver = build_rl_driver()
driver.load_weights(model_path)

@tf.function
def _forward(frames, speeds):
    steer_mu, _, throttle_mu, _, _ = driver([frames, speeds], training=False)
    return steer_mu, throttle_mu

os.makedirs(output_dir, exist_ok=True)
print(f"track: {track}, throttle boost: {throttle_boost:.3f}")

env = gym.make(track, conf=conf)
obs, info = env.reset()

# pad buffers
reset_buffers(obs, info)

store = {"frames": [], "speeds": [], "actions": []}
ep = {"frames": [], "speeds": [], "actions": []}
episode = 0
laps_done = 0
prev_lap = info.get("lap_count", 0)

try:
    while laps_done < n_laps:
        frames_in, speeds_in = model_inputs()
        steer_mu, throttle_mu = _forward(frames_in, speeds_in)
        steering = float(np.clip(steer_mu.numpy()[0][0], -1.0, 1.0))
        throttle = float(np.clip(throttle_mu.numpy()[0][0] + throttle_boost, 0.0, 1.0))

        if steering == 1.0 or throttle == 1.0:
            print(f"clipped: {steering}, {throttle}")

        ep["frames"].append(np.stack(frame_buffer))
        ep["speeds"].append(np.array(speed_buffer, dtype=np.float32))
        ep["actions"].append([steering, throttle])

        obs, _, terminated, truncated, info = env.step(
            np.array([steering, throttle], dtype=np.float32)
        )
        frame_buffer.append(np.asarray(obs, dtype=np.uint8))
        speed_buffer.append(info.get("speed", 0.0) / max_speed)

        lap = info.get("lap_count", prev_lap)
        if lap > prev_lap:
            laps_done += lap - prev_lap
            print(f"  lap {laps_done}/{n_laps}")
        prev_lap = lap

        if terminated or truncated:
            discard = discard_on_crash if terminated else 0
            kept = commit(ep, store, discard)
            print(f"  episode {episode} ended, kept {kept} samples")
            episode += 1

            obs, info = env.reset()
            reset_buffers(obs, info)
            prev_lap = info.get("lap_count", 0)
except KeyboardInterrupt:
    pass

commit(ep, store)
env.close()
time.sleep(5)

if store["frames"]:
    path = os.path.join(output_dir, f"{track}_{uuid.uuid4().hex[:8]}.npz")
    np.savez_compressed(
        path,
        frames=np.stack(store["frames"]),
        speeds=np.stack(store["speeds"]).astype(np.float32),
        actions=np.array(store["actions"], dtype=np.float32),
    )
    print(f"saved {len(store['frames'])} samples to {path}")