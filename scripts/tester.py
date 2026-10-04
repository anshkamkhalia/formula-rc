import numpy as np
import tensorflow as tf
import gym_donkeycar
import gymnasium as gym
from collections import deque

from src.rl_driver.models.rl_driverV2 import build_rl_driver

WEIGHTS = "checkpoints/best1.weights.h5"
N_TIMESTEPS = 6
STOCHASTIC = False  # True = sample like training, False = deterministic (mean action)

model = build_rl_driver()
model.load_weights(WEIGHTS)

def pad_frame_buffer(buffer):
    buffer = list(buffer)
    if len(buffer) < N_TIMESTEPS:
        pad = [np.zeros((120, 160, 3), dtype=np.float32)
               for _ in range(N_TIMESTEPS - len(buffer))]
        buffer = pad + buffer
    return np.concatenate(buffer, axis=-1)

def get_action(buffer):
    state = tf.convert_to_tensor([pad_frame_buffer(buffer)], dtype=tf.float32)
    steer_mu, steer_raw_std, throttle_mu, throttle_raw_std, value = model(state)

    if STOCHASTIC:
        steer_std = tf.math.softplus(steer_raw_std) + 1e-5
        throttle_std = tf.math.softplus(throttle_raw_std) + 1e-5
        raw_steer = steer_mu + steer_std * tf.random.normal(tf.shape(steer_mu))
        raw_throttle = throttle_mu + throttle_std * tf.random.normal(tf.shape(throttle_mu))
    else:
        raw_steer, raw_throttle = steer_mu, throttle_mu

    steering = float(tf.math.tanh(raw_steer))
    throttle = float(tf.math.sigmoid(raw_throttle))
    return steering, throttle

conf = {
    "exe_path": "remote",
    "port": 9091,
    "cam_resolution": (160, 120, 3),
    "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
    "max_cte": 2.5,
}

env = gym.make("donkey-generated-track-v0", conf=conf)

# maxlen=6 matches the training truncation (frame_buffer[-6:])
frame_buffer = deque(maxlen=N_TIMESTEPS)
obs, info = env.reset()
frame_buffer.append(obs / 255.0)

episode_reward, episode_steps = 0.0, 0

try:
    while True:
        steering, throttle = get_action(frame_buffer)
        obs, reward, terminated, truncated, info = env.step([steering, throttle])
        frame_buffer.append(obs / 255.0)
        episode_steps += 1

        if terminated or truncated:
            print(f"episode over | steps {episode_steps} | cte {info.get('cte'):.2f}")
            frame_buffer.clear()
            obs, info = env.reset()
            frame_buffer.append(obs / 255.0)
            episode_steps = 0
except KeyboardInterrupt:
    pass
finally:
    env.close()