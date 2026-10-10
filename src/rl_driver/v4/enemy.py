import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym
import gym_donkeycar

from src.rl_driver.v3.rl_driverV3 import build_rl_driver

n_timesteps = 6

rl_driver = build_rl_driver()
rl_driver.load_weights("checkpoints/speed_demon.weights.h5")

conf = {
    "exe_path": "remote",
    "port": 9091,
    "cam_resolution": (160, 120, 3),
    "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
    "max_cte": 3.5,
    "host": "192.168.68.64",

    # car appearance
    "body_style": "f1",
    "body_rgb": (71, 71, 71),
    "car_name": "rl_driver",
    "font_size": 50,
}

def make_std(raw):
    return tf.clip_by_value(tf.math.softplus(raw) + 1e-5, 1e-3, 1.0)

def pad_frame_buffer(buffer):
    if len(buffer) < n_timesteps:
        pad = [np.zeros((120, 160, 3), dtype=np.float32)
            for _ in range(n_timesteps - len(buffer))]
        buffer = pad + list(buffer)
    return np.concatenate(buffer, axis=-1)

def utils_action(buffer):
    pred_buffer = pad_frame_buffer(buffer) # pad inputs

    # convert to tensor and action
    state_tensor = tf.convert_to_tensor([pred_buffer], dtype=tf.float32)
    steer_mu, steer_raw_std, throttle_mu, throttle_std_raw, value = rl_driver(state_tensor)

    # enforce positive deviations
    steer_std = make_std(steer_raw_std)
    throttle_std = make_std(throttle_std_raw)

    # create base normal distributions
    steer_dist = tfp.distributions.Normal(loc=steer_mu, scale=steer_std)
    throttle_dist = tfp.distributions.Normal(loc=throttle_mu, scale=throttle_std)

    # sample raw actions
    raw_steering = steer_dist.sample()
    raw_throttle = throttle_dist.sample()

    # squash boundaries with tanh and sigmoid
    steering = tf.math.tanh(raw_steering) # steering is between -1, 1
    throttle = tf.math.sigmoid(raw_throttle) # throttle between 0, 1

    # calculate log probabilities
    log_prob_steer = steer_dist.log_prob(raw_steering) - tf.math.log(1.0 - steering ** 2 + 1e-6)
    log_prob_throttle = throttle_dist.log_prob(raw_throttle) - tf.math.log(throttle * (1.0 - throttle) + 1e-6)

    # total log probability
    total_log_prob = log_prob_steer + log_prob_throttle

    # convert value from tensors to standard scalars
    return float(steering), float(throttle), float(raw_steering), float(raw_throttle), float(total_log_prob), float(value), pred_buffer

env = gym.make(sys.argv[1], conf=conf)
frame_buffer = []

try:
    obs, info = env.reset()
    frame_buffer.append(np.array(obs/255.0, dtype=np.float32))

    while True:
        steering, throttle, raw_steering, raw_throttle, total_logp, value, inputs = utils_action(frame_buffer)
        obs, _, terminated, truncated, info = env.step([steering, throttle])

        frame_buffer.append(obs / 255.0)

        if len(frame_buffer) > 6:
            frame_buffer = frame_buffer[-6:] # take 6 most recent elements

        if terminated or truncated:
            frame_buffer = []
            frame_buffer.append(np.array(obs/255.0, dtype=np.float32))
finally:
    env.close()