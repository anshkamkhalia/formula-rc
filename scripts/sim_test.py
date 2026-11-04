import gymnasium as gym
import gym_donkeycar

conf = {
    "exe_path": "remote",
    "port": 9091,
    "cam_resolution": (160, 120, 3),
    "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
    "throttle_min": -1.0,
    "throttle_max": 1.0,
}

env = gym.make("donkey-generated-track-v0", conf=conf)
obs, info = env.reset()

for step in range(300):
    if step > 50:
        obs, reward, terminated, truncated, info = env.step([0.0, -1.0]) # only throttle
    else:
        obs, reward, terminated, truncated, info = env.step([0.0, 1.0]) # only throttle
    print(info["speed"])

env.close()