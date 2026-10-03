import gymnasium as gym
import gym_donkeycar

conf = {
    "exe_path": "remote",
    "port": 9091,
    "cam_resolution": (160, 120, 3),
    "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
}

env = gym.make("donkey-generated-track-v0", conf=conf)
obs, info = env.reset()

for i in range(1):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(info)
    
    if terminated or truncated:
        obs, info = env.reset()
    

env.close()