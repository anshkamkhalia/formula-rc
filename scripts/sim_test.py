import gymnasium as gym
import gym_donkeycar
import ctypes, platform
from screeninfo import get_monitors
from pynput import mouse, keyboard

monitors = get_monitors()

import time
from pynput.mouse import Button, Controller

mouse = Controller()

conf = {
    "exe_path": "remote",
    "port": 9091,
    "cam_resolution": (160, 120, 3),
    "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
}

env = gym.make("donkey-generated-track-v0", conf=conf)
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step([0.0, 0.2]) # only throttle
    env.close()
    mouse.position = (1586, 1036)
    time.sleep(3)
    mouse.click(Button.left, 2)
    print("clicked button")
    time.sleep(3)
    break

env = gym.make("donkey-generated-roads-v0", conf=conf)
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step([0.0, 0.2]) # only throttle
    env.close()
    mouse.position = (1586, 1036)
    time.sleep(3)
    mouse.click(Button.left, 2)
    print("clicked button")
    time.sleep(3)
    break