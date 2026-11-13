import cv2 as cv
import numpy as np
import gym_donkeycar
import gymnasium as gym
import os
import subprocess
import uuid
from pynput import keyboard
import time

pressed_keys = set()

def on_press(key):
    pressed_keys.add(key)

def on_release(key):
    pressed_keys.discard(key)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

STEER_AMOUNT = 0.6
THROTTLE_AMOUNT = 0.3

def get_teleop():
    steering = 0.0
    throttle = 0.0
    if keyboard.Key.left in pressed_keys:
        steering -= STEER_AMOUNT
    if keyboard.Key.right in pressed_keys:
        steering += STEER_AMOUNT
    if keyboard.Key.up in pressed_keys:
        throttle += THROTTLE_AMOUNT
    if keyboard.Key.down in pressed_keys:
        throttle -= THROTTLE_AMOUNT
    return steering, throttle

# make output directory
os.makedirs("vae_data", exist_ok=True)

# conf stays the same
conf = {
    "exe_path": "remote",
    "port": 9091,
    "cam_resolution": (160, 120, 3),
    "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
    "max_cte": 10.0,

    # car appearance
    "body_style": "f1", 
    "body_rgb": (71, 71, 71),      
    "car_name": "rl_driver",
    "font_size": 50,
}

# commands
OPEN_DONKEY_CMD = ["open", "-a", "donkey_sim"]
KILL_DONKEY_CMD = ["killall", "donkey_sim"]

tracks = ["donkey-generated-track-v0", "donkey-warehouse-v0", "donkey-minimonaco-track-v0", "donkey-mountain-track-v0"]
n_laps_per_track = 5

for track in tracks:

    print(f"track: {track}\n")
    data = [] # store observations here

    # create env
    subprocess.Popen(OPEN_DONKEY_CMD, start_new_session=True, close_fds=True) # start process and detach to hand over to cpu
    time.sleep(5) # give donkeysim time to start
    env = gym.make(track, conf=conf)
    obs, info = env.reset()

    while info["lap_count"] < n_laps_per_track:    

        # teleop goes here to get steering and throttle
        steering, throttle = get_teleop()

        obs, reward, terminated, truncated, info = env.step([steering, throttle])
        data.append(np.array(obs, dtype=np.uint8))

    # kill donkeysim and then move to next track
    subprocess.run(KILL_DONKEY_CMD)
    time.sleep(5)

    # write data
    print(f"writing out {len(data)} images")
    for idx in range(len(data)):
        img = cv.cvtColor(data[idx], cv.COLOR_RGB2BGR)
        cv.imwrite(f"lane_seg_data/sim/{uuid.uuid1()}.png", img)