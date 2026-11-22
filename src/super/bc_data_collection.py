import os
import time
import uuid
from collections import deque
import numpy as np
import gymnasium as gym
import gym_donkeycar
from pynput import keyboard

OUT_DIR = "bc_data"
N_STACK = 6
SPEED_SCALE = 6.0
N_LAPS_PER_TRACK = 3
DISCARD_ON_CRASH = 20

TRACKS = ["donkey-generated-track-v0"]

conf = {
    "exe_path": "remote",
    "port": 9091,
    "cam_resolution": (160, 120, 3),
    "cam_config": {
        "img_w": 160, "img_h": 120, "img_d": 3, "img_enc": "JPG",
        "fov": 63,
    },
    "max_cte": 5.0,
    "throttle_min": 0.0,
    "throttle_max": 1.0,
    "body_style": "f1",
    "body_rgb": (71, 71, 71),
    "car_name": "rl_driver",
    "font_size": 50,
}

MAX_STEER = 1.0
STEER_RATE = 3.0
COUNTER_RATE = 6.0
CENTER_RATE = 4.0
MAX_THROTTLE = 1.0
THROTTLE_RATE = 1.0
BRAKE_RATE = 3.0
COAST_RATE = 0.8

pressed_keys = set()
stop_requested = False


def on_press(key):
    global stop_requested
    pressed_keys.add(key)
    if key == keyboard.Key.esc:
        stop_requested = True


def on_release(key):
    pressed_keys.discard(key)


keyboard.Listener(on_press=on_press, on_release=on_release).start()


def approach(value, target, max_delta):
    if value < target:
        return min(value + max_delta, target)
    return max(value - max_delta, target)


class Teleop:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.last_t = time.monotonic()

    def update(self):
        now = time.monotonic()
        dt = min(now - self.last_t, 0.1)
        self.last_t = now

        left = keyboard.Key.left in pressed_keys
        right = keyboard.Key.right in pressed_keys
        up = keyboard.Key.up in pressed_keys
        down = keyboard.Key.down in pressed_keys

        target = MAX_STEER * ((1.0 if right else 0.0) - (1.0 if left else 0.0))
        if target == 0.0:
            rate = CENTER_RATE
        elif self.steering * target < 0:
            rate = COUNTER_RATE
        else:
            rate = STEER_RATE
        self.steering = approach(self.steering, target, rate * dt)

        if down:
            self.throttle = approach(self.throttle, 0.0, BRAKE_RATE * dt)
        elif up:
            self.throttle = approach(self.throttle, MAX_THROTTLE, THROTTLE_RATE * dt)
        else:
            self.throttle = approach(self.throttle, 0.0, COAST_RATE * dt)

        return self.steering, self.throttle


teleop = Teleop()


def new_frame_buffer(first_obs):
    first = np.asarray(first_obs, dtype=np.uint8)
    return deque([first] * N_STACK, maxlen=N_STACK)


def new_speed_buffer(first_speed):
    return deque([first_speed / SPEED_SCALE] * N_STACK, maxlen=N_STACK)


def commit(ep, store, discard=0):
    n = max(len(ep["frames"]) - discard, 0)
    for k in ("frames", "speeds", "actions"):
        store[k].extend(ep[k][:n])
        ep[k].clear()
    return n


os.makedirs(OUT_DIR, exist_ok=True)

for track in TRACKS:
    print(f"track: {track}")
    store = {"frames": [], "speeds": [], "actions": []}
    ep = {"frames": [], "speeds": [], "actions": []}
    episode = 0
    laps_done = 0

    time.sleep(5)
    env = gym.make(track, conf=conf)

    obs, info = env.reset()
    teleop.reset()
    frame_buf = new_frame_buffer(obs)
    speed_buf = new_speed_buffer(info.get("speed", 0.0))
    prev_lap = info.get("lap_count", 0)

    while laps_done < N_LAPS_PER_TRACK and not stop_requested:
        steering, throttle = teleop.update()
        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        ep["frames"].append(np.stack(frame_buf))
        ep["speeds"].append(np.array(speed_buf, dtype=np.float32))
        ep["actions"].append([steering, throttle])

        obs, reward, terminated, truncated, info = env.step(
            np.array([steering, throttle], dtype=np.float32)
        )
        frame_buf.append(np.asarray(obs, dtype=np.uint8))
        speed_buf.append(info.get("speed", 0.0) / SPEED_SCALE)

        lap = info.get("lap_count", prev_lap)
        if lap > prev_lap:
            laps_done += lap - prev_lap
            print(f"  lap {laps_done}/{N_LAPS_PER_TRACK}")
        prev_lap = lap

        if terminated or truncated:
            discard = DISCARD_ON_CRASH if terminated else 0
            kept = commit(ep, store, discard)
            print(f"  episode {episode} ended, kept {kept} samples")
            episode += 1

            obs, info = env.reset()
            teleop.reset()
            frame_buf = new_frame_buffer(obs)
            speed_buf = new_speed_buffer(info.get("speed", 0.0))
            prev_lap = info.get("lap_count", 0)

    commit(ep, store)
    env.close()
    time.sleep(5)

    if store["frames"]:
        path = os.path.join(OUT_DIR, f"{track}_{uuid.uuid4().hex[:8]}.npz")
        np.savez_compressed(
            path,
            frames=np.stack(store["frames"]),
            speeds=np.stack(store["speeds"]).astype(np.float32),
            actions=np.array(store["actions"], dtype=np.float32),
        )
        print(f"saved {len(store['frames'])} samples to {path}")

    if stop_requested:
        break