import tensorflow as tf
import numpy as np
import os
from src.super.driver import build_rl_driver
from collections import deque
import subprocess
import time
import gymnasium as gym
import gym_donkeycar
import matplotlib.pyplot as plt

import sys

class BCTrainerSuper:

    def __init__(self, input_shape=None, model=None):

        self.driver = model # trainer passes it in, allows for loading previous models easily
        self.optimizer = tf.keras.optimizers.Adam(2e-4) # optimizer
        self.data_dir = "bc_data" # dir for data
        self.batch_size = 16 # small batches for now
        self.loss = tf.keras.losses.MeanSquaredError() # simple regression
        self.epochs = 100 # change later ig
        self.input_shape = input_shape # needed for dynamic code
        self.sim_binary = os.path.abspath("sim/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim")
        self.eval_history = [] # for plotting
        self.update_history = [] # also for plotting
        self.log_dir = "training_logs/bc"
        self.max_cte = 4
        self.n_timesteps = 6
        self.max_speed = 6.0
        self.prev_laps = 0
        self.slow_speed = 1.0

        self.save_this_model = False
        self.best_eval_stats = (float("inf"), float("inf"))

        self.img_shape = (120, 160, 3)
        self.lap_times = []
        self.REWARD_SCALE = 0.1
        self.env = None
        self.frame_buffer = deque(maxlen=self.n_timesteps)
        self.speed_buffer = deque(maxlen=self.n_timesteps)

        # configs
        self.conf = {
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
            "max_cte": self.max_cte,
            "throttle_min": 0.0, # removing braking temporarily
            "throttle_max": 1.0, 

            # car appearance
            "body_style": "f1", 
            "body_rgb": (71, 71, 71),      
            "car_name": "rl_driver",
            "font_size": 50,
        }

    def restart_episode(self):
        obs, info = self.env.reset()
        first = self.process_obs(obs)
        first_speed = float(info.get("speed", 0.0))
        self.frame_buffer = deque([first] * self.n_timesteps, maxlen=self.n_timesteps)
        self.speed_buffer = deque([first_speed] * self.n_timesteps, maxlen=self.n_timesteps)

    def plot_eval(self, epoch, metrics, method):

        os.makedirs(self.log_dir, exist_ok=True)
        self.eval_history.append({"epoch": epoch, **metrics})

        panels = [
            ("return", "episode return (raw)"),
            ("lap_time", "lap time (s)"),
            ("cte", "mean |CTE|"),
            ("crash_rate", "crash rate"),
        ]
        epochs = np.array([h["epoch"] for h in self.eval_history])

        for key, label in panels:
            vals = np.array([h[key] for h in self.eval_history], dtype=float)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, vals, marker="o", markersize=3)
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            ax.set_title(f"PPO eval {label} ({method})")
            ax.grid(alpha=0.3)
            if key == "crash_rate":
                ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, f"{method}_ppo_{key}.png"), dpi=120)
            plt.close(fig)

    def plot_update_stats(self, stats, epoch, method):

        os.makedirs(self.log_dir, exist_ok=True)
        self.update_history.append({
            "epoch": epoch,
            "loss": float(stats["losses"][-1]),
        })
        epochs = np.array([h["epoch"] for h in self.update_history])

        for key, label in [("loss", "total loss")]:
            vals = np.array([h[key] for h in self.update_history], dtype=float)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, vals, marker="o", markersize=3)
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            ax.set_title(f"BC {label} ({method})")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, f"{method}_bc_{key}.png"), dpi=120)
            plt.close(fig)

    def plot_throttle(self, throttle_history, epoch, method):

        throttle_dir = os.path.join(self.log_dir, "throttle")
        os.makedirs(throttle_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        for i, ep_throttles in enumerate(throttle_history):
            ax.plot(np.arange(len(ep_throttles)), ep_throttles, linewidth=1, label=f"episode {i + 1}")

        ax.set_xlabel("step")
        ax.set_ylabel("throttle")
        ax.set_ylim(0.0 - 0.05, 1.05)
        ax.set_title(f"eval throttle epoch {epoch}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(throttle_dir, f"{method}_throttle_epoch_{epoch}.png"), dpi=120)
        plt.close(fig)

    def pad_buffers(self, buffer):
        buffer = list(buffer)
        if len(buffer) == 0:
            buffer = [np.zeros(self.img_shape, dtype=np.float32)]
        if len(buffer) < self.n_timesteps:
            buffer = [buffer[0]] * (self.n_timesteps - len(buffer)) + buffer
        return np.concatenate(buffer, axis=-1)

    def get_speeds(self):
        speeds = list(self.speed_buffer)
        if len(speeds) == 0:
            speeds = [0.0]
        if len(speeds) < self.n_timesteps:
            speeds = [speeds[0]] * (self.n_timesteps - len(speeds)) + speeds
        return np.array(speeds, dtype=np.float32) / self.max_speed
        
    def reward(self, info, steps, terminated=False, train=False, epoch=None):
        max_cte = self.max_cte
        cte = abs(info["cte"])

        # keep the sign so reversing is visible
        speed = float(np.clip(info["speed"], -self.max_speed, self.max_speed))
        
        if abs(speed) <= 0.005:
            speed = 0.0 # fluctuations (not moving)
        laps = info["lap_count"]

        if steps <= 1:
            self.prev_laps = laps

        # terminal penalty
        if terminated:
            return -20.0 * self.REWARD_SCALE

        speed_norm = speed / self.max_speed
        cte_frac = cte / max_cte
        centering = max(0.0, 1.0 - cte_frac)

        curr_reward = 2.0 * speed_norm * centering
        curr_reward -= 0.5 * cte_frac ** 2

        if speed < self.slow_speed:
            curr_reward -= 3.0

        lap_time = float(info.get("last_lap_time", 0.0))

        # lap reward, paid once when lap_count increments
        if laps > self.prev_laps:
            self.prev_laps = laps

            if lap_time <= 0.0: # sim hasn't updated it yet, treat as a slow lap
                lap_time = 100.0
            else:
                self.lap_times.append(lap_time)

            target = 5 # small track
            bonus = float(np.clip(10.0 + 15.0 * (target - lap_time), 2.0, 15.0))

            print(f"log: lap {laps} in {lap_time:.2f}s, bonus {bonus:.1f}")
            curr_reward += bonus

        return curr_reward

    def evaluate(self, episodes_per_track=2, max_steps=2000, epoch=0, method="raw_frame"):
        lap_times, ctes, ep_returns = [], [], []
        throttle_history = [] # one list of throttles per eval episode
        successful_laps = 0
        crashed_laps = 0

        subprocess.Popen([self.sim_binary],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        time.sleep(5) # let sim init

        self.env = gym.make("donkey-generated-track-v0", conf=self.conf) # create env
        self.restart_episode()

        for _ in range(episodes_per_track):
            env = self.env
            prev_laps, ep_return, ep_ctes = None, 0.0, []
            ep_throttles = []

            for step in range(1, max_steps + 1):
                state = self.pad_buffers(self.frame_buffer)
                speeds = self.get_speeds()
                steer_mu, _, throttle_mu, _, _ = self._forward(
                    [tf.convert_to_tensor([state], dtype=tf.float32),
                    tf.convert_to_tensor([speeds], dtype=tf.float32)],
                    training=False,
                )
                # steer = float(tf.math.tanh(steer_mu))
                # throttle = float(tf.math.sigmoid(throttle_mu)) # temporary switch to just coasting

                steer = float(steer_mu[0, 0])
                throttle = float(throttle_mu[0, 0])

                obs, _, terminated, truncated, info = env.step([steer, throttle])
                ep_throttles.append(throttle)
                self.frame_buffer.append(self.process_obs(obs))
                self.speed_buffer.append(info["speed"])

                r = self.reward(info, steps=step)
                ep_return += r
                ep_ctes.append(abs(info["cte"]))

                if prev_laps is None:
                    prev_laps = info["lap_count"]
                elif info["lap_count"] > prev_laps:
                    successful_laps += info["lap_count"] - prev_laps # count the lap even if time is missing
                    prev_laps = info["lap_count"]
                    lt = float(info.get("last_lap_time", 0.0))
                    if lt > 0:
                        lap_times.append(lt)

                if terminated:
                    crashed_laps += 1 # the lap in progress failed
                    break
                if truncated:
                    break

            ctes.append(np.mean(ep_ctes))
            ep_returns.append(ep_return)
            throttle_history.append(ep_throttles)
        
        self.env.close()
        subprocess.run(["killall", "-9", "donkey_sim"])
        time.sleep(5)

        attempts = crashed_laps + successful_laps
        out = {
            "crash_rate": crashed_laps / attempts if attempts > 0 else 1.0,  # lower is better
            "lap_time": float(np.mean(lap_times)) if lap_times else 60.0,
            "cte": float(np.mean(ctes)),
            "return": float(np.mean(ep_returns)),
        }

        if out["crash_rate"] <= self.best_eval_stats[0] and out["lap_time"] <= self.best_eval_stats[1]: # check crash rate and lap time
            self.save_this_model = True # save this is the best model so far
            self.best_eval_stats = (out["crash_rate"], out["lap_time"]) # update

        self.plot_throttle(throttle_history, epoch, method)

        return out

    @tf.function # faster
    def _forward(self, inputs, training):
        return self.driver(inputs, training=training)

    def process_obs(self, arr): # override by child
        return

    def generator(self):
        data = np.load(f"{self.data_dir}/merged.npz")
        frames, speeds, actions = data["frames"], data["speeds"], data["actions"]
        for i in range(frames.shape[0]):
            frame = frames[i].astype(np.float32) / 255.0
            yield (frame, speeds[i].astype(np.float32)), actions[i].astype(np.float32)

    def create_datasets(self):

        output_signature=(
            (tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),   # frame
            tf.TensorSpec(shape=(6,), dtype=tf.float32)),              # speed
            tf.TensorSpec(shape=(2,), dtype=tf.float32) # action
        )

        self.dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.generator(),
            output_signature=output_signature,
        )

        self.dataset = (
            self.dataset
            .shuffle(buffer_size=300)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    @tf.function
    def train_step(self, frames, speeds, y):
        with tf.GradientTape() as tape:
            steering_mu, _, throttle_mu, _, _ = self._forward([frames, speeds], training=True) # rest of the outputs are for ppo
            pred = tf.concat([steering_mu, throttle_mu], axis=-1)
            err = tf.square(y - pred)
            loss = tf.reduce_mean(2.0 * err[:, 0] + err[:, 1])

        # apply gradients
        grads = tape.gradient(loss, self.driver.trainable_variables)
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, self.driver.trainable_variables) if g is not None]
        )
        return loss

    def train(self, method="raw_frame"):

        self.create_datasets()

        stats = {"losses": []}

        for epoch in range(self.epochs):
            print(f"epoch {epoch+1}\n")

            # training
            total_loss, samples = 0.0, 0
            for (frames, speeds), y in self.dataset:
                loss = self.train_step(frames, speeds, y)
                n = int(y.shape[0])
                total_loss += float(loss) * n
                samples += n

            train_loss = total_loss / samples
            print(f"loss {train_loss:.5f}")
            stats["losses"].append(train_loss)
            
            # validation (runs on real track to report out stats)
            metrics = self.evaluate(epoch=epoch, method=method)
            self.plot_eval(epoch, metrics, method)
            self.plot_update_stats(stats=stats, epoch=epoch, method=method)

            if self.save_this_model:
                self.driver.save_weights("checkpoints/best_bc_raw_frame.weights.h5", overwrite=True)
                self.save_this_model = False

            self.driver.save_weights("checkpoints/latest_bc_raw_frame.weights.h5", overwrite=True)