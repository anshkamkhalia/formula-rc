import tensorflow as tf
import numpy as np
import cv2 as cv
import gymnasium as gym
import os
import subprocess
import time
import matplotlib
matplotlib.use("Agg") # no gui
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from collections import deque
from src.super.critics import build_critic
import gym_donkeycar

class RlDriverSACEnv:

    def __init__(self, model, tracks, max_cte=5.0, max_speed=7.0, n_timesteps=None, n_latents=None, latent_dim=64, img_size=(84, 128, 3), replay_buff_size=1_000_000):

        self.uses_latents = None
        if n_timesteps:
            self.uses_latents = False
            self.n_timesteps = n_timesteps
            self.img_size = img_size
        elif n_latents:
            self.uses_latents = True
            self.n_latents = n_latents
            self.latent_dim = latent_dim
        else:
            raise ValueError

        if self.uses_latents:
            self.state_shape = (self.n_latents, self.latent_dim)
            self.state = np.zeros(self.state_shape) # (timesteps, latent dim)
        else:
            self.state_shape = (self.n_timesteps, self.img_size[0], self.img_size[1], self.img_size[2])
            self.state = np.zeros(self.state_shape) # (timesteps, height, width, channels)

        self.replay_buff_size = replay_buff_size # this is super important, tune accordingly
        
        self.max_cte = max_cte
        self.tracks = tracks
        self.max_speed = max_speed
        self.track = None

        self.REWARD_SCALE = 0.1
        self.MAX_EP_STEPS = 2000
        self.prev_laps = 0
        self.lap_times = []
        self.obs_buffer = deque(maxlen=50) # needed if evaluate() is called before train()

        # save best eval model
        # basically we want the most consistent model so we are going to save the one with the lowest crash rate
        self.best_eval_key = (1.0, float("inf"))

        # sim connection state
        self.env = None
        self.current_track = None

        # graphs for convergence
        self.log_dir = "training_logs/sac"
        os.makedirs(self.log_dir, exist_ok=True)
        self.eval_history = []

        # time scale at 2.0 with 30fps
        self.conf = {
            "exe_path": "remote",
            "port": 9091,
            "cam_resolution": (160, 120, 3),
            "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
            "max_cte": self.max_cte,
            "throttle_min": -1.0,
            "throttle_max": 1.0,

            # car appearance
            "body_style": "f1", 
            "body_rgb": (71, 71, 71),      
            "car_name": "rl_driver",
            "font_size": 50,
        }

        self.driver = model
        self.driver_optimizer = tf.keras.optimizers.Adam(3e-4)

        self.q1 = build_critic(self.state_shape, self.uses_latents)
        self.q2 = build_critic(self.state_shape, self.uses_latents)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.target_entropy = -2.0

        self.q1_target = build_critic(self.state_shape, self.uses_latents)
        self.q2_target = build_critic(self.state_shape, self.uses_latents)
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())
        self.q1_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.q2_optimizer = tf.keras.optimizers.Adam(3e-4)

        self.sim_binary = os.path.abspath("sim/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim") # donkeysim

        # open sim at startup
        subprocess.Popen(
            [self.sim_binary],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

    def process_state(self): # new state based on new observation
        pass # overridden by child

    def model_paths(self, method, tag):
        d = f"checkpoints/sac/{method}"
        return {
            "driver": f"{d}/{tag}_{method}_sac.weights.h5",
            "q1": f"{d}/{tag}_{method}_sac_q1.weights.h5",
            "q2": f"{d}/{tag}_{method}_sac_q2.weights.h5",
            "q1_target": f"{d}/{tag}_{method}_sac_q1_target.weights.h5",
            "q2_target": f"{d}/{tag}_{method}_sac_q2_target.weights.h5",
        }

    def save_models(self, method, tag):
        os.makedirs(f"checkpoints/sac/{method}", exist_ok=True)
        for name, path in self.model_paths(method, tag).items():
            getattr(self, name).save_weights(path)

    def load_models(self, method, tag="latest"):
        paths = self.model_paths(method, tag)
        if not os.path.exists(paths["driver"]):
            print(f"no {tag} checkpoint for {method}, starting fresh")
            return

        self.driver.load_weights(paths["driver"])
        for name in ("q1", "q2"):
            if os.path.exists(paths[name]):
                getattr(self, name).load_weights(paths[name])
            else:
                print(f"warning: {name} missing, critic starts fresh")

        for targ, src in (("q1_target", "q1"), ("q2_target", "q2")):
            if os.path.exists(paths[targ]):
                getattr(self, targ).load_weights(paths[targ])
            else:
                getattr(self, targ).set_weights(getattr(self, src).get_weights())

        print(f"loaded {tag} models for {method}")

    def get_action(self, states):
        steer_mu, steer_std_raw, thr_mu, thr_std_raw, _ = self.driver(states, training=True)
        steer_std = self.make_std(steer_std_raw)
        thr_std = self.make_std(thr_std_raw)

        raw_s = steer_mu + steer_std * tf.random.normal(tf.shape(steer_mu))
        raw_t = thr_mu + thr_std * tf.random.normal(tf.shape(thr_mu))
        steer = tf.math.tanh(raw_s)
        thr = tf.math.tanh(raw_t)

        logp_s = tfp.distributions.Normal(steer_mu, steer_std).log_prob(raw_s) - tf.math.log(1.0 - steer ** 2 + 1e-6)
        logp_t = tfp.distributions.Normal(thr_mu, thr_std).log_prob(raw_t) - tf.math.log(1.0 - thr ** 2 + 1e-6)

        actions = tf.concat([tf.reshape(steer, [-1, 1]), tf.reshape(thr, [-1, 1])], axis=1)
        logp = tf.reshape(logp_s, [-1]) + tf.reshape(logp_t, [-1])
        return actions, logp

    def pad_obs_buffer(self):
        if self.uses_latents:
            while len(self.obs_buffer) < self.n_latents:
                self.obs_buffer.append(self.obs_buffer[0])

        else:
            while len(self.obs_buffer) < self.n_timesteps:
                self.obs_buffer.append(self.obs_buffer[0])

    def make_std(self, raw):
        return tf.clip_by_value(tf.math.softplus(raw) + 1e-5, 1e-3, 1.0)

    def get_reward(self, info, steps):
        max_cte = self.max_cte
        cte = abs(info["cte"])
        speed = float(np.clip(info["speed"], -self.max_speed, self.max_speed))
        if abs(speed) <= 0.005:
            speed = 0.0
        laps = info["lap_count"]

        # new episode: sync to whatever the sim reports (prevents a phantom lap bonus)
        if steps <= 1:
            self.prev_laps = laps

        # terminal penalties
        if cte > max_cte or info["hit"] != "none":
            return -20.0

        if speed < 0.0:
            return -8.0 - 15.0 * (abs(speed) / self.max_speed)

        speed_norm = speed / self.max_speed
        centering = max(0.0, 1.0 - (cte / max_cte) ** 2)

        curr_reward = 1.5 * speed_norm * centering
        curr_reward -= 0.5 * (cte / max_cte) ** 2

        if speed < 1.0:
            curr_reward -= 1.0 * (1.0 - speed)

        lap_time = float(info.get("last_lap_time", 0.0))

        # lap reward, paid once when lap_count increments
        if laps > self.prev_laps:
            self.prev_laps = laps

            if lap_time <= 0.0: # sim hasn't updated it yet, treat as a slow lap
                lap_time = 30.0
            else:
                self.lap_times.append(lap_time)

            target = 10.2
            bonus = float(np.clip(10.0 + 15.0 * (target - lap_time), 2.0, 15.0))

            print(f"log: lap {laps} in {lap_time:.2f}s, bonus {bonus:.1f}")
            curr_reward += bonus

        return curr_reward

    @staticmethod
    def apply(opt, grads, variables):
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None] # value head gets no grads
        opt.apply_gradients(pairs)

    def start_sim(self, track=None):
        if track is None:
            track = np.random.choice(self.tracks)

        if self.env is not None and track == self.current_track:
            obs, info = self.env.reset() # same track: just reset the car
        else:
            self.close_sim() # different track: back to menu, load new scene
            self.env = self.make_env(track)
            self.current_track = track
            obs, info = self.env.reset()

        self.track = track
        self.obs_buffer.clear()
        self.obs_buffer.append(obs)
        self.pad_obs_buffer()
        self.state = np.asarray(self.process_state(), dtype=np.float32)
        return obs, info

    def evaluate(self, episodes_per_track=2, max_steps=2000):
        lap_times, ctes, ep_returns = [], [], []
        successful_laps = 0
        crashed_laps = 0

        for _ in range(episodes_per_track):
            self.start_sim(track="donkey-minimonaco-track")
            prev_laps, ep_return, ep_ctes = None, 0.0, []

            for step in range(1, max_steps + 1):
                steer_mu, _, thr_mu, _, _ = self.driver(
                    tf.convert_to_tensor(self.state[None], dtype=tf.float32), training=False)
                steering = float(tf.math.tanh(steer_mu)[0, 0]) # deterministic action
                throttle = float(tf.math.tanh(thr_mu)[0, 0])

                obs, _, terminated, truncated, info = self.env.step([steering, throttle])
                self.obs_buffer.append(obs)
                self.state = np.asarray(self.process_state(), dtype=np.float32)

                ep_return += self.get_reward(info, step)
                ep_ctes.append(abs(info["cte"]))

                if prev_laps is None:
                    prev_laps = info["lap_count"]
                elif info["lap_count"] > prev_laps:
                    successful_laps += info["lap_count"] - prev_laps
                    prev_laps = info["lap_count"]
                    lt = float(info.get("last_lap_time", 0.0))
                    if lt > 0:
                        lap_times.append(lt)

                if terminated:
                    crashed_laps += 1
                    break
                if truncated:
                    break

            ctes.append(np.mean(ep_ctes))
            ep_returns.append(ep_return)

        self.close_sim()
        attempts = crashed_laps + successful_laps
        out = {
            "crash_rate": crashed_laps / attempts if attempts > 0 else 1.0,  # lower is better
            "lap_time": float(np.mean(lap_times)) if lap_times else 60.0,
            "cte": float(np.mean(ctes)),
            "return": float(np.mean(ep_returns)),
        }

        return out

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
            ax.set_title(f"SAC eval {label} ({method})")
            ax.grid(alpha=0.3)
            if key == "crash_rate":
                ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, f"{method}_sac_{key}.png"), dpi=120)
            plt.close(fig)

        columns = ["epoch"] + [key for key, _ in panels]
        np.savetxt(
            os.path.join(self.log_dir, f"{method}_sac_eval.csv"),
            np.array([[h[c] for c in columns] for h in self.eval_history], dtype=float),
            delimiter=",", header=",".join(columns), comments="",
        )

    def close_sim(self):
        if self.env is None:
            return
        try:
            self.env.unwrapped.viewer.exit_scene() # sim returns to main menu
            time.sleep(1.5)
        except Exception as e:
            print(f"exit_scene failed: {e}")
        try:
            self.env.close()
        except Exception as e:
            print(f"env.close failed: {e}")
        self.env = None
        self.current_track = None
        time.sleep(1.5) # let the sim settle before the next connect

    def make_env(self, track, retries=30, delay=2.0):
        # retry until the sim is accepting connections
        for attempt in range(retries):
            try:
                return gym.make(track, conf=self.conf)
            except Exception as e:
                print(f"sim not ready ({attempt + 1}/{retries}): {e}")
                time.sleep(delay)
        raise RuntimeError("sim never came up")

    def reset_episode(self):
        self.start_sim()

    @tf.function
    def update(self, s, a, r, s2, d):
        alpha = tf.exp(self.log_alpha)

        # critic target
        a2, logp2 = self.get_action(s2)
        q_targ = tf.minimum(self.q1_target([s2, a2]), self.q2_target([s2, a2]))[:, 0]
        y = tf.stop_gradient(r + self.gamma * (1.0 - d) * (q_targ - alpha * logp2))

        # critic update
        with tf.GradientTape(persistent=True) as tape:

            q1 = self.q1([s, a])[:, 0]
            q2 = self.q2([s, a])[:, 0]
            q1_loss = 0.5 * tf.reduce_mean((q1 - y) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q2 - y) ** 2)

        self.apply(self.q1_optimizer, tape.gradient(q1_loss, self.q1.trainable_variables), self.q1.trainable_variables)
        self.apply(self.q2_optimizer, tape.gradient(q2_loss, self.q2.trainable_variables), self.q2.trainable_variables)
        del tape # clear tape before second use

        # actor update
        with tf.GradientTape() as tape:

            a_new, logp = self.get_action(s)
            q_new = tf.minimum(self.q1([s, a_new]), self.q2([s, a_new]))[:, 0]
            actor_loss = tf.reduce_mean(alpha * logp - q_new)

        self.apply(self.driver_optimizer, tape.gradient(actor_loss, self.driver.trainable_variables), self.driver.trainable_variables)

        # temperature update
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(logp + self.target_entropy))

        self.apply(self.alpha_optimizer, tape.gradient(alpha_loss, [self.log_alpha]), [self.log_alpha])

        # polyak averaging of target critics
        for net, targ in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            for v, tv in zip(net.trainable_variables, targ.trainable_variables):
                tv.assign(self.tau * v + (1.0 - self.tau) * tv)

        return q1_loss, q2_loss, actor_loss, alpha

    def train(self, epochs=100, method="raw_frame", steps=5_000, batch_size=256,
              gamma=0.99, tau=0.005, warmup_steps=5_000, updates_per_epoch=None):

        self.obs_buffer = deque(maxlen=50) # holds all observations, capped at length 50
        self.replay_buffer = deque(maxlen=self.replay_buff_size) # very important
        self.lap_times = [] # tracks lap times
        updates_per_epoch = updates_per_epoch or steps # utd ratio of 1 by default
        self.tau = tau
        self.gamma = gamma

        total_steps = 0
        for epoch in range(epochs + 1):
            print(f"epoch: {epoch}\n")
            
            self.start_sim()
            self.lap_times = []
            ep_rewards, ep_lengths, ep_crashed = [], [], []
            track_steps = steps // len(self.tracks) # equal share per track, total steps (and utd) unchanged

            # collect experience
            for track in self.tracks:
                self.start_sim(track=track) # loads track, resets car, obs buffer and state
                ep_steps, ep_reward = 0, 0.0

                for step in range(track_steps):
                    if total_steps < warmup_steps: # random exploration to seed the buffer
                        steering = float(np.random.uniform(-1.0, 1.0))
                        throttle = float(np.random.uniform(-1.0, 1.0))
                    else:
                        a, _ = self.get_action(tf.convert_to_tensor(self.state[None], dtype=tf.float32))
                        steering, throttle = (float(x) for x in a[0].numpy())

                    obs, _, terminated, truncated, info = self.env.step([steering, throttle])
                    ep_steps += 1
                    total_steps += 1
                    reward = self.get_reward(info, ep_steps)
                    ep_reward += reward

                    if ep_steps % 100 == 0:
                        cv.imwrite("training_logs/obs.jpeg", obs)

                    if ep_steps >= self.MAX_EP_STEPS and not terminated:
                        truncated = True

                    self.obs_buffer.append(obs)
                    next_state = np.asarray(self.process_state(), dtype=np.float32)
                    done = float(terminated) # only true terminals cut bootstrapping, truncation doesn't

                    self.replay_buffer.append((self.state.copy(), np.array([steering, throttle], dtype=np.float32),
                                               np.float32(reward * self.REWARD_SCALE), next_state.copy(), np.float32(done)))
                    self.state = next_state

                    if terminated or truncated:
                        ep_rewards.append(ep_reward)
                        ep_lengths.append(ep_steps)
                        ep_crashed.append(bool(terminated))
                        ep_steps, ep_reward = 0, 0.0
                        self.start_sim(track=track) # restart on the same track, not a random one

            # minibatching and gradient taping
            self.close_sim() # exit scene and disconnect while updating
            losses = []
            for _ in range(updates_per_epoch):

                idx = np.random.randint(len(self.replay_buffer), size=batch_size)
                batch = [self.replay_buffer[i] for i in idx]
                s, a, r, s2, d = (tf.convert_to_tensor(np.stack(x), dtype=tf.float32) for x in zip(*batch))
                losses.append([float(v) for v in self.update(s, a, r, s2, d)])

            q1_l, q2_l, pi_l, alpha = np.mean(losses, axis=0)

            mean_ep = np.mean(ep_rewards) if ep_rewards else float("nan")
            mean_len = np.mean(ep_lengths) if ep_lengths else float("nan")
            crash_rate = np.mean(ep_crashed) if ep_crashed else float("nan")

            print(f"epoch {epoch} | episodes {len(ep_rewards)} | mean return {mean_ep:.1f} | "
                  f"mean len {mean_len:.0f} | crash rate {crash_rate:.2f} | q1 {q1_l:.3f} | q2 {q2_l:.3f} | "
                  f"actor {pi_l:.3f} | alpha {alpha:.4f} | buffer {len(self.replay_buffer)}")

            if self.lap_times:
                print(f"laps this epoch: {len(self.lap_times)} | avg lap time {np.mean(self.lap_times):.2f}s")

            self.save_models(method, "latest")

            metrics = self.evaluate()
            eval_key = (metrics["crash_rate"], metrics["lap_time"])
            if eval_key <= self.best_eval_key:
                self.best_eval_key = eval_key
                self.save_models(method, "best")
                print(f"new best model: crash rate {eval_key[0]:.2f}, lap time {eval_key[1]:.2f}s")

            self.plot_eval(epoch, metrics, method)
            self.lap_times = [] # keep eval laps out of training stats
            print(f"eval epoch {epoch} | {metrics}")

            # reset simulator after every epoch to prevent freezing
            subprocess.run(["killall", "-9", "donkey_sim"])
            if (epoch + 1) % 7 == 0:
                print("\nwaiting longer\n")
                time.sleep(300)

            print("\nwaiting\n")
            time.sleep(60)
            subprocess.Popen([self.sim_binary], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)