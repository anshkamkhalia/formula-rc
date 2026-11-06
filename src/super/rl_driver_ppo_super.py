import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym_donkeycar
import gymnasium as gym
import os
import time
import matplotlib
matplotlib.use("Agg") # no gui
import matplotlib.pyplot as plt
import subprocess
import cv2 as cv
from collections import deque

class RlDriverPPOEnv:

    def __init__(self, model, tracks, max_cte=5.0, timesteps=6, max_speed=7.0, track_steps=1_000, img_shape=(120, 160, 3), max_ep_steps=2_000, randomize_generated=True, regen_every="block", regen_settle=2.0, track_seed=None):

        self.max_cte = max_cte
        self.tracks = tracks
        self.track_steps = track_steps
        self.img_shape = img_shape

        # configs
        self.conf = {
            "exe_path": "remote",
            "port": 9091,
            "cam_resolution": (160, 120, 3),
            "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
            "max_cte": self.max_cte,
            "throttle_min": 0.0, # removing braking temporarily
            "throttle_max": 1.0, 

            # car appearance
            "body_style": "f1", 
            "body_rgb": (71, 71, 71),      
            "car_name": "rl_driver",
            "font_size": 50,
        }

        # save best eval model
        # basically we want the most consistent model so we are going to save the one with the lowest crash rate
        self.best_eval_stats = (1.0, np.inf) # crash rate, lap time
        self.save_this_model = False
        
        self.rl_driver = model
        self.n_timesteps = timesteps
        self.optimizer = tf.keras.optimizers.Adam(2e-4)

        self.max_speed = max_speed
        self.prev_laps = 0
        self.lap_times = []

        # stall handling
        self.slow_speed = 0.3 # one threshold for the penalty and the stall timeout
        self.stall_limit = 40 # consecutive slow steps before a stall ends the episode
        self.stall_grace = 60 # steps at episode start where stalls aren't checked (launch)
        self.full_brake_speed = 2.0 # speed at or above which full braking is allowed
        self.stall_steps = 0

        self.max_ep_steps = max_ep_steps

        self.frame_buffer = deque(maxlen=self.n_timesteps) # this used to be part of the old code but i dont wanna change it everywhere so now the individual track observation buffer becomes this
        self.speed_buffer = deque(maxlen=self.n_timesteps) # new

        self.REWARD_SCALE = 0.1

        # sim binary
        self.sim_binary = os.path.abspath("sim/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim")

        subprocess.Popen([self.sim_binary],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # graphs for convergence
        self.log_dir = "training_logs/ppo"
        os.makedirs(self.log_dir, exist_ok=True)
        self.eval_history = [] # one dict per eval: epoch + evaluate() output
        
        # sim connection state
        self.env = None
        self.current_track = None

        self.GENERATED_TRACK = "donkey-generated-track-v0"
        self.randomize_generated = randomize_generated
        self.regen_every = regen_every
        self.regen_settle = regen_settle
        self.track_rng = np.random.default_rng(track_seed)
        self.current_layout_seed = None

    def get_speeds(self):
        while len(self.speed_buffer) < self.n_timesteps:
            self.speed_buffer.appendleft(0.0)  # not moving
        return np.array(self.speed_buffer, dtype=np.float32) / self.max_speed  # signed, roughly -1..1
    
    def regen_track(self, seed=None):
        if seed is None:
            seed = int(self.track_rng.integers(0, 1_000_000))
        msg = {
            "msg_type": "regen_road",
            "road_style": "0",
            "rand_seed": str(seed),
            "turn_increment": "0.0",
        }
        handler = self.env.unwrapped.viewer.handler
        if hasattr(handler, "blocking_send"):
            handler.blocking_send(msg)
        else:
            handler.queue_message(msg)

        time.sleep(self.regen_settle)
        self.current_layout_seed = seed
        print(f"log: generated track layout seed {seed}")
        return self.env.reset()

    def start_sim(self, track=None, new_layout=None):
        if track is None:
            track = np.random.choice(self.tracks)

        reloaded = False
        if self.env is not None and track == self.current_track:
            # same track: just reset the car, no scene reload
            obs, info = self.env.reset()
        else:
            # different track (or nothing loaded): back to menu, load new scene
            self.close_sim()
            self.env = self.make_env(track)
            self.current_track = track
            obs, info = self.env.reset()
            reloaded = True

        if track == self.GENERATED_TRACK and self.randomize_generated:
            if new_layout is None:
                new_layout = reloaded or self.regen_every == "episode"
            if new_layout:
                obs, info = self.regen_track()

        self.frame_buffer.clear()
        self.speed_buffer.clear()
        self.stall_steps = 0
        self.frame_buffer.append(self.process_obs(obs))
        self.speed_buffer.append(info["speed"])
        return obs, info

    def evaluate(self, episodes_per_track=2, max_steps=2000, epoch=0, method="raw_frame"):
        lap_times, ctes, ep_returns = [], [], []
        throttle_history = [] # one list of throttles per eval episode
        successful_laps = 0
        crashed_laps = 0

        for _ in range(episodes_per_track):
            self.frame_buffer.clear()
            self.speed_buffer.clear()
            obs, info = self.start_sim(track="donkey-minimonaco-track-v0") # use eval track
            env = self.env
            prev_laps, ep_return, ep_ctes = None, 0.0, []
            ep_throttles = []

            for step in range(1, max_steps + 1):
                state = self.pad_buffers(self.frame_buffer)
                speeds = self.get_speeds()
                steer_mu, _, throttle_mu, _, _ = self.rl_driver(
                    [tf.convert_to_tensor([state], dtype=tf.float32),
                    tf.convert_to_tensor([speeds], dtype=tf.float32)],
                    training=False,
                )
                steer = float(tf.math.tanh(steer_mu))
                throttle = float(tf.math.sigmoid(throttle_mu)) # temporary switch to just coasting

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
                    break # unfinished lap, not a crash (excluded)

            ctes.append(np.mean(ep_ctes))
            ep_returns.append(ep_return)
            throttle_history.append(ep_throttles)

        self.close_sim()
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

    def restart_episode(self, track):

        self.start_sim(track=track)

    def process_obs(self, obs): # override this for child class

        return obs

    def make_std(self, raw):

        return tf.clip_by_value(tf.math.softplus(raw) + 1e-5, 1e-3, 1.0)

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

        columns = ["epoch"] + [key for key, _ in panels]
        np.savetxt(
            os.path.join(self.log_dir, f"{method}_ppo_eval.csv"),
            np.array([[h[c] for c in columns] for h in self.eval_history], dtype=float),
            delimiter=",", header=",".join(columns), comments="",
        )

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

    def action(self):

        pred_buffer = self.pad_buffers(self.frame_buffer) # pad inputs
        speeds = self.get_speeds()

        # convert to tensor and action
        state_tensor = tf.convert_to_tensor([pred_buffer], dtype=tf.float32)
        speed_tensor = tf.convert_to_tensor([speeds])
        steer_mu, steer_raw_std, throttle_mu, throttle_std_raw, value = self.rl_driver(
            [state_tensor, speed_tensor], training=False
        )

        # enforce positive deviations
        steer_std = self.make_std(steer_raw_std)
        throttle_std = self.make_std(throttle_std_raw)

        # create base normal distributions
        steer_dist = tfp.distributions.Normal(loc=steer_mu, scale=steer_std)
        throttle_dist = tfp.distributions.Normal(loc=throttle_mu, scale=throttle_std)

        # sample raw actions
        raw_steer = steer_dist.sample()
        raw_throttle = throttle_dist.sample()
        
        # squash boundaries with tanh or sigmoid
        steer = tf.math.tanh(raw_steer) # steering is between -1, 1
        throttle = tf.math.sigmoid(raw_throttle) # temporary switch to coasting

        # log probs
        log_prob_steer = steer_dist.log_prob(raw_steer) - tf.math.log(1.0 - steer ** 2 + 1e-6)
        log_prob_throttle = (throttle_dist.log_prob(raw_throttle)
                        + tf.math.softplus(raw_throttle) + tf.math.softplus(-raw_throttle))
        total_log_prob = log_prob_steer + log_prob_throttle

        return (float(steer), float(throttle), float(raw_steer), float(raw_throttle), # remove remap
                float(total_log_prob), float(value), pred_buffer, speeds)

    def pad_buffers(self, buffer):
        if len(buffer) < self.n_timesteps:
            pad = [np.zeros(self.img_shape, dtype=np.uint8)
                for _ in range(self.n_timesteps - len(buffer))]
            buffer = pad + list(buffer)
        return np.concatenate(buffer, axis=-1)
    
    def reward(self, info, steps, train=False, epoch=None):
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
        if cte > max_cte or info["hit"] != "none":
            return -15.0 # was -30.0

        speed_norm = speed / self.max_speed
        cte_frac = cte / max_cte
        centering = max(0.0, 1.0 - cte_frac)

        curr_reward = 2.0 * speed_norm * centering
        curr_reward -= 0.5 * cte_frac ** 2

        if speed < self.slow_speed:
            curr_reward -= 0.3

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

    def run_critic_on_truncated(self):
        pred_buffer = self.pad_buffers(self.frame_buffer)
        state_tensor = tf.convert_to_tensor([pred_buffer], dtype=tf.float32)
        speed_tensor = tf.convert_to_tensor([self.get_speeds()], dtype=tf.float32)
        _, _, _, _, value = self.rl_driver([state_tensor, speed_tensor], training=False)
        return float(value[0][0])

    def train(self, epochs=100, episodes=15, gamma=0.99, lam=0.95, clip_eps=0.2, ppo_epochs=4, minibatch_size=128, max_grad_norm=0.5, method="raw_frame", mode="steps"):

        # new fixed step track switching
        for epoch in range(epochs+1):
            print(f"epoch: {epoch}")
            self.lap_times = []  # training laps for this epoch only

            finished_lengths = []
            finished_crashed = []

            main_buffer = { # start each as empty lists and then access by name later
                "observations": [],
                "speeds": [], # new: tracks speeds for model inputs
                "raw_steer": [],
                "raw_throttle": [],
                "logp_vals": [],
                "advantages": [],
                "returns": [],
            }

            # ok so basically what happens is each track computes the stuff needed for the main buffer and then its cleared and added to that main buffer to be used in training
            # gae is computer per track to make sure that advantages dont leak between tracks

            for track in self.tracks:
                track_buffer = { # start each as empty lists and then access by name later
                    "observations": [],
                    "raw_steer": [],
                    "speeds": [], # same over here
                    "raw_throttle": [],
                    "logp_vals": [],
                    "advantages": [],
                    "returns": [],

                    "rewards": [], # not part of main rollout buffer, for gae
                    "dones": [], # not part of main rollout buffer, for gae
                    "values": [], # not part of main rollout buffer, for gae
                }

                self.frame_buffer = deque(maxlen=self.n_timesteps) # this is what is fed into our policy, capped at length of self.n_timesteps
                self.speed_buffer = deque(maxlen=self.n_timesteps)
                obs, info = self.start_sim(track=track) # use track instead of randomization (im not changing the function)

                episode_steps = 0
                prev_throttle = 0.0 # for throttle change penalty

                for step in range(self.track_steps):
                    steer, throttle, raw_steer, raw_throttle, logp, value, inputs, speeds = self.action()
                    obs, _, terminated, truncated, info = self.env.step([steer, throttle])
                    obs = self.process_obs(obs)
                    self.frame_buffer.append(obs)
                    self.speed_buffer.append(info["speed"]) # keep speeds in sync with frames
                    episode_steps += 1
                    new_reward = self.reward(info, episode_steps)

                    prev_throttle = throttle
                    truncated = (truncated or episode_steps >= self.max_ep_steps) and not terminated

                    track_buffer["observations"].append(inputs)
                    track_buffer["speeds"].append(speeds) # the speeds the policy actually saw
                    track_buffer["raw_steer"].append(raw_steer)
                    track_buffer["raw_throttle"].append(raw_throttle)
                    track_buffer["logp_vals"].append(logp)
                    track_buffer["rewards"].append(new_reward * self.REWARD_SCALE)
                    track_buffer["values"].append(value)
                    track_buffer["dones"].append(terminated or truncated)

                    # no advantages or returns yet, computed during gae

                    # decide truncation first, then bootstrap, log, restart
                    if terminated or truncated:
                        if truncated:
                            track_buffer["rewards"][-1] += gamma * self.run_critic_on_truncated()

                        finished_lengths.append(episode_steps)
                        finished_crashed.append(bool(terminated))
                        print(f"{track}: cte {info['cte']:.2f}, len {episode_steps}")

                        self.restart_episode(track=track)
                        episode_steps = 0
                        prev_throttle = 0.0

                end_of_rollout = self.run_critic_on_truncated()
                self.close_sim()

                # gae
                rewards = np.array(track_buffer["rewards"], dtype=np.float32)
                values = np.array(track_buffer["values"], dtype=np.float32)
                dones = np.array(track_buffer["dones"], dtype=np.float32)

                advantages = np.zeros_like(rewards)
                last_adv = 0.0

                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_value = end_of_rollout
                        next_non_terminal = 1.0 - dones[t] 
                    else:
                        next_value = values[t + 1]
                        next_non_terminal = 1.0 - dones[t]

                    delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                    
                    last_adv = delta + gamma * lam * next_non_terminal * last_adv
                    advantages[t] = last_adv

                returns = advantages + values # do not normalize critic targets
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)            
                track_buffer["advantages"] = advantages
                track_buffer["returns"] = returns

                # add track buffer to main buffer
                main_buffer["observations"].extend(track_buffer["observations"])
                main_buffer["speeds"].extend(track_buffer["speeds"]) # add speeds
                main_buffer["raw_steer"].extend(track_buffer["raw_steer"])
                main_buffer["raw_throttle"].extend(track_buffer["raw_throttle"])
                main_buffer["logp_vals"].extend(track_buffer["logp_vals"])
                main_buffer["advantages"].extend(track_buffer["advantages"])
                main_buffer["returns"].extend(track_buffer["returns"])

            main_buffer["observations"] = np.array(main_buffer["observations"])
            main_buffer["speeds"] = np.array(main_buffer["speeds"], dtype=np.float32)
            for k in ["raw_steer", "raw_throttle", "logp_vals", "advantages", "returns"]:
                main_buffer[k] = tf.constant(main_buffer[k], dtype=tf.float32)

            n = main_buffer["observations"].shape[0]
            stats = {"kl": [], "clip": [], "policy": [], "value": [], "entropy": [], "grad_norm": []}

            for ppo_epoch in range(ppo_epochs):
                indices = tf.random.shuffle(tf.range(n))
                kl = 0
                n_kl = 0

                for start in range(0, n, minibatch_size):
                    idx = indices[start:start + minibatch_size]

                    # gather the minibatch
                    mb_obs = tf.convert_to_tensor((main_buffer["observations"][idx.numpy()]), dtype=tf.float32)
                    mb_speeds = tf.convert_to_tensor((main_buffer["speeds"][idx.numpy()]), dtype=tf.float32)
                    mb_raw_steer = tf.gather(main_buffer["raw_steer"], idx)
                    mb_raw_throttle = tf.gather(main_buffer["raw_throttle"], idx)
                    mb_old_logp = tf.gather(main_buffer["logp_vals"], idx)
                    mb_adv = tf.gather(main_buffer["advantages"], idx)
                    mb_returns = tf.gather(main_buffer["returns"], idx)

                    # gradient tape
                    with tf.GradientTape() as tape:
                        
                        # forward pass
                        steer_mu, steer_std_raw, throttle_mu, throttle_std_raw, new_value = self.rl_driver(
                            [mb_obs, mb_speeds], training=True
                        )

                        # squeeze everything
                        steer_mu = tf.squeeze(steer_mu, axis=-1)
                        steer_std_raw = tf.squeeze(steer_std_raw, axis=-1)
                        throttle_mu = tf.squeeze(throttle_mu, axis=-1)
                        throttle_std_raw = tf.squeeze(throttle_std_raw, axis=-1)
                        new_value = tf.squeeze(new_value, axis=-1)

                        # softplus
                        steer_std = self.make_std(steer_std_raw)
                        throttle_std = self.make_std(throttle_std_raw)

                        # normal distributions
                        steer_dist = tfp.distributions.Normal(loc=steer_mu, scale=steer_std)
                        throttle_dist = tfp.distributions.Normal(loc=throttle_mu, scale=throttle_std)

                        # squash stored raw actions
                        squashed_steer = tf.math.tanh(mb_raw_steer) # tanh for -1,1
                        squashed_throttle = tf.math.sigmoid(mb_raw_throttle) # temporary switch

                        # calculate log probabilities
                        log_prob_steer = steer_dist.log_prob(mb_raw_steer) - tf.math.log(1.0 - squashed_steer ** 2 + 1e-6)
                        log_prob_throttle = (throttle_dist.log_prob(mb_raw_throttle)
                        + tf.math.softplus(mb_raw_throttle) + tf.math.softplus(-mb_raw_throttle))
                        new_logp = log_prob_steer + log_prob_throttle # combine to get total

                        # policy loss
                        ratio = tf.exp(new_logp - mb_old_logp)
                        unclipped = ratio * mb_adv
                        clipped = tf.clip_by_value(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                        policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

                        # value loss against unormalized returns with huber
                        value_loss = tf.reduce_mean(
                            tf.keras.losses.huber(mb_returns[:, None], new_value[:, None], delta=1.0)
                        )

                        # entropy bonus
                        entropy = tf.reduce_mean(steer_dist.entropy() + throttle_dist.entropy())

                        # total
                        self.loss = policy_loss + 0.25 * value_loss - 0.001 * entropy
            
                    # update vars
                    variables = self.rl_driver.trainable_variables
                    grads = tape.gradient(self.loss, variables)
                    grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                    self.optimizer.apply_gradients(zip(grads, variables))

                    approx_kl = float(tf.reduce_mean(mb_old_logp - new_logp))
                    kl += approx_kl
                    n_kl += 1
                    clip_frac = float(tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > clip_eps, tf.float32)))

                    stats["kl"].append(approx_kl)
                    stats["clip"].append(clip_frac)
                    stats["policy"].append(float(policy_loss))
                    stats["value"].append(float(value_loss))
                    stats["entropy"].append(float(entropy))
                    stats["grad_norm"].append(float(grad_norm))

                    # regression test: first minibatch of first pass should have ratio close to 1
                    if ppo_epoch == 0 and start == 0:
                        print("first minibatch ratio min/mean/max:",
                            float(tf.reduce_min(ratio)), float(tf.reduce_mean(ratio)), float(tf.reduce_max(ratio)))

                # safety valve
                if (kl / n_kl) > 0.05:
                    print(f"stopping, approx kl is: {kl/n_kl}")
                    break

            mean_len = np.mean(finished_lengths) if finished_lengths else float("nan")
            crash_rate = np.mean(finished_crashed) if finished_crashed else float("nan")

            print(
                f"epoch {epoch} "
                f"mean len {mean_len:.0f} | crash rate {crash_rate:.2f} | "
                f"updates {len(stats['kl'])} | kl {np.mean(stats['kl']):.4f} |"
                f"policy {np.mean(stats['policy']):.4f} | value {np.mean(stats['value']):.3f} | "
                f"entropy {np.mean(stats['entropy']):.3f} | gradnorm {np.mean(stats['grad_norm']):.3f}"
            )

            os.makedirs("checkpoints", exist_ok=True)

            if self.lap_times:
                epoch_lap_time = float(np.mean(self.lap_times))
                print(f"laps this epoch: {len(self.lap_times)} | avg lap time {epoch_lap_time:.2f}s")

            self.rl_driver.save_weights(f"checkpoints/latest_{method}_ppo.weights.h5")

            metrics = self.evaluate(epoch=epoch, method=method)
            if self.save_this_model:
                self.rl_driver.save_weights(f"checkpoints/best_{method}_ppo.weights.h5")
                self.save_this_model = False
            self.plot_eval(epoch, metrics, method)

            print(f"eval epoch {epoch} | {metrics}")

            # reset simulator after every epoch to prevent freezing
            subprocess.run(["killall", "-9", "donkey_sim"])

            if (epoch+1) % 7 == 0:
                print("\nwaiting longer\n")
                time.sleep(300)
            print("\nwaiting\n")
            time.sleep(60) # wait

            subprocess.Popen([self.sim_binary],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )