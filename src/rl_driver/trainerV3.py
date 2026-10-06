import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym_donkeycar
import gymnasium as gym
from src.rl_driver.models.rl_driverV3 import build_rl_driver
import os
import sys

class FormulaRCEnv:

    def __init__(self):

        self.max_cte = 3.5

        # configs
        self.conf = {
            "exe_path": "remote",
            "port": 9091,
            "cam_resolution": (160, 120, 3),
            "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
            "max_cte": self.max_cte,

            # car appearance
            "body_style": "f1", 
            "body_rgb": (71, 71, 71),      
            "car_name": "rl_driver",
            "font_size": 50,
        }
        
        self.rl_driver = build_rl_driver()
        self.n_timesteps = 6
        self.optimizer = tf.keras.optimizers.Adam(2e-4)

        self.max_speed = 7.0
        self.prev_laps = 0
        self.lap_times = []
        self.best_lap_time = np.inf

        self.REWARD_SCALE = 0.1

        # load model flag
        if sys.argv[1] == "new":
            pass
        else:
            self.rl_driver.load_weights(f"checkpoints/{sys.argv[1]}.weights.h5")

        # steps flag
        print(f"{sys.argv[2]} episodes per epoch\n")
        self.MAX_EP_STEPS = 2000

    def make_std(self, raw):
        return tf.clip_by_value(tf.math.softplus(raw) + 1e-5, 1e-3, 1.0)

    def utils_action(self, buffer):
        pred_buffer = self.pad_frame_buffer(buffer) # pad inputs

        # convert to tensor and action
        state_tensor = tf.convert_to_tensor([pred_buffer], dtype=tf.float32)
        steer_mu, steer_raw_std, throttle_mu, throttle_std_raw, value = self.rl_driver(state_tensor)

        # enforce positive deviations
        steer_std = self.make_std(steer_raw_std)
        throttle_std = self.make_std(throttle_std_raw)

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

    def action(self):
        pred_buffer = self.pad_frame_buffer(self.frame_buffer) # pad inputs

        # convert to tensor and action
        state_tensor = tf.convert_to_tensor([pred_buffer], dtype=tf.float32)
        steer_mu, steer_raw_std, throttle_mu, throttle_std_raw, value = self.rl_driver(state_tensor)

        # enforce positive deviations
        steer_std = self.make_std(steer_raw_std)
        throttle_std = self.make_std(throttle_std_raw)

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

    def pad_frame_buffer(self, buffer):
        if len(buffer) < self.n_timesteps:
            pad = [np.zeros((120, 160, 3), dtype=np.float32)
                for _ in range(self.n_timesteps - len(buffer))]
            buffer = pad + list(buffer)
        return np.concatenate(buffer, axis=-1)

    def reward(self, info, steps):
        max_cte = self.max_cte
        cte = abs(info["cte"])
        speed = float(np.clip(info["speed"], 0.0, self.max_speed))
        laps = info["lap_count"]

        # new episode: sync to whatever the sim reports (prevents a phantom lap bonus)
        if steps <= 1:
            self.prev_laps = laps

        # terminal penalties
        if cte > max_cte or info["hit"] != "none":
            return -10.0

        speed_norm = speed / self.max_speed
        centering = max(0.0, 1.0 - (cte / max_cte) ** 2)

        curr_reward = 1.5 * speed_norm * centering
        curr_reward -= 0.5 * (cte / max_cte) ** 2

        if speed < 1.0:
            curr_reward -= 1.0 * (1.0 - speed)

        # lap reward, paid once when lap_count increments
        if laps > self.prev_laps:
            self.prev_laps = laps

            lap_time = float(info.get("last_lap_time", 0.0))
            if lap_time <= 0.0: # sim hasn't updated it yet, treat as a slow lap
                lap_time = 30.0
            else:
                self.lap_times.append(lap_time)

            target = 10.2
            bonus = float(np.clip(10.0 + 15.0 * (target - lap_time), 2.0, 25.0))

            print(f"log: lap {laps} in {lap_time:.2f}s, bonus {bonus:.1f}")
            curr_reward += bonus

        return curr_reward

    def run_critic_on_truncated(self):
        pred_buffer = self.pad_frame_buffer(self.frame_buffer) # pad inputs
        
        # convert to tensor and action
        state_tensor = tf.convert_to_tensor([pred_buffer], dtype=tf.float32)
        _, _, _, _, value = self.rl_driver(state_tensor)

        return float(value[0][0])

    def train(self, epochs=100, episodes=15, gamma=0.99, lam=0.95, clip_eps=0.2, ppo_epochs=4, minibatch_size=128, max_grad_norm=0.5):

        self.frame_buffer = [] # stores last n frames

        episode_reward = 0.0 # running total for the current episode

        for epoch in range(epochs):

            observations = []
            rewards = []
            done_vals = []
            logp_vals = []
            raw_steerings = []
            raw_throttles = []
            values = []

            finished_episodes = []
            finished_lengths = []
            finished_crashed = []
            episodes_completed = 0
            episode_steps = 0

            self.lap_times = []

            # create env every epoch
            self.env = gym.make("donkey-generated-track-v0", conf=self.conf)
            obs, info = self.env.reset()
            self.frame_buffer.append(np.array(obs/255.0, dtype=np.float32))

            while episodes_completed < episodes:

                # step loop, collect data
                steering, throttle, raw_steering, raw_throttle, total_logp, value, inputs = self.action() # get action
                obs, _, terminated, truncated, info = self.env.step([steering, throttle]) # custom reward
                episode_steps += 1

                self.frame_buffer.append(obs / 255.0)

                if len(self.frame_buffer) > 6:
                    self.frame_buffer = self.frame_buffer[-6:] # take 6 most recent elements
                step_reward = self.reward(info, steps=episode_steps) # calculate custom reward
                episode_reward += step_reward

                if episode_steps >= self.MAX_EP_STEPS and not terminated:
                    truncated = True

                # data collection for rollout buffer
                observations.append(np.round(inputs * 255).astype(np.uint8))
                rewards.append(step_reward * self.REWARD_SCALE)
                done_vals.append(terminated or truncated)
                if terminated or truncated:
                    finished_episodes.append(episode_reward) # total return, not per-step
                    finished_lengths.append(episode_steps)
                    finished_crashed.append(bool(terminated))
                    episode_reward = 0.0

                logp_vals.append(total_logp)
                raw_throttles.append(raw_throttle)
                raw_steerings.append(raw_steering)
                values.append(value)

                if terminated:
                    obs, info = self.env.reset()
                    episode_steps = 0
                    self.frame_buffer = []
                    self.frame_buffer.append(np.array(obs/255.0, dtype=np.float32))
                    episodes_completed += 1

                elif truncated:
                    last_value = gamma * self.run_critic_on_truncated()
                    last_stored_reward = rewards[-1]
                    rewards[-1] = last_stored_reward + last_value
                    obs, info = self.env.reset()
                    episode_steps = 0
                    self.frame_buffer = []
                    self.frame_buffer.append(np.array(obs/255.0, dtype=np.float32))
                    episodes_completed += 1

            end_of_rollout = self.run_critic_on_truncated()

            obs, info = self.env.reset()
            episode_steps = 0
            self.frame_buffer = []
            self.frame_buffer.append(np.array(obs/255.0, dtype=np.float32))

            self.env.close() # close while updating

            # gae

            # convert to arrays instead of python lists
            rewards_arr = np.array(rewards, dtype=np.float32)
            values_arr = np.array(values, dtype=np.float32)
            dones_arr = np.array(done_vals, dtype=np.float32)

            advantages = np.zeros_like(rewards_arr)
            last_adv = 0.0

            for t in reversed(range(len(rewards_arr))):
                if t == len(rewards_arr) - 1:
                    next_value = end_of_rollout
                else:
                    next_value = values_arr[t + 1]

                non_terminal = 1.0 - dones_arr[t]
                delta = rewards_arr[t] + gamma * next_value * non_terminal - values_arr[t]
                last_adv = delta + gamma * lam * non_terminal * last_adv
                advantages[t] = last_adv

            returns = advantages + values_arr # do not normalize critic targets
            expl_var = 1.0 - np.var(returns - values_arr) / (np.var(returns) + 1e-8)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # convert data to tensors
            returns = tf.convert_to_tensor(returns, dtype=tf.float32)
            advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
            old_logp = tf.convert_to_tensor(logp_vals, dtype=tf.float32)
            raw_steerings = tf.convert_to_tensor(raw_steerings, dtype=tf.float32)
            raw_throttles = tf.convert_to_tensor(raw_throttles, dtype=tf.float32)
            observations = np.stack(observations)

            n = observations.shape[0]
            stats = {"kl": [], "clip": [], "policy": [], "value": [], "entropy": [], "grad_norm": []}

            for ppo_epoch in range(ppo_epochs):
                indices = tf.random.shuffle(tf.range(n))
                kl = 0
                n_kl = 0

                for start in range(0, n, minibatch_size):
                    idx = indices[start:start + minibatch_size]

                    # gather the minibatch
                    mb_obs = tf.convert_to_tensor(observations[idx.numpy()], dtype=tf.float32) / 255.0
                    mb_raw_steer = tf.gather(raw_steerings, idx)
                    mb_raw_throttle = tf.gather(raw_throttles, idx)
                    mb_old_logp = tf.gather(old_logp, idx)
                    mb_adv = tf.gather(advantages, idx)
                    mb_returns = tf.gather(returns, idx)

                    # gradient tape
                    with tf.GradientTape() as tape:
                        
                        # forward pass
                        steer_mu, steer_std_raw, throttle_mu, throttle_std_raw, new_value = self.rl_driver(mb_obs)

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
                        squashed_throttle = tf.math.sigmoid(mb_raw_throttle) # sigmoid for 0,1

                        # calculate log probabilities
                        log_prob_steer = steer_dist.log_prob(mb_raw_steer) - tf.math.log(1.0 - squashed_steer ** 2 + 1e-6)
                        log_prob_throttle = throttle_dist.log_prob(mb_raw_throttle) - tf.math.log(squashed_throttle * (1.0 - squashed_throttle) + 1e-6)
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
                    mean_ep_reward = np.mean(finished_episodes) if finished_episodes else float("nan")

                    # regression test: first minibatch of first pass should have ratio close to 1
                    if ppo_epoch == 0 and start == 0:
                        print("first minibatch ratio min/mean/max:",
                            float(tf.reduce_min(ratio)), float(tf.reduce_mean(ratio)), float(tf.reduce_max(ratio)))

                # safety valve
                if (kl / n_kl) > 0.05:
                    print(f"stopping, approx kl is: {kl/n_kl}")
                    break

            mean_ep_reward = np.mean(finished_episodes) if finished_episodes else float("nan")
            mean_len = np.mean(finished_lengths) if finished_lengths else float("nan")
            crash_rate = np.mean(finished_crashed) if finished_crashed else float("nan")

            print(
                f"epoch {epoch} | episodes {len(finished_episodes)} | mean return {mean_ep_reward:.1f} | "
                f"mean len {mean_len:.0f} | crash rate {crash_rate:.2f} | "
                f"updates {len(stats['kl'])} | kl {np.mean(stats['kl']):.4f} | clipfrac {np.mean(stats['clip']):.3f} | "
                f"policy {np.mean(stats['policy']):.4f} | value {np.mean(stats['value']):.3f} | "
                f"entropy {np.mean(stats['entropy']):.3f} | gradnorm {np.mean(stats['grad_norm']):.3f} | "
                f"expl var {expl_var:.2f}"
            )

            os.makedirs("checkpoints", exist_ok=True)

            if self.lap_times:
                epoch_lap_time = float(np.mean(self.lap_times))
                print(f"laps this epoch: {len(self.lap_times)} | avg lap time {epoch_lap_time:.2f}s | best {self.best_lap_time:.2f}s")
                if epoch_lap_time < self.best_lap_time:
                    self.best_lap_time = epoch_lap_time
                    self.rl_driver.save_weights("checkpoints/best_lap.weights.h5")
                    print(f"saved new best-lap model ({epoch_lap_time:.2f}s)")

            if epoch % 5 == 0:
                self.rl_driver.save_weights("checkpoints/latest.weights.h5")

if __name__ == "__main__":
    trainer = FormulaRCEnv().train(epochs=10_000, episodes=int(sys.argv[2]))