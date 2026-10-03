import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym_donkeycar
import gymnasium as gym
from src.rl_driver.models.rl_driverV1 import build_rl_driver
import os
import sys

class FormulaRCEnv:

    def __init__(self):

        # configs
        self.conf = {
            "exe_path": "remote",
            "port": 9091,
            "cam_resolution": (160, 120, 3),
            "cam_config": {"img_w": 160, "img_h": 120, "img_d": 3},
            "max_cte": 2.5,
        }

        # create env
        self.env = gym.make("donkey-generated-track-v0", conf=self.conf)
        self.rl_driver = build_rl_driver()
        self.n_timesteps = 6
        self.optimizer = tf.keras.optimizers.Adam(3e-4)
        self.best_reward = -np.inf # track best reward instead of best loss

        # load model flag
        if sys.argv[1] == "new":
            pass
        else:
            self.rl_driver.load_weights(f"checkpoints/{sys.argv[1]}.weights.h5")

        # steps flag
        print(f"{sys.argv[2]} steps per epoch\n")

    def action(self):
        pred_buffer = self.pad_frame_buffer(self.frame_buffer) # pad inputs

        # convert to tensor and action
        state_tensor = tf.convert_to_tensor([pred_buffer], dtype=tf.float32)
        steer_mu, steer_raw_std, throttle_mu, throttle_std_raw = self.rl_driver(state_tensor)

         # enforce positive deviations
        steer_std = tf.math.softplus(steer_raw_std) + 1e-5
        throttle_std = tf.math.softplus(throttle_std_raw) + 1e-5

        # create base normal distributions
        steer_dist = tfp.distributions.Normal(loc=steer_mu, scale=steer_std)
        throttle_dist = tfp.distributions.Normal(loc=throttle_mu, scale=throttle_std)

        # sample raw actions
        raw_steering = steer_dist.sample()
        raw_throttle = throttle_dist.sample()
        
        # squash boundaries with tanh and sigmoid
        steering = tf.math.tanh(raw_steering) # steering is between -1, 1
        throttle = tf.math.sigmoid(raw_throttle) # throttle between 0, 1

        # calcualte log probabilities
        log_prob_steer = steer_dist.log_prob(raw_steering) - tf.math.log(1.0 - steering ** 2 + 1e-6)
        log_prob_throttle = throttle_dist.log_prob(raw_throttle) - tf.math.log(throttle * (1.0 - throttle) + 1e-6)

        # total log probability
        total_log_prob = log_prob_steer + log_prob_throttle

        # convert value from tensors to standard scalars
        return float(steering[0][0]), float(throttle[0][0]), total_log_prob[0][0], pred_buffer

    def pad_frame_buffer(self, buffer):
        if len(buffer) < self.n_timesteps:
            pad = [np.zeros((120, 160, 3), dtype=np.float32)
                for _ in range(self.n_timesteps - len(buffer))]
            buffer = pad + list(buffer)
        return np.concatenate(buffer, axis=-1)

    def reward(self, info, max_cte=2.5):
        curr_reward = 0

        # cross track error
        cte = info["cte"]
        if abs(cte) > max_cte:
            curr_reward -= 3.0 # hard penalty for being off the track
        std_dev = 0.4
        curr_reward += np.exp(-0.5 * (cte / std_dev) ** 2)
        
        # collisions 
        if info["hit"] != "none":
            curr_reward -= 3.0 # hard penalty for collisions

        return curr_reward

    def train(self, epochs=100, steps=1000, gamma=0.99):

        obs, info = self.env.reset()
        self.frame_buffer = [] # stores last n frames

        for epoch in range(epochs):

            observations = []
            actions = []
            rewards = []
            next_states = []
            terminated_vals = []
            truncated_vals = []
            logp_vals = []

            for step in range(steps):

                # step loop, collect data
                steering, throttle, logp, inputs = self.action() # get action
                obs, _, terminated, truncated, info = self.env.step([steering, throttle]) # custom reward
                self.frame_buffer.append(obs / 255.0)
                if len(self.frame_buffer) > 6:
                    self.frame_buffer = self.frame_buffer[-6:] # take 6 most recent elements
                step_reward = self.reward(info) # calculate custom reward

                # data collection
                observations.append(inputs)
                actions.append([steering, throttle])
                rewards.append(step_reward)
                next_states.append(np.array(self.pad_frame_buffer(self.frame_buffer)))
                terminated_vals.append(terminated)
                truncated_vals.append(truncated)
                logp_vals.append(logp)

                if terminated or truncated:
                    obs, info = self.env.reset()
                    self.frame_buffer = []

            # update weights
            discounted_returns = []
            G = 0.0 # total cumulative future reward

            for r, term, trunc in zip(reversed(rewards), reversed(terminated_vals), reversed(truncated_vals)):
                if term: # if episode ended at this step we reset returns to 0
                    G = 0.0
                G = r + gamma * G # return for current step is the step reward plus discounted future rewards
                discounted_returns.insert(0, G) # insert at the front of the list to restore order

            # convert to tensors
            states_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
            logp_tensor = tf.convert_to_tensor(logp_vals, dtype=tf.float32)
            returns_tensor = tf.convert_to_tensor(discounted_returns, dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)

            returns_mean = tf.math.reduce_mean(returns_tensor)
            returns_std = tf.math.reduce_std(returns_tensor) + 1e-8
            normalized_returns = (returns_tensor - returns_mean) / returns_std

            # gradient tape
            with tf.GradientTape() as tape:

                # forward pass
                steer_mu, steer_raw_std, throttle_mu, throttle_raw_std = self.rl_driver(states_tensor)

                # std adjustments
                steer_std = tf.math.softplus(steer_raw_std) + 1e-5
                throttle_std = tf.math.softplus(throttle_raw_std) + 1e-5

                # reconstruct distributions
                steer_dist = tfp.distributions.Normal(loc=steer_mu, scale=steer_std)
                throttle_dist = tfp.distributions.Normal(loc=throttle_mu, scale=throttle_std)

                # unsquash for log probs
                steer_actions = tf.clip_by_value(actions_tensor[:, 0:1], -0.999, 0.999)
                throttle_actions = tf.clip_by_value(actions_tensor[:, 1:2], 0.001, 0.999)

                raw_steer_logged = tf.math.atanh(steer_actions)
                raw_throttle_logged = tf.math.log(throttle_actions / (1.0 - throttle_actions))

                # recalculate current log probs linked to tape context
                log_p_steer = steer_dist.log_prob(raw_steer_logged) - tf.math.log(1.0 - steer_actions ** 2 + 1e-6)
                log_p_throttle = throttle_dist.log_prob(raw_throttle_logged) - tf.math.log(throttle_actions * (1.0 - throttle_actions) + 1e-6)

                # squeeze to match vector length
                current_log_probs = tf.squeeze(log_p_steer + log_p_throttle, axis=-1)

                # minimize negative loss to maximize rewards
                self.loss = -tf.math.reduce_mean(current_log_probs * normalized_returns)

            # extract gradients
            gradients = tape.gradient(self.loss, self.rl_driver.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.rl_driver.trainable_variables))
            epoch_reward = np.sum(rewards)

            if epoch_reward >= self.best_reward:
                os.makedirs("checkpoints", exist_ok=True)
                self.rl_driver.save_weights("checkpoints/best.weights.h5")
                self.best_reward = epoch_reward
                print(f"saved new best model\n\n")

        self.env.close()

trainer = FormulaRCEnv().train(epochs=10_000, steps=int(sys.argv[2]))