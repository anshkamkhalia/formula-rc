import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import gym_donkeycar
import gymnasium as gym
from src.super.driver import build_rl_driver

class SimEvaluator:

    def __init__(self, input_rep, n_episodes, model, max_cte, timesteps, input_shape):
        self.driver = model
        self.input_rep = input_rep
        self.n_episodes = n_episodes
        self.max_cte = max_cte
        self.timesteps = timesteps
        self.input_shape = input_shape
        self.max_speed = 6
        self.max_steps = 2000

        self.graph_dir = os.path.join("src", "data", "graphs", self.input_rep)
        os.makedirs(self.graph_dir, exist_ok=True)
        self.state_path = os.path.join(self.graph_dir, "eval_state.json")

        # sim config
        self.conf = {
            "exe_path": "remote",
            "port": 9091,
            "cam_resolution": (160, 120, 3),
            "cam_config": {
                "img_w": 160,
                "img_h": 120,
                "img_d": 3,
                "img_enc": "JPG",
                "fov": 63,
            },
            "max_cte": self.max_cte,
            "throttle_min": 0.0,
            "throttle_max": 1.0, 
            "body_style": "f1", 
            "body_rgb": (71, 71, 71),      
            "car_name": "rl_driver",
            "font_size": 50,
        }

    def process_obs(self, obs): # override
        return

    @tf.function
    def _forward(self, frames, speeds):
        steer_mu, _, throttle_mu, _, _ = self.driver([frames, speeds], training=False)
        return steer_mu, throttle_mu

    def _save(self, fig, name):
        for ax in fig.axes:
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.graph_dir, name), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                state = json.load(f)
            print(f"resuming from episode {state['episode']}\n")
            return state["episode"], state["n_crashes"], state["metrics"]
        return 0, 0, {"lap_time": [], "crash_rate": [], "speed": []}

    def save_state(self, episode, n_crashes, metrics):
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"episode": episode, "n_crashes": n_crashes, "metrics": metrics}, f)
        os.replace(tmp_path, self.state_path)

    def plot_metrics_per_episode(self, episode, episode_metrics):
        speeds = np.array(episode_metrics["speed"]) * self.max_speed
        fig, ax = plt.subplots()
        ax.plot(speeds)
        ax.axhline(speeds.mean(), linestyle="--", color="tab:red", label=f"mean {speeds.mean():.2f}")
        ax.set_title(f"Episode {episode} - Speed")
        ax.set_xlabel("step")
        ax.set_ylabel("speed")
        ax.legend()
        self._save(fig, f"episode_{episode}_speed.png")

        lap_times = episode_metrics["lap_time"]
        if lap_times:
            laps = np.arange(1, len(lap_times) + 1)
            fig, ax = plt.subplots()
            ax.plot(laps, lap_times, marker="o")
            ax.axhline(np.mean(lap_times), linestyle="--", color="tab:red", label=f"mean {np.mean(lap_times):.2f}s")
            ax.set_title(f"Episode {episode} - Lap Times")
            ax.set_xlabel("lap")
            ax.set_ylabel("lap time (s)")
            ax.set_xticks(laps)
            ax.legend()
            self._save(fig, f"episode_{episode}_lap_time.png")

    def plot_avg_metrics(self, metrics):
        episodes = np.arange(len(metrics["speed"]))

        avg_speeds = [np.mean(s) * self.max_speed for s in metrics["speed"]]
        fig, ax = plt.subplots()
        ax.bar(episodes, avg_speeds)
        ax.axhline(np.mean(avg_speeds), linestyle="--", color="tab:red", label=f"overall mean {np.mean(avg_speeds):.2f}")
        ax.set_title("Average Speed per Episode")
        ax.set_xlabel("episode")
        ax.set_ylabel("speed")
        ax.set_xticks(episodes)
        ax.legend()
        self._save(fig, "avg_speed.png")

        lap_episodes = [i for i, l in enumerate(metrics["lap_time"]) if l]
        if lap_episodes:
            avg_lap_times = [np.mean(metrics["lap_time"][i]) for i in lap_episodes]
            all_lap_times = [t for i in lap_episodes for t in metrics["lap_time"][i]]
            overall = np.mean(all_lap_times)
            fig, ax = plt.subplots()
            ax.bar([str(i) for i in lap_episodes], avg_lap_times)
            ax.axhline(overall, linestyle="--", color="tab:red", label=f"overall mean {overall:.2f}s ({len(all_lap_times)} laps)")
            ax.set_title("Average Lap Time per Episode")
            ax.set_xlabel("episode")
            ax.set_ylabel("lap time (s)")
            ax.legend()
            self._save(fig, "avg_lap_time.png")

        fig, ax = plt.subplots()
        ax.plot(episodes, metrics["crash_rate"], marker="o")
        ax.set_title(f"Crash Rate (final {metrics['crash_rate'][-1]:.2f})")
        ax.set_xlabel("episode")
        ax.set_ylabel("crash rate")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(episodes)
        self._save(fig, "crash_rate.png")

    def eval(self):

        start_episode, n_crashes, metrics = self.load_state()

        # plot per episode in separate graphs, and average into one graph (for each stat)

        for episode in range(start_episode, self.n_episodes):
            print(f"episode {episode}\n")

            # do this before every run
            self.env = gym.make("donkey-generated-track-v0", conf=self.conf) # create env
            obs, info = self.env.reset()
            obs = self.process_obs(obs)
            speed = abs(info.get("speed", 0.0))
            self.frame_buffer = deque([obs] * self.timesteps, maxlen=self.timesteps)
            self.speed_buffer = deque([speed / self.max_speed] * self.timesteps, maxlen=self.timesteps)

            prev_lap_count = info.get("lap_count", 0)
            crashed = False

            episode_metrics = {
                "lap_time": [],
                "crashed": False,
                "laps_completed": 0,
                "speed": [],
            }

            for step in range(self.max_steps): # until crash

                frame_tensor_input = tf.convert_to_tensor(np.concatenate(self.frame_buffer, axis=-1)[None], dtype=tf.float32)
                speed_tensor_input = tf.convert_to_tensor(np.array(self.speed_buffer)[None], dtype=tf.float32)
                steer, throttle = self._forward(frames=frame_tensor_input, speeds=speed_tensor_input) # returns as a list
                action = [float(steer[0][0]), float(throttle[0][0])]
                obs, _, terminated, truncated, info = self.env.step(action) # action
                obs = self.process_obs(obs) # process new obs
                speed = float(abs(info["speed"])) / self.max_speed # get speed norm
                self.frame_buffer.append(obs) # append
                self.speed_buffer.append(speed) # and append

                lap_count = info["lap_count"]
                if lap_count > prev_lap_count: # lap completed
                    prev_lap_count = lap_count
                    episode_metrics["laps_completed"] += 1
                    lap_time = float(info.get("last_lap_time", 0.0))
                    if lap_time > 0:
                        episode_metrics["lap_time"].append(lap_time)

                episode_metrics["speed"].append(speed)

                if terminated or truncated:
                    crashed = terminated and not truncated
                    cte = info["cte"]
                    print(f"crashed, cte of {cte}")
                    break

            self.env.close()

            episode_metrics["crashed"] = crashed
            if crashed:
                n_crashes += 1

            metrics["lap_time"].append(episode_metrics["lap_time"])
            metrics["crash_rate"].append(n_crashes / (episode + 1))
            metrics["speed"].append(episode_metrics["speed"])

            # self.plot_metrics_per_episode(episode, episode_metrics)
            self.save_state(episode + 1, n_crashes, metrics)

        self.plot_avg_metrics(metrics)

if __name__ == "__main__":

    class Eval(SimEvaluator):
        def __init__(self, input_rep, n_episodes, model, max_cte, timesteps, input_shape):
            super().__init__(input_rep, n_episodes, model, max_cte, timesteps, input_shape)
        def process_obs(self, obs):
            return obs/255.0

    model = build_rl_driver()
    model.load_weights("checkpoints/best_bc_raw_frame.weights.h5")

    evaluate = Eval(
        input_rep="raw_frame",
        n_episodes=20,
        model=model,
        max_cte=4,
        timesteps=6,
        input_shape=(120,160,18),
    )

    evaluate.eval()