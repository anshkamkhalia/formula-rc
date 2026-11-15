import os
import sys
import time
import random
import logging
import subprocess
from datetime import datetime
import pygame

os.environ["DONKEY_SIM_PATH"] = "remote"
os.environ["DONKEY_SIM_PORT"] = "9091"
os.environ["DONKEY_SIM_HEADLESS"] = "0"

from gym_donkeycar.envs.donkey_env import (
    GeneratedTrackEnv, MiniMonacoEnv, WarehouseEnv, MountainTrackEnv, WaveshareEnv
)

CHOSEN_MAP = "generated_track"
REGEN_SETTLE = 2.0  # seconds to wait for the sim to rebuild the road
LAUNCH_SIM = True   # False if you start the sim yourself (then its log stays in ~/Library/Logs)

# --- logging setup ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # formula_rc/
LOG_DIR = os.path.join(PROJECT_DIR, "donkeysim_logs")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SIM_BINARY = os.path.join(PROJECT_DIR, "sim/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"teleop_{RUN_ID}.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("teleop")

MAPS = {
    "generated_track": GeneratedTrackEnv,
    "mini_monaco": MiniMonacoEnv,
    "warehouse": WarehouseEnv,
    "mountain": MountainTrackEnv,
    "waveshare": WaveshareEnv,
}

env_class = MAPS.get(CHOSEN_MAP, GeneratedTrackEnv)

current_steering = 0.0
current_throttle = 0.0


def launch_sim():
    sim_log = os.path.join(LOG_DIR, f"sim_{RUN_ID}.log")
    proc = subprocess.Popen(
        [SIM_BINARY, "-logFile", sim_log],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info(f"launched sim, unity log -> {sim_log}")
    return proc


def make_env(retries=30, delay=2.0):
    for attempt in range(retries):
        try:
            return env_class()
        except Exception as e:
            log.info(f"sim not ready ({attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("sim never came up")


def regen_track(env, seed):
    """Same message the training script sends."""
    msg = {
        "msg_type": "regen_road",
        "road_style": "0",
        "rand_seed": str(seed),
        "turn_increment": "0.0",
    }
    handler = env.unwrapped.viewer.handler
    if hasattr(handler, "blocking_send"):
        handler.blocking_send(msg)
    else:
        handler.queue_message(msg)
    time.sleep(REGEN_SETTLE)
    log.info(f"track seed {seed}")
    return env.reset()


def get_input():
    global current_steering, current_throttle
    command = None  # one-shot actions from key presses

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            command = "quit"
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                command = "quit"
            elif event.key == pygame.K_n:
                command = "new"
            elif event.key == pygame.K_r:
                command = "same"
            elif event.key == pygame.K_RIGHTBRACKET:
                command = "next"
            elif event.key == pygame.K_LEFTBRACKET:
                command = "prev"
            elif event.key == pygame.K_SPACE:
                command = "reset"

    keys = pygame.key.get_pressed()

    target_steering = 0.0
    if keys[pygame.K_LEFT]:
        target_steering = -1.0
    elif keys[pygame.K_RIGHT]:
        target_steering = 1.0
    current_steering += (target_steering - current_steering) * 0.12

    target_throttle = 0.0
    if keys[pygame.K_UP]:
        target_throttle = 0.45
    elif keys[pygame.K_DOWN]:
        target_throttle = -0.45
    current_throttle += (target_throttle - current_throttle) * 0.07

    return current_steering, current_throttle, command


def draw_hud(screen, font, seed, info):
    screen.fill((20, 20, 20))
    lines = [
        f"map:   {CHOSEN_MAP}",
        f"seed:  {seed}",
        f"laps:  {info.get('lap_count', 0)}   last: {info.get('last_lap_time', 0.0):.2f}s",
        f"cte:   {info.get('cte', 0.0):+.2f}   speed: {info.get('speed', 0.0):.2f}",
        "",
        "N new  R same  [ ] prev/next",
        "SPACE reset car   ESC quit",
    ]
    for i, text in enumerate(lines):
        screen.blit(font.render(text, True, (220, 220, 220)), (12, 12 + i * 26))
    pygame.display.flip()


def main():
    global current_steering, current_throttle
    generated = CHOSEN_MAP == "generated_track"

    if LAUNCH_SIM:
        launch_sim()

    pygame.init()
    screen = pygame.display.set_mode((360, 220))
    pygame.display.set_caption("teleop")
    font = pygame.font.SysFont("menlo", 16)

    log.info(f"connecting to donkeycar environment using map: [{CHOSEN_MAP}]")
    env = make_env()
    obs, info = env.reset()

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(0, 999_999)
    if generated:
        obs, info = regen_track(env, seed)

    clock = pygame.time.Clock()
    prev_laps = info.get("lap_count", 0)

    while True:
        steering, throttle, command = get_input()

        if command == "quit":
            break

        if command in ("new", "same", "next", "prev") and generated:
            if command == "new":
                seed = random.randint(0, 999_999)
            elif command == "next":
                seed += 1
            elif command == "prev":
                seed = max(0, seed - 1)
            current_steering = current_throttle = 0.0
            obs, info = regen_track(env, seed)
            prev_laps = info.get("lap_count", 0)
            continue

        if command == "reset":
            current_steering = current_throttle = 0.0
            obs, info = env.reset()
            prev_laps = info.get("lap_count", 0)
            continue

        obs, reward, terminated, truncated, info = env.step([steering, throttle])

        if info["lap_count"] > prev_laps:
            log.info(f"lap {info['lap_count']} in {info['last_lap_time']:.2f}s (seed {seed})")
            prev_laps = info["lap_count"]

        if terminated or truncated:
            log.info(f"episode finished (seed {seed}). cte: {info['cte']:.2f}, hit: {info['hit']}")
            current_steering = current_throttle = 0.0
            obs, info = env.reset()
            prev_laps = info.get("lap_count", 0)

        draw_hud(screen, font, seed, info)
        clock.tick(60)

    log.info("quitting")
    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()