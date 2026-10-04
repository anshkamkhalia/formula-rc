import os
import sys
import pygame

os.environ["DONKEY_SIM_PATH"] = "remote"
os.environ["DONKEY_SIM_PORT"] = "9091"
os.environ["DONKEY_SIM_HEADLESS"] = "0"

from gym_donkeycar.envs.donkey_env import (
    GeneratedTrackEnv, MiniMonacoEnv, WarehouseEnv, MountainTrackEnv, WaveshareEnv
)

CHOSEN_MAP = "generated_track" 

MAPS = {
    "generated_track": GeneratedTrackEnv,
    "mini_monaco": MiniMonacoEnv,
    "warehouse": WarehouseEnv,
    "mountain": MountainTrackEnv,
    "waveshare": WaveshareEnv,
}

if CHOSEN_MAP in MAPS:
    env_class = MAPS[CHOSEN_MAP]
else:
    env_class = GeneratedTrackEnv

pygame.init()
screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption("teleop")

current_steering = 0.0
current_throttle = 0.0

def get_smoothed_input():
    global current_steering, current_throttle
    quit_teleop = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_teleop = True

    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_ESCAPE]:
        quit_teleop = True

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
        
    return current_steering, current_throttle, quit_teleop

def main():
    print(f"connecting to donkeycar environment using map: [{CHOSEN_MAP}]")
    env = env_class()
    
    obs, info = env.reset()

    clock = pygame.time.Clock()
    running = True

    while running:
        steering, throttle, quit_teleop = get_smoothed_input()
        
        if quit_teleop:
            running = False
            break
        action = [steering, throttle]

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished. Resetting environment...")
            obs, info = env.reset()

        clock.tick(60)

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
