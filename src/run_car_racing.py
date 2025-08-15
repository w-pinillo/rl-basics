import gymnasium as gym
import numpy as np
import random
import argparse

def run_car_racing(episodes):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)

    # Simple random agent
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()  # Take a random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CarRacing-v3 environment.")
    parser.add_argument("--episodes", type=int, default=5, help="The number of episodes to run.")
    args = parser.parse_args()
    run_car_racing(args.episodes)
