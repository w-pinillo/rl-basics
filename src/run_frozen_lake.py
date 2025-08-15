import gymnasium as gym
import numpy as np
import random
import pandas as pd
import argparse

def run_frozen_lake(episodes):
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    alpha = 0.2
    gamma = 0.95
    eps = 0.4

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[obs])
            (next_obs, reward, terminated, truncated, info) = env.step(a)
            done = terminated or truncated
            # Q-learning update
            Q[obs, a] = Q[obs, a] + alpha * (reward + gamma * np.max(Q[next_obs]) - Q[obs, a])
            obs = next_obs

    # Test greedy policy and show a game
    obs, _ = env.reset()
    print(env.render())  # prints initial board
    done = False
    while not done:
        a = np.argmax(Q[obs])
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        print(env.render())
    print("Final reward:", reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FrozenLake-v1 environment.")
    parser.add_argument("--episodes", type=int, default=100000, help="The number of episodes to run.")
    args = parser.parse_args()
    run_frozen_lake(args.episodes)
