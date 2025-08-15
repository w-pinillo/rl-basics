import gymnasium as gym
import numpy as np
import random
import argparse

# Discretize the continuous state space
def discretize_state(state, pos_space, vel_space):
    pos_bin = np.digitize(state[0], pos_space)
    vel_bin = np.digitize(state[1], vel_space)
    return pos_bin, vel_bin

def run_mountain_car(episodes):
    env = gym.make("MountainCar-v0", render_mode="human")

    # Define the state space bins
    pos_space = np.linspace(-1.2, 0.6, 20)
    vel_space = np.linspace(-0.07, 0.07, 20)

    n_states = (len(pos_space)+1, len(vel_space)+1)
    n_actions = env.action_space.n

    Q = np.zeros(n_states + (n_actions,))
    alpha = 0.1
    gamma = 0.99
    eps = 0.1

    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, pos_space, vel_space)
        done = False
        while not done:
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[state])
            (next_obs, reward, terminated, truncated, info) = env.step(a)
            done = terminated or truncated
            next_state = discretize_state(next_obs, pos_space, vel_space)
            # Q-learning update
            Q[state + (a,)] = Q[state + (a,)] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state + (a,)])
            state = next_state

    # Test greedy policy and show a game
    obs, _ = env.reset()
    state = discretize_state(obs, pos_space, vel_space)
    print(env.render())
    done = False
    total_reward = 0
    while not done:
        a = np.argmax(Q[state])
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        state = discretize_state(obs, pos_space, vel_space)
        total_reward += reward
        print(env.render())
    print("Final reward:", total_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MountainCar-v0 environment.")
    parser.add_argument("--episodes", type=int, default=5000, help="The number of episodes to run.")
    args = parser.parse_args()
    run_mountain_car(args.episodes)
