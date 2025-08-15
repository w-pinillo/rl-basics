import gymnasium as gym
import numpy as np
import random
import argparse

# Discretize the continuous state space
def discretize_state(state, pos_space, vel_space, ang_space, ang_vel_space):
    pos_bin = np.digitize(state[0], pos_space)
    vel_bin = np.digitize(state[1], vel_space)
    ang_bin = np.digitize(state[2], ang_space)
    ang_vel_bin = np.digitize(state[3], ang_vel_space)
    return pos_bin, vel_bin, ang_bin, ang_vel_bin

def run_cartpole(episodes):
    env = gym.make("CartPole-v1", render_mode="human")

    # Define the state space bins
    pos_space = np.linspace(-4.8, 4.8, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-0.418, 0.418, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    n_states = (len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1)
    n_actions = env.action_space.n

    Q = np.zeros(n_states + (n_actions,))
    alpha = 0.1
    gamma = 0.99
    eps = 0.1

    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, pos_space, vel_space, ang_space, ang_vel_space)
        done = False
        while not done:
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[state])
            (next_obs, reward, terminated, truncated, info) = env.step(a)
            done = terminated or truncated
            next_state = discretize_state(next_obs, pos_space, vel_space, ang_space, ang_vel_space)
            # Q-learning update
            Q[state + (a,)] = Q[state + (a,)] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state + (a,)])
            state = next_state

    # Test greedy policy and show a game
    obs, _ = env.reset()
    state = discretize_state(obs, pos_space, vel_space, ang_space, ang_vel_space)
    print(env.render())
    done = False
    total_reward = 0
    while not done:
        a = np.argmax(Q[state])
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        state = discretize_state(obs, pos_space, vel_space, ang_space, ang_vel_space)
        total_reward += reward
        print(env.render())
    print("Final reward:", total_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CartPole-v1 environment.")
    parser.add_argument("--episodes", type=int, default=500, help="The number of episodes to run.")
    args = parser.parse_args()
    run_cartpole(args.episodes)
