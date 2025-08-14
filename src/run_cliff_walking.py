import gymnasium as gym
import numpy as np
import random
import pandas as pd

env = gym.make("CliffWalking-v0", render_mode="human")
n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))
alpha = 0.8
gamma = 0.95
eps = 0.1
episodes = 2000

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

# Show final Q table
actions_labels = ["Up", "Right", "Down", "Left"]
states_labels = [f"S{i}" for i in range(env.observation_space.n)]
Q_df = pd.DataFrame(Q, index=states_labels, columns=actions_labels)
print("\nFinal Q-table learned:\n")
print(Q_df)

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
