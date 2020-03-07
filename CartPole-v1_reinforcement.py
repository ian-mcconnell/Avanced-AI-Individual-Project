import gym
import random
import numpy as np

env_name = "CartPole-v1"
env = gym.make(env_name)

class CartPoleAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.build_model()

    def build_model(self):
        self.weights = 1e-4*np.random.rand(*self.state_dim, self.action_size)
        self.best_reward = -np.Inf
        self.best_weights = np.copy(self.weights)
        self.noise_scale = 1e-2

    def get_action(self, state):
        p = np.dot(state, self.weights)
        action = np.argmax(p)
        return action

    def update_model(self, reward):
        if reward >= self.best_reward:
            self.best_reward = reward
            self.best_weights = np.copy(self.weights)
            self.noise_scale = max(self.noise_scale * 2, 2)
        else:
            self.noise_scale = min(self.noise_scale * 2, 2)

        self.weights = self.best_weights + self.noise_scale * np.random.rand(*self.state_dim, self.action_size)


agent = CartPoleAgent(env)
num_episodes = 100

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()
        total_reward += reward

    agent.update_model(total_reward)
    print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))
