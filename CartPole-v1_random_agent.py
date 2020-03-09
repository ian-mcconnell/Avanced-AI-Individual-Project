import gym
import random

env_name = "CartPole-v1"
env = gym.make(env_name)

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size: ", self.action_size)

    def get_action(self):
        action = random.choice(range(self.action_size))
        return action


agent = Agent(env)
num_episodes = 100

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action()
        state, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
    print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))
