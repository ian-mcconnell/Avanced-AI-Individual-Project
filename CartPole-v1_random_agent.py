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
state = env.reset()

for _ in range(200):
    action = agent.get_action()
    env.step(action)
    env.render()
