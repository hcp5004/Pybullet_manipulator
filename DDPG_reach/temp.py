import gymnasium as gym
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import panda_gym

env = gym.make('PandaReach-v3', render_mode="human")
print("####"+str(env.observation_space))
observation, info = env.reset()
print("####"+str(observation))

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()