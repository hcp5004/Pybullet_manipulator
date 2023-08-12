import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import datetime
from agent import Agent
import time
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import panda_gym

env = gym.make('PandaReach-v3',render_mode="human")
agent = Agent(env, write=False)

agent.actor_model = tf.keras.models.load_model('models/KukaReach_actor.h5')
agent.critic_model = tf.keras.models.load_model('models/KukaReach_critic.h5')
# Takes about 4 min to train
for ep in range(10):
    # Test Policy
    prev_state, info = env.reset()
    prev_state = np.concatenate((prev_state['observation'], prev_state['desired_goal']))
    episodic_reward = 0
    while True:

        time.sleep(0.5)
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state, agent.ou_noise, noisy=False)
        # Recieve state and reward from environment.
        state, reward, terminated, truncated, info = env.step(action)

        state = np.concatenate((state['observation'], state['desired_goal']))

        episodic_reward += reward

        # End this episode when `done` is True
        if terminated or truncated:
            break

        prev_state = state

print("Reward {}".format(episodic_reward))