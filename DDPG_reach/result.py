import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import datetime
from agent import Agent

env = gym.make('Pendulum-v0',g=9.81)
agent = Agent(env)
agent.actor_model = tf.keras.models.load_model('models/pendulum_actor.h5')
agent.critic_model = tf.keras.models.load_model('models/pendulum_critic.h5')
# Takes about 4 min to train
for ep in range(10):
    # Test Policy
    prev_state = env.reset()
    episodic_reward = 0
    while True:
        env.render("human")
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state, agent.ou_noise, noise=False)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        episodic_reward += reward

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

print("Reward {}".format(episodic_reward))