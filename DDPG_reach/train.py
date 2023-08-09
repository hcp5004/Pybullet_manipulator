import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt

import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import datetime
from agent import Agent
import time
import panda_gym

env = gym.make('PandaReach-v3', render_mode="human")
agent = Agent(env)

for ep in range(agent.total_episodes):
    prev_state, info = env.reset()
    prev_state = prev_state['observation']
    while True:

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state, agent.ou_noise)
        # Recieve state and reward from environment.
        state, reward, terminated, truncated, info = env.step(action)

        agent.buffer.record((prev_state, action, reward, state['observation']))
        #episodic_reward += reward

        agent.buffer.learn(ep, terminated or truncated)
        agent.update_target(agent.target_actor.variables, agent.actor_model.variables, agent.tau)
        agent.update_target(agent.target_critic.variables, agent.critic_model.variables, agent.tau)

        # End this episode when `done` is True
        if terminated or truncated:
            break

        prev_state = state['observation']

    # Test Policy
    prev_state, info = env.reset()
    prev_state = prev_state['observation']
    episodic_reward = 0

    while True:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state, agent.ou_noise, noise=False)
        # Recieve state and reward from environment.
        state, reward, terminated, truncated, info = env.step(action)

        episodic_reward += reward

        # End this episode when `done` is True
        if terminated or truncated:
            break

        prev_state = state['observation']

    agent.ep_reward_list.append(episodic_reward)

    with agent.summary_writer.as_default():
        tf.summary.scalar('mean_reward', episodic_reward, step=ep)

agent.actor_model.save("models/KukaReach_actor.h5")
agent.critic_model.save("models/KukaReach_critic.h5")