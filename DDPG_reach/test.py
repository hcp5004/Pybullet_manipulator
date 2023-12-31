import pybullet_multigoal_gym as pmg
import matplotlib.pyplot as plt

import gym
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

env = gym.make('Pendulum-v0',g=9.81)
agent = Agent(env)

for ep in range(agent.total_episodes):
    agent.write=True
    prev_state= env.reset()
    prev_state = prev_state
    while True:

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = [agent.policy(tf_prev_state, agent.ou_noise)]
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(agent.upper_bound * action)
        agent.buffer.record((prev_state, action, reward, state))
        #episodic_reward += reward

        critic_loss, actor_loss = agent.buffer.learn()
        agent.update_target(agent.target_actor.variables, agent.actor_model.variables, agent.tau)
        agent.update_target(agent.target_critic.variables, agent.critic_model.variables, agent.tau)

        # End this episode when `done` is True
        if done:
            agent.buffer.WriteBoard(critic_loss, actor_loss, ep)
            break

        prev_state = state

    #Test Policy

    if ep%10 == 0:
        ep_reward_list = []
        agent.write=False
        for ep_test in range(5):
            prev_state = env.reset()
            prev_state = prev_state
            episodic_reward = 0
            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = [agent.policy(tf_prev_state, agent.ou_noise, noisy=False)]
                # Recieve state and reward from environment.
                state, reward, done, info = env.step(agent.upper_bound * action)

                episodic_reward += reward

                # End this episode when `done` is True
                if done:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

        avg_reward = np.mean(ep_reward_list[-5:])
        with agent.summary_writer.as_default():
            tf.summary.scalar('mean_reward', avg_reward, step=ep)

agent.actor_model.save("models/pendulum_actor.h5")
agent.critic_model.save("models/pendulum_critic.h5")