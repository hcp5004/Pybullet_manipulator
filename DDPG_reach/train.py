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

camera_setup = [
    {
        'cameraEyePosition': [-0.9, -0.0, 0.4],
        'cameraTargetPosition': [-0.45, -0.0, 0.0],
        'cameraUpVector': [0, 0, 1],
        'render_width': 224,
        'render_height': 224
    },
    {
        'cameraEyePosition': [-1.0, -0.25, 0.6],
        'cameraTargetPosition': [-0.6, -0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 224,
        'render_height': 224
    },
]

env = pmg.make_env(task='reach',
                   gripper='parallel_jaw',
                   render=False,
                   binary_reward=True,
                   max_episode_steps=50,
                   image_observation=True,
                   depth_image=False,
                   goal_image=True,
                   visualize_target=True,
                   camera_setup=camera_setup,
                   observation_cam_id=[0],
                   goal_cam_id=1,
                   )

agent = Agent(env)

for ep in range(agent.total_episodes):
    prev_state = env.reset()
    prev_state = prev_state['policy_state']
    while True:

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state, agent.ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        agent.buffer.record((prev_state, action, reward, state['policy_state']))
        #episodic_reward += reward

        agent.buffer.learn(ep, done)
        agent.update_target(agent.target_actor.variables, agent.actor_model.variables, agent.tau)
        agent.update_target(agent.target_critic.variables, agent.critic_model.variables, agent.tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state['policy_state']

    # Test Policy
    prev_state = env.reset()
    prev_state = prev_state['policy_state']
    episodic_reward = 0

    while True:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state, agent.ou_noise, noise=False)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        episodic_reward += reward

        # End this episode when `done` is True
        if done:
            break

        prev_state = state['policy_state']

    agent.ep_reward_list.append(episodic_reward)

    with agent.summary_writer.as_default():
        tf.summary.scalar('mean_reward', episodic_reward, step=ep)

agent.actor_model.save("models/KukaReach_actor.h5")
agent.critic_model.save("models/KukaReach_critic.h5")