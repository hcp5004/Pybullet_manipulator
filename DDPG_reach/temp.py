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
                   render=True,
                   binary_reward=True,
                   max_episode_steps=5,
                   image_observation=True,
                   depth_image=False,
                   goal_image=True,
                   visualize_target=True,
                   camera_setup=camera_setup,
                   observation_cam_id=[0],
                   goal_cam_id=1,
                   )

# obs = env.reset()
# while True:
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     print('state: ', obs['policy_state'], '\n',
#           'desired_goal: ', obs['desired_goal'], '\n',
#           'achieved_goal: ', obs['achieved_goal'], '\n',
#           'reward: ', reward, '\n')
#     plt.pause(0.00001)
#     if done:
#         env.reset()

print("#################" + str(env.action_space))
print("#################" + str(env.action_space.high))
print("#################" + str(env.action_space.low))