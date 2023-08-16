import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import datetime
from agent import Agent
import time
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import panda_gym

env = gym.make('PandaReachDense-v3',render_mode="rgb_array")
agent = Agent(env, write=False)

agent.actor_model = tf.keras.models.load_model('models/KukaReach_actor.h5')
agent.critic_model = tf.keras.models.load_model('models/KukaReach_critic.h5')
# Takes about 4 min to train
image = []
for ep in range(3):
    # Test Policy
    prev_state, info = env.reset()
    image.append(env.render())
    prev_state = np.concatenate((prev_state['observation'], prev_state['desired_goal']))
    episodic_reward = 0
    time.sleep(1)
    while True:

        time.sleep(0.1)
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state, agent.ou_noise, noisy=False)
        # Recieve state and reward from environment.
        state, reward, terminated, truncated, info = env.step(action)

        state = np.concatenate((state['observation'], state['desired_goal']))

        episodic_reward += reward

        image.append(env.render())

        # End this episode when `done` is True
        if truncated:
            break

        prev_state = state

print("Reward {}".format(episodic_reward))

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

save_frames_as_gif(image)