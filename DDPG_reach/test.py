import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import datetime
from agent import Agent

if __name__ == "__main__":
    env = gym.make('Pendulum-v0',g=9.81)
    agent = Agent(env)
    # Takes about 4 min to train
    for ep in range(agent.total_episodes):
        prev_state = env.reset()
        #episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = [agent.policy(tf_prev_state, agent.ou_noise)]
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            agent.buffer.record((prev_state, action, reward, state))
            #episodic_reward += reward

            agent.buffer.learn(ep, done)
            agent.update_target(agent.target_actor.variables, agent.actor_model.variables, agent.tau)
            agent.update_target(agent.target_critic.variables, agent.critic_model.variables, agent.tau)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        # Test Policy
        prev_state = env.reset()
        episodic_reward = 0

        while True:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = [agent.policy(tf_prev_state, agent.ou_noise, noise=False)]
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            episodic_reward += reward

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        agent.ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        #avg_reward = np.mean(agent.ep_reward_list[-40:])
        #print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        with agent.summary_writer.as_default():
            tf.summary.scalar('mean_reward', episodic_reward, step=ep)
        # agent.avg_reward_list.append(avg_reward)


    """
    If training proceeds correctly, the average episodic reward will increase with time.

    Feel free to try different learning rates, `tau` values, and architectures for the
    Actor and Critic networks.

    The Inverted Pendulum problem has low complexity, but DDPG work great on many other
    problems.

    Another great environment to try this on is `LunarLandingContinuous-v2`, but it will take
    more episodes to obtain good results.
    """

    # Save the weights
    agent.actor_model.save("models/pendulum_actor.h5")
    agent.critic_model.save("models/pendulum_critic.h5")

    #agent.target_actor.save_weights("pendulum_target_actor.h5")
    #agent.target_critic.save_weights("pendulum_target_critic.h5")

    """
    Before Training:

    ![before_img](https://i.imgur.com/ox6b9rC.gif)
    """

    """
    After 100 episodes:

    ![after_img](https://i.imgur.com/eEH8Cz6.gif)
    """