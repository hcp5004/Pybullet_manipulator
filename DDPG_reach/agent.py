"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
Date created: 2020/06/04
Last modified: 2020/09/21
Description: Implementing DDPG algorithm on the Inverted Pendulum Problem.
Accelerator: NONE
"""
"""
## Introduction

**Deep Deterministic Policy Gradient (DDPG)** is a model-free off-policy algorithm for
learning continous actions.

It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network).
It uses Experience Replay and slow-learning target networks from DQN, and it is based on
DPG,
which can operate over continuous action spaces.

This tutorial closely follow this paper -
[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)

## Problem

We are trying to solve the classic **Inverted Pendulum** control problem.
In this setting, we can take only two actions: swing left or swing right.

What make this problem challenging for Q-Learning Algorithms is that actions
are **continuous** instead of being **discrete**. That is, instead of using two
discrete actions like `-1` or `+1`, we have to select from infinite actions
ranging from `-2` to `+2`.

## Quick theory

Just like the Actor-Critic method, we have two networks:

1. Actor - It proposes an action given a state.
2. Critic - It predicts if the action is good (positive value) or bad (negative value)
given a state and an action.

DDPG uses two more techniques not present in the original DQN:

**First, it uses two Target networks.**

**Why?** Because it add stability to training. In short, we are learning from estimated
targets and Target networks are updated slowly, hence keeping our estimated targets
stable.

Conceptually, this is like saying, "I have an idea of how to play this well,
I'm going to try it out for a bit until I find something better",
as opposed to saying "I'm going to re-learn how to play this entire game after every
move".
See this [StackOverflow answer](https://stackoverflow.com/a/54238556/13475679).

**Second, it uses Experience Replay.**

We store list of tuples `(state, action, reward, next_state)`, and instead of
learning only from recent experience, we learn from sampling all of our experience
accumulated so far.

Now, let's see how is it implemented.
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
The `Buffer` class implements Experience Replay.

---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---


**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""

class Buffer:
    def __init__(self, agent, buffer_capacity=100000, batch_size=64, write=True):
        self.write = write
        
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.agent = agent
        self.state_buffer = np.zeros((self.buffer_capacity, self.agent.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.agent.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, agent.num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.agent.target_actor(next_state_batch, training=True)
            y = reward_batch + self.agent.gamma * self.agent.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.agent.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.agent.critic_model.trainable_variables)
        self.agent.critic_optimizer.apply_gradients(
            zip(critic_grad, self.agent.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.agent.actor_model(state_batch, training=True)
            critic_value = self.agent.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.agent.actor_model.trainable_variables)
        self.agent.actor_optimizer.apply_gradients(
            zip(actor_grad, self.agent.actor_model.trainable_variables)
        )

        return critic_loss, actor_loss

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return self.update(state_batch, action_batch, reward_batch, next_state_batch)
    
    def WriteBoard(self,critic_loss, actor_loss, ep):
        with self.agent.summary_writer.as_default():
            tf.summary.scalar('critic_loss', critic_loss, step=ep)  

        with self.agent.summary_writer.as_default():
            tf.summary.scalar('actor_loss', actor_loss, step=ep)


class Agent:
    def __init__(self, env, step=0, done=None, write=True):
        
        self.write = write
        self.Init_board()

        self.env = env

        #self.num_states = env.observation_space.shape[0] #dimension
        self.num_states = env.observation_space['observation'].shape[0]
        self.num_actions = env.action_space.shape[0] #dimension

        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low

        print("num_states: {}".format(self.num_states))
        print("num_actions: {}".format(self.num_actions))
        print("upper_bound: {}".format(self.upper_bound))
        print("lower_bound: {}".format(self.lower_bound))
        """
        ## Training hyperparameters
        """
        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.std_dev) * np.ones(self.num_actions))

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.actor_model.summary()
        self.critic_model.summary()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = 0.001
        self.actor_lr = 0.0001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        self.total_episodes = 20000
        # Discount factor for future rewards    .
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.001

        self.buffer = Buffer(self, 1000000, 256, write=self.write)

        self.actor_l2_weight = 0.01
        self.critic_l2_weight = 0.01

        self.step = step
        self.done = done

        """
        Now we implement our main training loop, and iterate over episodes.
        We sample actions using `policy()` and train with `learn()` at each time step,
        along with updating the Target networks at a rate `tau`.
        """
    
    def Init_board(self):
        if self.write:
            self.log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        else:
            self.log_dir = None
            self.summary_writer = None

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for a, b in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


    """
    Here we define the Actor and Critic networks. These are basic Dense models
    with `ReLU` activation.

    Note: We need the initialization for last layer of the Actor to be between
    `-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
    the initial stages, which would squash our gradients to zero,
    as we use the `tanh` activation.
    """


    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(400, activation="relu")(inputs)
        out = layers.Dense(300, activation="relu")(out)
        #out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=last_init)(out)
        outputs = outputs * tf.expand_dims(tf.convert_to_tensor(self.upper_bound), axis=0)
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)
        #State as input
        # state_input = layers.Input(shape=(self.num_states,))
        # state_out = layers.Dense(16, activation="relu")(state_input)
        # state_out = layers.Dense(32, activation="relu")(state_out)

        # # Action as input
        # action_input = layers.Input(shape=(self.num_actions,))
        # action_out = layers.Dense(32, activation="relu")(action_input)

        # # Both are passed through seperate layer before concatenating
        # concat = layers.Concatenate()([state_out, action_out])

        # out = layers.Dense(400, activation="relu")(concat)
        # out = layers.Dense(300, activation="relu")(out)
        # outputs = layers.Dense(1,kernel_initializer=last_init)(out)
        # model = tf.keras.Model([state_input, action_input], outputs)
        
        state_input = layers.Input(shape=(self.num_states))
        action_input = layers.Input(shape=(self.num_actions))
        input= layers.Concatenate()([state_input, action_input])
        out = layers.Dense(400, activation="relu")(input)
        out = layers.Dense(300, activation="relu")(out)     
        #out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1,kernel_initializer=last_init)(out)
        model = tf.keras.Model([state_input, action_input], outputs)

        return model


    """
    `policy()` returns an action sampled from our Actor network plus some noise for
    exploration.
    """


    def policy(self, state, noise_object, noisy=True):
        sampled_actions = tf.squeeze(self.actor_model(state, training=False))
        if noisy:
            noise = noise_object()
            # Adding noise to action
            sampled_actions = sampled_actions.numpy() + noise
        else:
            sampled_actions = sampled_actions.numpy()
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return np.squeeze(legal_action)
