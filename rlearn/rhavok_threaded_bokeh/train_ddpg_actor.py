#!/usr/bin/env python3

###########################################################################
#  Laboratory of Computational Physics // University of Padua, AY 2021/22
#  Group 2202 / Barone Nagaro Ninni Valentini
#
#    Train an actor critic network (reinforcement learning) with
#    Deep Deterministic Policy Gradient for Lorenz transient
#    state recovery.
#
#  coder: Barone Francesco, last edit: 16 may 2022
#--------------------------------------------------------------------------
#  adapted from keras.io documentation 
#      @ https://keras.io/examples/rl/ddpg_pendulum/
#  under Apache License 2.0
#--------------------------------------------------------------------------


try:
    import cayde_env  # importing a custom environment for my server
    cayde_env.tensorflow('CPU')
except: print('no need of custom environment')
print('\n\n')

import sys
import os
from datetime import datetime

import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from rhavok.gym import DynamicalSystem
from rhavok.systems import Lorenz
from rhavok.utils import OUActionNoise
#from rhavok.utils import Buffer

######### parameters ###########

rp = { 'fpd_scale' : 0.90, 'dx_penalty' : 0.1, 'critical_penalty' : None }

system = Lorenz(x0_dataset = './data/havok_critic_Lorenz.csv')
env = DynamicalSystem(system, dt = 0.01, reward_param = rp)
env.seed(420)

num_states = 3
print("Size of State Space ->  {}".format(num_states))
num_actions = 1
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = +10
lower_bound = -10

print("Value of Action ->  {} to {}".format(lower_bound, upper_bound))

max_steps_per_episode = 300
total_episodes = 120
action_smooth_factor = 0.1
reward_averaging_stat = 10

do_plot = False  # plot while training

############# IO ###############

MODEL_PATH = './models/'

################################

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() #+ noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]
    
# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

class learn_buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

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
    def update(self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

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

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

################################

# training hyperparameters
std_dev = 0.1
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# weights initially equal
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)


# discount factor for future rewards
gamma = 0.99
# used to update target networks
tau = 0.005

buffer = learn_buffer(50000, 64)

################################
################################

if do_plot: env.init_render()

print('\n\n------ BEGIN OPTIMIZATION ------')

# store reward history of each episode
ep_reward_list = []
# store average reward history of last few episodes
avg_reward_list = []


for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0
    action_smoothed = 0
    action_hist = []
    reward_hist = []
    
    for timestep in range(0, max_steps_per_episode):

        if do_plot: env.render(action_hist, reward_hist, mode='full')

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        # note: action is [array(FLOAT)]
        action_smoothed = action_smooth_factor*action[0] + (1-action_smooth_factor)*action_smoothed
        state, reward, done, info = env.step(action_smoothed)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        
        action_hist.append(action_smoothed)
        reward_hist.append(reward)
        
        if done: break  # end episode if actor returns done

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # mean of last episodes
    avg_reward = np.mean(ep_reward_list[-reward_averaging_stat:])
    print("[episode {}] reward {:.2f}, avg_{} {:.2f}".format(ep, episodic_reward, reward_averaging_stat,avg_reward))
    avg_reward_list.append(avg_reward)


################################

dt_string = datetime.now().strftime("%m-%d_%H-%M")

# plot: episodes versus avg. rewards
plt.figure(figsize=(8,5)) 
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.savefig(f'{MODEL_PATH}lorenz_reward_{dt_string}.png')
plt.show()

################################

fstring = f'{MODEL_PATH}lorenz_{dt_string}__'
print('saving model as', fstring+'*')

# save the weights
#actor_model.save_weights(f'{fstring}actor.h5')
#critic_model.save_weights(f'{fstring}critic.h5')

#target_actor.save_weights(f'{fstring}target_actor.h5')
#target_critic.save_weights(f'{fstring}target_critic.h5')

fstring = f'{MODEL_PATH}lorenz_{dt_string}mm__'
actor_model.save(f'{fstring}actor.h5')
critic_model.save(f'{fstring}critic.h5')
#target_actor.save(f'{fstring}target_actor.h5')
#target_critic.save(f'{fstring}target_critic.h5')

print('saved!')
################################

env.init_render()

print('\n\n------ BEGIN DEMO ------')

for ep in range(100):
    try:
        prev_state = env.reset()
        episodic_reward = 0
        action_smoothed = 0
        
        action_hist = []
        reward_hist = []
        
        for timestep in range(0, max_steps_per_episode):
            env.render(action_hist, reward_hist, mode='actionreward')

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)
            action_smoothed = action_smooth_factor*action[0] + (1-action_smooth_factor)*action_smoothed
            state, reward, done, info = env.step(action_smoothed)
            
            action_hist.append(action_smoothed)
            reward_hist.append(reward)
            
            if done: break
            prev_state = state
        
    except:
        print('\nexit from demo')
        break
        
# eop
os._exit(os.EX_OK)
