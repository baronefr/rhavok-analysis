#!/usr/bin/env python3

###########################################################################
#  Laboratory of Computational Physics // University of Padua, AY 2021/22
#  Group 2202 / Barone Nagaro Ninni Valentini
#
#    Visualize a trained actor critic model with matplotlib.
#
#  coder: Barone Francesco, last edit: 17 may 2022
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

######### parameters ###########

rp = { 'fpd_scale' : 0.90, 'dx_penalty' : 0.1, 'critical_penalty' : None }

system = Lorenz(x0_dataset = './data/havok_critic_Lorenz.csv')
env = DynamicalSystem(system, dt = 0.01, reward_param = rp)
env.seed(420)

num_states = 3
print("Size of State Space ->  {}".format(num_states))
num_actions = 1
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = +12
lower_bound = -12

print("Value of Action ->  {} to {}".format(lower_bound, upper_bound))

max_steps_per_episode = 300
total_episodes = 100
action_smooth_factor = 0.1
reward_averaging_stat = 10

############# IO ###############

MODEL_PATH = './models/'
fstring = f'{MODEL_PATH}lorenz_05-16_22-29mm__'

################################

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # add noise to action
    sampled_actions = sampled_actions.numpy() #+ noise

    # check action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

################################

# training hyperparameters
std_dev = 0.1
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# load model from file
print('loading model:', f'{fstring}actor.h5')
actor_model = tf.keras.models.load_model(f'{fstring}actor.h5')
print('loaded!')

################################
################################

env.init_render()

print('\n\n------ BEGIN DEMO ------')

ep_reward_list = []
avg_reward_list = []


for ep in range(total_episodes):

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

        episodic_reward += reward
        
        action_hist.append(action_smoothed)
        reward_hist.append(reward)
        
        if done: break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # mean of last episodes
    avg_reward = np.mean(ep_reward_list[-reward_averaging_stat:])
    print("[demo {}] reward {:.2f}, avg_{} {:.2f}".format(ep, episodic_reward, reward_averaging_stat,avg_reward))
    avg_reward_list.append(avg_reward)


# eop
os._exit(os.EX_OK)
