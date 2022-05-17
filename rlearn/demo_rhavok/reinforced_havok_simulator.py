#!/usr/bin/env python3

###########################################################################
#  Laboratory of Computational Physics // University of Padua, AY 2021/22
#  Group 2202 / Barone Nagaro Ninni Valentini
#
#    Simulate a reinforced HAVOK model.
#
#  coder: Barone Francesco, last edit: 17 may 2022
#--------------------------------------------------------------------------
#  released under Creative Commons Zero v1.0 Universal license
#--------------------------------------------------------------------------

try:
    import cayde_env  # importing a custom environment for my server
    cayde_env.tensorflow('CPU')
except: print('no need of custom environment')
print('\n\n')

import sys
import os
import time
from datetime import datetime
#from threading import Thread
import multiprocessing
from multiprocessing import shared_memory

import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from rhavok.gym import DynamicalSystem
from rhavok.systems import Lorenz
from rhavok.utils import OUActionNoise
from rhavok.utils import mybuffer
from rhavok.utils import Lorenz_dashboard
from rhavok.havok import mowin

##### simulation properties #####

dt = 0.001
action_smooth_factor = 0.01
action_utime_rateo = 10

############# IO ###############

U_FILE = './data/u.csv'
S_FILE = './data/sigma.csv'

MODEL_PATH = './models/'
fstring = f'{MODEL_PATH}lorenz_05-16_22-29mm__'

######### RL model ###########

num_states = 3
print("Size of State Space ->  {}".format(num_states))
num_actions = 1
print("Size of Action Space ->  {}".format(num_actions))
upper_bound,lower_bound = +12, -12
print("Value of Action ->  {} to {}".format(lower_bound, upper_bound))

def policy(state, noise_object = None):
    sampled_actions = tf.squeeze(actor_model(state))
    
    if noise_object is not None: noise = noise_object()
    else: noise = 0
    
    # add noise to action
    sampled_actions = sampled_actions.numpy() + noise
    
    # check action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    
    return [np.squeeze(legal_action)]

# load model from file
print('loading actor critic model:', f'{fstring}actor.h5')
actor_model = tf.keras.models.load_model(f'{fstring}actor.h5')
print('loaded!')

##########################

rp = { 'fpd_scale' : 0.90, 'dx_penalty' : 0.01, 'critical_penalty' : None }

system = Lorenz()
env = DynamicalSystem(system, dt = dt, reward_param = rp)
env.seed(420)

mw_size = 100
prev_state, init_mw = env.reset(burnout_steps = mw_size, burnout_return=True)  # reset system

######## env vars #########

action_active = False
action_smoothed = 0
action_counter = 0
action_duration = 4000

havok_monitor = True
havok_monitor_deadtime = 1500

bff = mybuffer(capacity=int(10e5), record_size = 7)
hwm = mowin(U_FILE, S_FILE, mwin = init_mw, thres = 0.002, idx = [0,14])

semaphore_plot_thread = True

###########################
###########################

render_thr = Lorenz_dashboard(bff, semaphore_plot_thread, system)

p = multiprocessing.Process(target=render_thr.run)#, args=(i, i, i))
p.start()
#render_thr.daemon = True # let the main thread exit even though the workers are blocking
#render_thr.start()  # approach: https://www.toptal.com/python/beginners-guide-to-concurrency-and-parallelism-in-python

print('simulation online')

ii = 0
while True:
    
    tic = time.process_time()
    
    if action_counter > action_duration:
        action_active = False
        if action_counter > action_duration + havok_monitor_deadtime:
            havok_monitor = True
            action_counter = 0
            print(f' [{ii}] HAVOK monitor on')
            
    if action_active:
        if(action_counter % action_utime_rateo == 0):
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = policy(tf_prev_state)
        
        action_counter += 1
    else:
        action = [0]
    
    # propagate system status
    action_smoothed = action_smooth_factor*action[0] + (1-action_smooth_factor)*action_smoothed
    state, reward, done, _ = env.step(action_smoothed)
    
    # havok analysis
    havok_v1, havok_vr = hwm.move(state[0])
    if havok_monitor:
        if abs(havok_vr) > hwm.threshold:
            print(f' [{ii}] HAVOK monitor off')
            action_active = True
            havok_monitor = False
    else:
        action_counter += 1
    
    # buffer, mainly for online plotting
    bff.record(state, havok_v1, havok_vr, action_smoothed, reward)
    
    done = False
    if done:
        print('DONE')
        #break
        
    toc = time.process_time()
    
    #if ii % 100 == 0:
    #    render_thr.run()
    # wait some time
    #print(toc - tic)
    
    time.sleep(0.001)
    prev_state = state
    ii += 1