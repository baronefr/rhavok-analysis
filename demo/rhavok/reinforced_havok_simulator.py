#!/usr/bin/env python3

###########################################################################
#  Laboratory of Computational Physics // University of Padua, AY 2021/22
#  Group 2202 / Barone Nagaro Ninni Valentini
#
#    Simulate a reinforced HAVOK model.
#
#  coder: Barone Francesco, last edit: 18 may 2022
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
from threading import Thread

import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from rhavok.gym import DynamicalSystem
from rhavok.systems import Lorenz
from rhavok.utils import OUActionNoise
from rhavok.utils import mybuffer
from rhavok.utils import plt_dashboard
from rhavok.havok import mowin


dt = 0.001
rp = { 'fpd_scale' : 0.90, 'dx_penalty' : 0.01, 'critical_penalty' : None }

system = Lorenz()
env = DynamicalSystem(system, dt = dt, reward_param = rp)
env.seed(420)


bff = mybuffer(capacity=int(10e5), record_size = 8)


class rhavok_simulator(Thread):
    def __init__(self, shared_buffer, system, semaphore = True):
        self.bff = shared_buffer
        self.system = system
        self.semaphore = semaphore    
        Thread.__init__(self)       
        ##### simulation properties #####
        
        self.dt = 0.001
        self.action_smooth_factor = 0.01
        self.action_utime_rateo = 10
        
        ############# IO ###############
        
        self.U_FILE = './data/u.csv'
        self.S_FILE = './data/sigma.csv'
        
        MODEL_PATH = './models/'
        self.fstring = f'{MODEL_PATH}lorenz_05-16_22-29mm__'
        
        ######### RL model ###########
        
        self.num_states = 3
        print("Size of State Space ->  {}".format(self.num_states))
        self.num_actions = 1
        print("Size of Action Space ->  {}".format(self.num_actions))
        self.upper_bound, self.lower_bound = +12, -12
        print("Value of Action ->  {} to {}".format(self.lower_bound, self.upper_bound))
        
        # load model from file
        print('loading actor critic model:', f'{self.fstring}actor.h5')
        self.actor_model = tf.keras.models.load_model(f'{self.fstring}actor.h5')
        print('loaded!')
        
        
    def policy(self, state, noise_object = None):
        sampled_actions = tf.squeeze(self.actor_model(state))
    
        if noise_object is not None: noise = noise_object()
        else: noise = 0
    
        # add noise to action
        sampled_actions = sampled_actions.numpy() + noise
    
        # check action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
    
        return [np.squeeze(legal_action)]
    
    def semaphore_toggle(self):
        print('\n [rs] ','pause' if self.semaphore else 'start', 'simulation\n')
        self.semaphore = not self.semaphore


    def run(self):
        print('simulation online')
        
        ######## env vars #########
        action_active = False
        action_smoothed = 0
        action_counter = 0
        action_duration = 4000
        
        
        havok_monitor = True
        havok_monitor_deadtime = 1500
        
        

        mw_size = 100
        prev_state, init_mw = env.reset(burnout_steps = mw_size, burnout_return=True)  # reset system
        action_sign_store = np.sign(prev_state[0])
        
        hwm = mowin(self.U_FILE, self.S_FILE, mwin = init_mw, thres = 0.002, idx = [0,14])
        
        
        ii = 0
        sim_t = 0
        
        if not self.semaphore: # wait to start simulation
            print('simulation ready, wait for semaphore')
        
        while True:
            
            if not self.semaphore:
                time.sleep(0.5)
                continue
            
            tic = time.process_time()
            
            if action_counter > action_duration:
                action_active = False
                if action_counter > action_duration + havok_monitor_deadtime:
                    havok_monitor = True
                    action_counter = 0
                    print(f' [{ii}] HAVOK monitor on')
                    this_sign = np.sign(prev_state[0])
                    if this_sign != action_sign_store:
                        print(f' [info] sign has switched, sorry!')
                        #self.system.sample_sign = this_sign
                        #action_sign_store = this_sign
                    
            if action_active:
                if(action_counter % self.action_utime_rateo == 0):
                    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                    action = self.policy(tf_prev_state)
                
                action_counter += 1
            else:
                action = [0]
            
            # propagate system status
            action_smoothed = self.action_smooth_factor*action[0] + (1-self.action_smooth_factor)*action_smoothed
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
            st = 0
            if havok_monitor: st += 1
            if action_active: st += 2
            bff.record(sim_t, state, havok_vr, action_smoothed, reward, st)
            
            done = False
            if done:
                print('DONE')
                #break
                
            toc = time.process_time()
            time.sleep( max(0.001-toc+tic,0) )
            
            prev_state = state
            ii += 1
            sim_t += self.dt
            
            
            
##########################
###########################


# start simulator process
rs = rhavok_simulator(bff, system, False)
rs.daemon = True # let the main thread exit even though the workers are blocking
rs.start()

print('model simulation online!')

###########################

## IF USE MATLAB:
#semaphore_plot_thread = True
#pdb = plt_dashboard(bff, semaphore_plot_thread, system)
#rs.run_simulation()
#pdb.run()  ## MATPLOTLIB





###########################
# use BOKEH WEB INTERFACE #
###########################

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Paragraph, Range1d, Slider
from bokeh.plotting import curdoc, figure

from bokeh.server.server import Server
from bokeh.models import Span
from bokeh.models import Button
from bokeh.models import Label

rollover = 2600



def stream():  # read data from buffer & update plots
    global bff
    dd = bff.read_lifo(n=rollover)
    
    cds0.stream(dict(x=dd[:,1], z=dd[:,3]), rollover=rollover)
    #cds1.stream(dict(x=dd[:,7], z=dd[:,3]), rollover=rollover)
    cds2.stream(dict(t=dd[:,0], v=dd[:,4]), rollover=rollover)
    cds3.stream(dict(t=dd[:,0], v=dd[:,5]), rollover=rollover)
    cds4.stream(dict(t=dd[:,0], v=dd[:,6]), rollover=rollover)
    
    st = int(dd[-1,7]);  havok_monitor = bool(st % 2);    action_active = bool(st - havok_monitor);
    havok_monitor_txt = '<b style="color:blue;">active</b>' if havok_monitor else '<b style="color:red;">triggered</b>' 
    info_div.text = info_div_base.format(havok_monitor_txt,'---')

    

###############
# bokeh plots #
###############

# coordinates plot
p_coords = figure(
        width=500, height=500, x_range=Range1d(-20.0, 20.0), y_range=Range1d(0, 50),
        toolbar_location=None, background_fill_color="#ffffff",
    )
p_coords.xaxis.axis_label = r"$$x$$"
p_coords.yaxis.axis_label = r"$$z$$"
p_coords.title = "Lorenz attractor (2D proj)"
cds0 = ColumnDataSource(data=dict(x=[], z=[]))
line0 = p_coords.line(source=cds0, x="x", y="z", line_width = 2, line_color = 'purple')



## HAVOK monitor plots
p_havok_vr = figure(
        width=600, height=250, y_range=Range1d(-0.02, 0.02), #x_range=Range1d(0, 2600),  y_range=Range1d(-0.1, 0.1),
        toolbar_location=None, background_fill_color="#ffffff",
    )
p_havok_vr.xaxis.axis_label = r"$$t$$"
p_havok_vr.yaxis.axis_label = r"$$v_r$$"
p_havok_vr.title = "HAVOK forcing"
cds2 = ColumnDataSource(data=dict(t=[], v=[]))
line2 = p_havok_vr.line(source=cds2, x="t", y="v", line_width = 2, line_color = 'orange')
hline = Span(location=0.002, dimension='width', line_color='red', line_width=1) # add thres line
p_havok_vr.renderers.extend([hline])

# action plot
p_action = figure(
        width=600, height=250, y_range=Range1d(-12, 12), #x_range=Range1d(0, 2600),  y_range=Range1d(-0.1, 0.1),
        toolbar_location=None, background_fill_color="#ffffff"
    )
p_action.xaxis.axis_label = r"$$t$$"
p_action.yaxis.axis_label = r"$$\Delta\rho$$"
p_action.title = "DDPG action"
cds3 = ColumnDataSource(data=dict(t=[], v=[]))
line3 = p_action.line(source=cds3, x="t", y="v", line_width = 2, line_color = 'red')

# reward plot
p_reward = figure(
        width=600, height=250, y_range=Range1d(-3, 1),
        toolbar_location=None, background_fill_color="#ffffff",
    )
p_reward.xaxis.axis_label = r"$$t$$"
p_reward.yaxis.axis_label = r""
p_reward.title = "DDPG reward"
cds4 = ColumnDataSource(data=dict(t=[], v=[]))
line4 = p_reward.line(source=cds4, x="t", y="v", line_width = 2, line_color = 'green')




##########
#  misc  #
##########

# button to start/stop simulation
bt = Button(label='start/pause')
bt.on_click(rs.semaphore_toggle)

# header
big = {"font-size": "150%", "font-weight": "bold"}
head_text = [
    Div(text="reinforced HAVOK model", style=big),
    Div(text=r"Laboratory of Computational Physics, mod B"),
    Div(text="""<b>Group 2202</b> // this coder:  Barone""")
]

# system controller info
info_div_base = "<b>System controllers</b><br> HAVOK monitor: {}<br> action: {}"
info_div = Div( text=info_div_base.format('-','-'), width=500, height=200, style = {"font-size": "150%"})

# CREATE PAGE
doc = curdoc()
doc.add_periodic_callback(stream, 200)

doc.add_root( column( column(head_text), 
                      row(column(p_coords,bt,column(info_div)), column(p_havok_vr, p_action, p_reward)), 
                    ) 
            )






