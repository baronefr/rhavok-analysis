
###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python module implements an OpenAI gym environment 
#  for the control of a Lorenz attractor dynamical system.
#
#  coder: Barone Francesco, last edit: 17 may
#--------------------------------------------------------------------------
#  adapted from tiagoCuervo @ https://github.com/tiagoCuervo/dynSysEnv
#  under MIT License
#--------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
from gym import spaces


class DynamicalSystem(gym.Env):
    # custom environment in a gym-like interface

    def __init__(self, system, dt=0.01, reward_param = None):
        super(DynamicalSystem, self).__init__()
        sysDim = system.numStateVars
        # setup spaces
        self.action_space = spaces.Box(low=np.array([-1.0] * sysDim), high=np.array([1.0] * sysDim), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf] * sysDim), high=np.array([np.inf] * sysDim),
                                            dtype=np.float32)
        
        # buffers
        self.system = system
        self.state = system.initialize()
        self.dt = dt
        self.trajectory = np.expand_dims(self.state, axis=0)
        self.fig = None
        
        # reward param
        if reward_param is None:
            self.rew_fpd_scale = 0.90
            self.rew_dx_penalty = 0.1
            self.rew_critical_penalty = -200
        else:
            print('take reward params from dict')
            self.rew_fpd_scale = reward_param['fpd_scale']
            self.rew_dx_penalty = reward_param['dx_penalty']
            self.rew_critical_penalty = reward_param['critical_penalty']
            
    def init_render(self, ratio = 1.6):
        self.fig = plt.figure(figsize=plt.figaspect(ratio))
      
      
    ################
    # reward funct #
    ################
    
    def reward_fpdist(self, x):
        b = self.rew_fpd_scale*self.system.init_fp_dist
        if x > b:
            return( 1 - np.exp(-x) )
        else:
            return( np.exp(-x + b+np.log(2) ) - 1 )
    
    def reward_fpdist_gaussian(self, x, dx = None):
        b = self.rew_fpd_scale*self.system.init_fp_dist
        
        if b < 10e-1 or x < 10e-1:
            return 0
        if x > b:
            return ( 2*np.exp(-(x-b)*(x-b) ) - 1 )
        else:
            return ( np.exp(-(x-b)*(x-b)/(b*b/np.log(2)) ) )
    
    def reward_fpdist_gaussian_dv(self, x, dx):
        b = self.rew_fpd_scale*self.system.init_fp_dist
        
        if b < 10e-1 or x < 10e-1:
            rt = 0
        if x > b:
            rt = ( 3*np.exp(-(x-b)*(x-b) ) - 2 )
        else:
            rt = ( np.exp(-(x-b)*(x-b)/(b*b/np.log(2)) ) )
        
        return rt - np.exp(-self.rew_dx_penalty*np.linalg.norm(dx))
    
    
    
    ##################
    ## GYM functions #
    ##################
    
    def step(self, action):  # noinspection PyTypeChecker
    
        # process system dynamics
        deltaState = self.system.dynamics(self.state, 
                                          dpar = np.array([action,0,0]) ) # correct only sigma
                                          #dpar = np.array([action-(self.system.encoding_action_p/2),0,0]) ) # correct only sigma
        self.state = self.state + deltaState * self.dt
        
        self.trajectory = np.concatenate((self.trajectory, np.expand_dims(self.state, axis=0)), axis=0)

        # unperturbed system dynamics
        if self.unperturbed_traj is not None:
            self.unperturbed_state = self.unperturbed_state + self.system.dynamics(self.unperturbed_state) * self.dt
            self.unperturbed_traj = np.concatenate((self.unperturbed_traj, np.expand_dims(self.unperturbed_state, axis=0)), axis=0)
        
        # reward policy: fixed magnitude, switch sign as x sign
        #rew_sign = np.sign(self.state[0]*self.system.sample_sign)
        #reward = rew_sign*5
        #done = False

        
        # reward policy: distance from target fixed point
        done = False
        reward = self.reward_fpdist_gaussian_dv( np.linalg.norm(self.system.init_fp - self.state), deltaState )
        rew_sign = np.sign(self.state[0]*self.system.sample_sign)
        
        if rew_sign < 0:
            self.rew_exception += 1
            if self.rew_exception > 200:
                done = True
                if self.rew_critical_penalty is not None:
                    reward = self.rew_critical_penalty
        else:
            if self.rew_exception > 200:
                self.rew_exception = 0
                
        if np.isnan(reward):
            done = True
            reward = -10
            
        return self.state, reward, done, {'episode': None}




    def reset(self, burnout_steps = 100, burnout_return = False, custom_init_point = None):
        brnout = None
        
        # get a new initial state from system policy
        self.state = self.system.initialize(custom = custom_init_point)  
        
        # when random sampling mode, advance some steps for stability
        if (self.system.dataset_len is None) or burnout_return:
            if burnout_return: brnout = np.empty(burnout_steps)
            for t in range(burnout_steps):
                deltaState = self.system.dynamics(self.state)
                self.state = self.state + deltaState * self.dt
                if burnout_return: brnout[t] = self.state[0]
        
        # check if first sample is in x > 0 or < 0 ...
        self.system.sample_sign = np.sign(self.state[0])  
        
        # ... & associate the target fixed point
        self.system.init_fp = self.system.fp1 if(self.system.sample_sign > 0) else self.system.fp2
        
        # compute the initial distance from associated fixed point
        self.system.init_fp_dist = np.linalg.norm(self.system.init_fp - self.state)
        
        # assign the first point for trajectory plot
        self.trajectory = np.expand_dims(self.state, axis=0)
        
        self.rew_exception = 0 # reward exception counter
        
        # compute unperturbed trajectory (only when working in x0_dataset mode)
        if self.system.dataset_len is not None:
            self.unperturbed_state = self.state
            self.unperturbed_traj = np.expand_dims(self.unperturbed_state, axis=0)
        else:
            self.unperturbed_state = None
            self.unperturbed_traj = None
        
        return self.state, brnout

    def render(self, action_hist=None, reward_hist=None, mode='only_actor', close=False):
    
        if mode == 'only_actor':
            ax = self.fig.gca(projection='3d')
        else:
            ax = self.fig.add_subplot(5, 1, (1, 3), projection='3d')
        
        if self.unperturbed_traj is not None:
            # add unperturbed trajectory
            ax.plot(self.unperturbed_traj[:, 0], self.unperturbed_traj[:, 1], self.unperturbed_traj[:, 2], 
                    '--', lw=1.3, c='gray', label='unperturbed')
        
        # plot perturbed trajectory
        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2], 
                lw=2, c='orangered', label='actor network')
        
        # add Lorenz fixed points
        ax.scatter3D(self.system.fp1[0], self.system.fp1[1], self.system.fp1[2], c='k', s=5)
        ax.scatter3D(self.system.fp2[0], self.system.fp2[1], self.system.fp2[2], c='k', s=5)
        
        ax.legend()
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(self.system.name + " System")
        
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        ax.set_zlim(0,50)
        
        if mode == 'actionreward':
            ax = self.fig.add_subplot(5, 1, 4) # action plot
            ax.plot(action_hist, c='goldenrod')
            ax.set_ylabel("action")
            ax.set_xticks([])
            ax.yaxis.tick_right()
            
            ax = self.fig.add_subplot(5, 1, 5) # reward plot
            ax.plot(reward_hist, c='seagreen')
            ax.set_ylabel("reward")
            ax.set_xticks([])
            ax.yaxis.tick_right()
        
        plt.show(block=False)
        plt.pause(self.dt/100) # overhead?
        plt.clf()



    def buffer_render(self):
    
    
    
        return 0
