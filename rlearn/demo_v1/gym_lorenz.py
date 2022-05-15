###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python class implements an OpenAI gym environment 
#  for the control of a Lorenz attractor dynamical system.
#
#  coder: Barone Francesco, last edit: 15 may
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
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, system, dt=0.01):
        super(DynamicalSystem, self).__init__()
        sysDim = system.numStateVars
        # Setup spaces
        self.action_space = spaces.Box(low=np.array([-1.0] * sysDim), high=np.array([1.0] * sysDim), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf] * sysDim), high=np.array([np.inf] * sysDim),
                                            dtype=np.float32)
        
        # buffers
        self.system = system
        self.state = system.initialize()
        self.dt = dt
        self.trajectory = np.expand_dims(self.state, axis=0)
        self.fig = plt.figure()

    
    def step(self, action):  # noinspection PyTypeChecker
    
        # process system dynamics
        deltaState = self.system.dynamics(self.state, 
                                          dpar = np.array([-0.5*action,0,0]) ) # correct only sigma
                                          #dpar = np.array([action-(self.system.encoding_action_p/2),0,0]) ) # correct only sigma
        self.state = self.state + deltaState * self.dt
        
        self.trajectory = np.concatenate((self.trajectory, np.expand_dims(self.state, axis=0)), axis=0)

        # unperturbed system dynamics
        if self.unperturbed_traj is not None:
            self.unperturbed_state = self.unperturbed_state + self.system.dynamics(self.unperturbed_state) * self.dt
            self.unperturbed_traj = np.concatenate((self.unperturbed_traj, np.expand_dims(self.unperturbed_state, axis=0)), axis=0)
        
        # reward policy
        rew_sign = np.sign(self.state[0]*self.system.sample_sign)
        reward = rew_sign*5
        done = False
        
        if rew_sign < 0:
            self.system.exception += 1
            if self.system.exception > 300:
                done = True
                reward = -200
        
        return self.state, reward, done, {'episode': None}




    def reset(self):
        self.state = self.system.initialize()
        self.system.sample_sign = np.sign(self.state[0])
        
        if self.system.dataset_len is None:
            for t in range(300):  # when random sampling, advance some steps for stability
                deltaState = self.system.dynamics(self.state)
                self.state = self.state + deltaState * self.dt
        self.trajectory = np.expand_dims(self.state, axis=0)
        
        if self.system.dataset_len is not None:
            self.unperturbed_state = self.state
            self.unperturbed_traj = np.expand_dims(self.unperturbed_state, axis=0)
        else:
            self.unperturbed_state = None
            self.unperturbed_traj = None
        
        return self.state




    def render(self, mode='human', close=False):
        ax = self.fig.gca(projection='3d')
        
        if self.unperturbed_traj is not None:
            ax.plot(self.unperturbed_traj[:, 0], self.unperturbed_traj[:, 1], self.unperturbed_traj[:, 2], 
                    '--', lw=1.3, c='gray', label='unperturbed')
        
        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2], 
                lw=2, c='orangered', label='actor critic DNN')
        
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
        
        plt.show(block=False)
        plt.pause(self.dt/10)
        plt.clf()

