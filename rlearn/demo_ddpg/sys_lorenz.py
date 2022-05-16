###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python class implements a Lorenz attractor system.
#
#  coder: Barone Francesco, last edit: 16 may
#--------------------------------------------------------------------------
#  adapted from tiagoCuervo @ https://github.com/tiagoCuervo/dynSysEnv
#  under MIT License
#--------------------------------------------------------------------------


import numpy as np


class LorenzSystem:

    # init object
    def __init__(self, sigma=10, beta=8/3, rho=28, x0_dataset = None):  # nact = 0,
        self.name = 'Lorenz'
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.numStateVars = 3
        
        # customizations
        #self.encoding_action_p = nact  # alpha, keep it?
        
        # the two fixed-points of Lorenz system
        self.fp1 = np.array([ np.sqrt(self.beta*(self.rho - 1.)),  np.sqrt(self.beta*(self.rho - 1.)) , (self.rho - 1.)])
        self.fp2 = np.array([-np.sqrt(self.beta*(self.rho - 1.)), -np.sqrt(self.beta*(self.rho - 1.)) , (self.rho - 1.)])

        # init sampling mode
        if x0_dataset is None:
            # > if no argument, sample random points
            self.dataset = None
            self.dataset_len = None
        else:
            # > sample from input file
            self.dataset = np.loadtxt(x0_dataset)
            self.dataset_len = len(self.dataset)
            print(f' [{self.name}] loaded dataset of', self.dataset_len, 'samples')


    def initialize(self):
    
        if self.dataset is None:
            # > sample random points
            sample = [0., 1., 1.05] + np.random.normal(loc=0, scale=0.1, size=(self.numStateVars,))
        else:
            # > sample random points
            sample = self.dataset[ np.random.randint(self.dataset_len, size=1) ][0]

        #self.sample_sign = np.sign(sample[0])
        self.exception = 0  # a counter for some system exeptions
        return sample
        
        
    def dynamics(self, state, dpar = np.zeros(3)):
        # dynamics parameter correction:   dpar = [ dsigma, dbeta, drho ]
        
        dx = (self.sigma + dpar[0]) * (state[1] - state[0])
        dy = (self.rho + dpar[2]) * state[0] - state[1] - state[0] * state[2]
        dz = state[0] * state[1] - (self.beta + dpar[1])* state[2]
        return np.array([dx, dy, dz])
