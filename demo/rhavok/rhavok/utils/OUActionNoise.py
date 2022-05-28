
###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python module implements noisy perturbations for actions,
#  in order to implement a better exploration of an Actor network.
#  The process used to generate noise is an Ornstein-Uhlenbeck, which
#  samples from a correlated normal distribution.
#
#  coder: Barone Francesco, last edit: 17 may
#--------------------------------------------------------------------------
#  adapted from keras.io documentation 
#      @ https://keras.io/examples/rl/ddpg_pendulum/
#  under Apache License 2.0
#--------------------------------------------------------------------------

import numpy as np

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # ref https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # store x into x_prev
        # makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
