#!/usr/bin/env python3

###########################################################################
#  Laboratory of Computational Physics // University of Padua, AY 2021/22
#  Group 2202 / Barone Nagaro Ninni Valentini
#
#    HAVOK analysis: module for moving window analysis
#
#  coder: Barone Francesco, last edit: 17 may 2022
#--------------------------------------------------------------------------
#  adapted from keras.io documentation 
#      @ https://keras.io/examples/rl/ddpg_pendulum/
#  under Apache License 2.0
#--------------------------------------------------------------------------

import numpy as np

class mowin():
    
    def __init__(self, U, sigma, mwin, thres, idx = None):
        #  inputs:
        #      U   :  the matrix of coefficients
        #            if string -> will read the file to retrieve the U matrix
        #            else ->      input the np matrix
        #
        #   sigma  :  sigma coefficients from havok SVD
        #   
        #    mwin  :  an array with the initialized moving window matrix
        #
        #   thres  :  threshold for havok analysis
        #
        #     idx  :  rows to extract from U matrix
        
        if isinstance(U, str):
            print('loading U matrix from file')
            self.Umat = np.loadtxt(U)            
        else:
            self.Umat = U
            
        self.mwsize = self.Umat.shape[0]
        print('moving window size:', self.mwsize)
        
        if idx is not None:
            self.Umat = self.Umat[:,idx]
        
        if len(mwin) == self.mwsize:
            self.mwdata = mwin
        else:
            print('[ERR] moving window has wrong size')
            return None
        
        if isinstance(sigma, str):
            print('loading sigma array from file')
            self.sigma = np.loadtxt(sigma)            
        else:
            self.sigma = sigma
        
        if idx is not None:
            if len(self.sigma) != len(idx):
                self.sigma = self.sigma[idx] # sigma is not cropped, so crop!
        
        self.threshold = thres
    
    
    
    def move(self, sample):
        # insert new sample in moving window ...
        self.mwdata = np.roll(self.mwdata, -1);
        self.mwdata[-1] = sample;
        
        # ... and compute the convolution
        conv = np.matmul(self.Umat.T, self.mwdata)/self.sigma
        return conv