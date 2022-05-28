#!/usr/bin/env python3
#
#  This code takes data from an input buffer 
#
#  Ref:   https://matplotlib.org/3.5.0/gallery/mplot3d/lorenz_attractor.html
#

import os
import select
import time
import numpy as np
import pipelib

PIPE_IN  = 'data.pipe'
PIPE_OUT = 'edata.pipe'

###############################

Umat = np.loadtxt('u.csv')
q = Umat.shape[0]  # size of convolution
r = 15

Umat = Umat[:,0:r]
print('U matrix loaded. Convolution size:', q)

data_mat = np.empty( (3, q) )

###############################


print(' [ CONVOLUTION ] my PID is', os.getpid())
print(' source data pipe: ', PIPE_IN )
print(' target data pipe: ', PIPE_OUT)

print(' waiting for target pipe to be opened...')
pp_out = os.open(PIPE_OUT, os.O_WRONLY)
print(' pipe has been opened')

wait_init = True

def compute_vr(msg, idx, **kwargs):
    global wait_init
    global data_mat
    
    thres = 0.002
    coeff = 1/9.740682624852331e-09
    
    msg = msg.split()
    ix, iy, iz = float(msg[0]), float(msg[1]), float(msg[2])
    
    if wait_init:
        data_mat[:, idx] = (ix, iy, iz)
        idx +=1
        if idx == q: wait_init = False
    
    else:
        data_mat = np.roll(data_mat, -1)
        data_mat[:, -1] = (ix, iy, iz)
        v = np.matmul(Umat.T, data_mat[0])
        trigger = 1 if( abs(v[-1]*coeff) > thres ) else 0
        
        value = str(ix) + ' ' + str(iy) + ' ' + str(iz) + ' ' + str(trigger) + '\n'
        os.write(pp_out, value.encode())
        print('.')
        
pipelib.iterative_nonblocking_pipe(PIPE_IN, compute_vr)