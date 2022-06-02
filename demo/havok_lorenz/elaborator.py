#!/usr/bin/env python3
#
#  This code takes data from an input buffer & plots in real time the attractor
#  evolution, color coding its trajectory with HAVOK switching prediction.
#
#  >> 09 may 2022, Francesco Barone
#  Laboratory of Computational Physics, University of Padua

import os
import sys
import select
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D

# set input buffer
try:
    PIPE_IN = sys.argv[1]
except:
    PIPE_IN  = 'data.pipe'


# animation properties
fps = 30

# parameters
thres = 0.002
coeff = 1/9.740682624852331e-09
r = 15
U_FILE = 'u.csv'
############### utilities ################

# interpret the input buffer string
def input_parse(msg, **kwargs):
    msg = msg.split()
    try:
        return (float(msg[0]), float(msg[1]), float(msg[2]))
    except:
        print(' pipe ended')
        exit(0)

##########################################
#################  main  #################
##########################################

print(' [ ELABORATOR ] my PID is', os.getpid())
print('source data pipe: ', PIPE_IN )


## load convolution matrix
print('loading U matrix from file...')

Umat = np.loadtxt(U_FILE)
q = Umat.shape[0]  # size of convolution
Umat = Umat[:,r-1]
print('U matrix loaded, using convolution size =', q)


##  initialize moving window vector (read from buffer)
print('waiting for target pipe to be opened...')
pp_in = open(PIPE_IN, 'r')

data_mat = np.empty(q)
for i in range(q):
    data_mat[i] = input_parse( pp_in.readline() )[0]

print('moving window initialized')


# define batch sizes
pp_batch_size = 100
plot_batch_size = 3000


# preallocate plot data batch
plot_batch = np.zeros( (plot_batch_size, 5) )
size_vect = 3*(np.linspace(0.01, 1, plot_batch_size)**2)
pactive = 0  # forcing counter

#-------- plot update function --------#

def update(i):
    # this function reads data from buffer & updates plot
    global data_mat, pp_batch_size, plot_batch, pactive
    
    xyz = []
    pp_counter = 0
    while pp_counter < pp_batch_size:
        # read data & move window
        (ix, iy, iz) = input_parse( pp_in.readline() )
        data_mat = np.roll(data_mat, -1);  data_mat[-1] = ix;
        
        # compute forcing term
        v = np.matmul(Umat.T, data_mat)*coeff
        trigger = 1 if( abs(v) > thres ) else 0
        
        # perform logic on color coding
        tt = abs(v) > thres
        if pactive == 0:
            cc = 1 if(tt) else 0
            if tt: pactive = 500
        else:
            cc = 1;   pactive -= 1;
            
        xyz.append( [ix, iy, iz, v, cc] )
        pp_counter += 1
    
    plot_batch = np.concatenate((plot_batch[pp_batch_size:], np.matrix(xyz)), axis=0) 
    
    ax.clear();   ax.view_init(7, i/2);  #ax.set_axis_off();
    ax.set_xlim3d(-30, 30); ax.set_ylim3d(-30, 30);ax.set_zlim3d(0, 45);
    
    ax.scatter(plot_batch[:,0], plot_batch[:,1], plot_batch[:,2],
               color = [ 'red' if x else 'black' for x in plot_batch[:,4] ], 
               s = size_vect )
    pp_counter = 0
    
#--------------------------------------#

print('elaborating...')

# spawn figure
fig = plt.figure(figsize = (11, 9))
ax = plt.axes(projection='3d')
fig.tight_layout()

# execute animation
ani = matplotlib.animation.FuncAnimation(fig, update, interval=(1/fps)*1000, blit=False)
plt.show()

print('All operations completed.')
exit(0)
