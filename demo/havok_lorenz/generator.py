#!/usr/bin/env python3
#
#  This code feeds data from a file to a buffer.
#
#  >> 09 may 2022, Francesco Barone
#  Laboratory of Computational Physics, University of Padua

import os
import time
import numpy as np

input_file = '../../data/sb2017/Lorenz_extended.csv'
dt = 0.001   # time step

# set input buffer
try:
    PIPE_OUT = sys.argv[1]
except:
    PIPE_OUT  = 'data.pipe'

###############################
#             main            #
###############################

print(' [GENERATOR] my PID is', os.getpid())
print('target data pipe: ', PIPE_OUT)

print('reading buffer from file')
xx, yy, zz = np.loadtxt(input_file, unpack=True)
nmax = len(xx)
print('file buffer acquired')


print('waiting for target pipe to be opened...')
pp_out = os.open(PIPE_OUT, os.O_WRONLY)
print('pipe has been opened, streaming...')


try:
    i = 0
    while i<nmax:
        xs, ys, zs = xx[i], yy[i], zz[i]

        # buffer data
        value = str(xs) + ' ' + str(ys) + ' ' + str(zs) + '\n'
        os.write(pp_out, value.encode())
        
        if not ( i%int(nmax/30) ): print('.')
        i += 1
        time.sleep(dt/4)
    
    print("\n --- file limit reached --")
    
except KeyboardInterrupt:
    print("\n WARNING, Keyboard interrupt!")
    exit(1)
except BrokenPipeError:
    print("\n ERROR, broken pipe!")
    exit(1)
finally:
    print("\n end of generator")
    os.close(pp_out)
    exit(0)