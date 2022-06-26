
###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python module implements a custom buffer.
#
#  last edit: 17 may 2022
#--------------------------------------------------------------------------
#  adapted from keras.io documentation 
#      @ https://keras.io/examples/rl/ddpg_pendulum/
#  under Apache License 2.0
#--------------------------------------------------------------------------

import numpy as np

class mybuffer:
    def __init__(self, capacity=int(10e5), record_size = 1, osplit = None, dtype = None):
    
        # max number of records to store
        self.buffer_capacity = capacity
        
        # allocate buffer
        if dtype is None:
            self.data = np.zeros((self.buffer_capacity, int(record_size)))
        else:  # sample dtype: 'uint8, float, bool, float64'
            if len(dtype.split()) != record_size:
                print('warning: record_size does not match dtype string')
            self.data = np.zeros((self.buffer_capacity,), dtype=dtype)
        
        self.reset()
        
        self.osplit = osplit
    
    
    def reset(self):
        self.record_counter = 0 # num of times record() was called
        self.read_counter = 0   # read index of fifo

    # add a (single) record to buffer
    def record(self, *args, **kwargs):
        # reset index to 0 if exceeds capacity, replacing old records
        index = self.record_counter % self.buffer_capacity
        
        self.data[index] = tuple(np.concatenate([ np.atleast_1d(ar) for ar in args]))
        self.record_counter += 1

    # read n values with fifo policy
    def read_fifo(self, n = 1):
        stop_idx = min(self.read_counter + n, self.record_counter)
        indexes = np.arange(start = self.read_counter,
                            stop = stop_idx) % self.buffer_capacity
        self.read_counter = stop_idx
        
        if self.osplit is None: return self.data[indexes]
        else: return np.split(self.data[indexes], self.osplit, axis=1)
    
    # read n values with lifo policy
    def read_lifo(self, n = 1):
        start_idx = max(self.read_counter, self.record_counter - n)
        indexes = np.arange(start = start_idx, stop = self.record_counter) % self.buffer_capacity
        self.read_counter = start_idx
        
        if self.osplit is None: return self.data[indexes]
        else: return np.split(self.data[indexes], self.osplit, axis=1)
