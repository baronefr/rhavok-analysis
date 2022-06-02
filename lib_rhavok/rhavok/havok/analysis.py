
###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python module implements the HAVOK analysis.
#
#  coder: Barone Francesco, last edit: 31 may 2022
#--------------------------------------------------------------------------
#  Open Access licence
#
#  reference paper: S. Brunton et al, 2017,
#                   Chaos as an intermittently forced linear system
#--------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import control # dynamic system simulation
import control.matlab as cnt

# SINDy routines
from rhavok.havok.sindy import sequential_threshold_least_squares
from rhavok.havok.sindy import SINDy_linear_library


class havok:
    
    # init object
    def __init__(self, dataset = None, dt = None, timing = None, crop = None,
                 time_delay_size = 100,
                 train_size = None, rank = 15, 
                 sindy_lib = 'linear', sindy_sparse = 'zero'):
        
        self.x = dataset
        self.dt = dt
        
        if timing is None:
            self.t = np.arange(len(self.x))*self.dt
        else: self.t = timing
        
        if crop is not None:
            print('applying crop index', crop)
            crop = int(crop)
            self.x = self.x[:crop]
            self.t = self.t[:crop]
        
        if time_delay_size < 2:
            print('ERROR: time delay must be greater than 1')
            return None
        else:
            self.q = time_delay_size
        
        if train_size is None: train_size = len(dataset)
        self.train_size = train_size
        self.datasize = len(self.x)
        if(self.train_size > self.datasize):
            print('WARNING: train size is greater than data size')
            self.train_size = self.datasize
        
        ### HAVOK objects ###
        self.H = None     # Hankel matrix
        
        self.svd_u = None  # SVD results
        self.svd_s = None
        self.svd_v = None
        
        self.rank = rank  # rank
        
        self.regr = None  # SINDy regression
        self.set_sindy()
        
        self.test_mowin = None  # mowin for test dataset
        self.test_mowin_time = None
        
        self.vsym = None  # simulator
        
        self.activity_criteria = None # functional for active criteria
        self.activity_criteria_args = {}
        
    
    def get_test_dataset(self):
        return self.x[self.train_size:]
    
    
    
    def build_Hankel(self, x = None):
        # create Hankel matrix
        
        if x is None: x = self.x
        m = self.train_size
        q = self.q
        
        H = np.zeros((q, m-q+1))
        for i in range(0, q):
            H[i,:] = x[ i:(m-q+1+i) ]
        
        print('Built Hankel matrix with shape', H.shape)
        
        self.H = H
        return self.H
    
    
    
    
    def svd(self, H = None):
        # perform SVD
        if H is None: H = self.H
        
        u, sigma, vh = np.linalg.svd(H, full_matrices=False, compute_uv=True)
        
        self.svd_u, self.svd_s, self.svd_v = u, sigma, vh
        return (u, sigma, vh)
    
    def svd_shapes(self):
        print('u:\t', self.svd_u.shape)
        print('sigma:\t', self.svd_t.shape)
        print('v:\t', self.svd_v.shape)
        
    def svd_plot_component(self, n=0):
        pass # plot a component of v SVD
    
    def svd_plot_embedded_3d(self):
        pass # plot embedded system (columns of v)
    
    
    
    
    def set_sindy(self, library = 'linear', sparser = 'zero'):
        # set the sindy regression routines
        
        if library == 'linear':
            self.sindy_library_generator = SINDy_linear_library
        else:
            print('WARNING unknown SINDy library, using default')
            self.sindy_library_generator = SINDy_linear_library
        
        if sparser == 'zero':     # zero at each value
            self.sindy_sparser = lambda k :0
        elif callable(sparser):  # custom function taking one parameter
            self.sindy_sparser = sparser
        else:
            print('WARNING unknown SINDy sparser, using default')
            self.sindy_sparser = lambda k :0
    
    
    def regression(self, vh = None, rank = None, thres = None):
        # perform sindy regression
        
        if rank is None: r = self.rank
        else: r = rank
        
        if vh is None: vh = self.svd_v
        
        # compute the derivative (4th order central difference)
        lv = vh.shape[1]
        dv_star = np.zeros( (lv-5,r) )
        for i in range(2,lv-3):
            dv_star[i-2,0:r] = (1/(12*self.dt))*(-vh[0:r, i+2]+8*vh[0:r, i+1]-8*vh[0:r, i-1]+vh[0:r, i-2])
        v_star = vh[0:r, 2:(dv_star.shape[0]+2)].T
        
        # generate library
        theta = self.sindy_library_generator(v_star)
        m = theta.shape[1]
        
        # normalize theta
        norm_theta = np.linalg.norm(theta,axis=0)  # required to normalize Xi
        theta = theta/norm_theta
        
        # sparse regression
        Xi = np.zeros( (m, r-1) )
        for k in range(0, r-1):  # apply separately on each column of Xi
            Xi[:,k] = sequential_threshold_least_squares(theta, dv_star[:,k],
                                                         self.sindy_sparser(k), 1)
        
        print('sparse regression Xi of shape', Xi.shape)
        
        # normalize Xi on Theta's 2-norm
        Xi = Xi/norm_theta[:,None]
        
        A = Xi[1:,:].T
        B = A[:,r-1]
        A = A[:,:r-1]
        B.shape = (r-1,1)
        
        self.regr = [A,B]
        return [A,B]
    
    
    
    def show_regression(self, regression = None, save = None):
        
        if regression is None: A, B = self.regr[0], self.regr[1]
        else: A, B = regression[0], regression[1]
        
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [14, 1]}, figsize=(7,6))
        fig.tight_layout()
        
        color_limit = max( np.ceil(np.abs(A).max()), np.ceil(np.abs(B).max()) )
        
        im = ax[0].imshow(A, cmap='RdBu', interpolation='nearest', aspect='equal', 
                          vmin=-color_limit, vmax=color_limit)
        ax[0].set(title='A')
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[0].get_yticklabels(), visible=False)
        ax[0].tick_params(axis='both', which='both', length=0)
         
        ax[1].imshow(B, cmap='RdBu', interpolation='nearest', aspect='equal', 
                     vmin=-color_limit, vmax=color_limit)
        ax[1].set(title='B')
        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_yticklabels(), visible=False)
        ax[1].tick_params(axis='both', which='both', length=0)
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.16, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        if save is not None:
            plt.savefig(save)
            print('Figure saved to', save)
        plt.show()
    
    
    def show_regression_numerical(self, A = None, rnd = 2):
        if A is None: A = self.regr[0]
        return pd.DataFrame(A).round(rnd)
    
    
    
    
    def compute_test_dataset(self):
        # uses a trained model (SVD) to compute the embedded
        # coordinates using a moving window
        train_size = self.train_size
        
        if train_size == self.datasize:
            print('ERROR: no data to compute')
            return None, None
        
        q = self.q
        r = self.rank
        
        mw = np.copy( self.x[(train_size-q):train_size] )
        u_conv = self.svd_u[:,0:r].T
        sigma_conv = self.svd_s[0:r]

        v_mw = np.zeros( (self.datasize-train_size, r) )
        for i in range(self.datasize-train_size):
            # roll the moving window & add new coordinate
            mw = np.roll(mw, -1)
            mw[-1] = self.x[train_size+i]
    
            # compute the convolution
            v_mw[i] = np.matmul(u_conv, mw)/sigma_conv
        t_mw = self.t[train_size:self.datasize]
        
        self.test_mowin = v_mw
        self.test_mowin_time = t_mw
        return v_mw, t_mw
    
    
    def get_simulation_forcing(self):
        route = self.vsym['mode']
        
        if route == 'train':
            forcing = self.svd_v[self.rank-1,:]
        elif route == 'test':
            forcing = self.test_mowin[:,-1]
        else:
            print('ERROR: undefined simulation behaviour')
            return 0
        
        return forcing
    
    
    def simulate_system(self, regression = None, init = 'train',
                        x0 = None, forcing = None, forcing_t = None):
        # given a trained model (SVD + regression), performs a dynamical
        # simulation using a forcing term v_r & initial condition x0
        #
        #  mode 'train'   takes init condition of train dataset
        #   "   'test'    takes init condition from last samples
        #                  of train dataset
        
        if regression is None: A, B = self.regr[0], self.regr[1]
        else: A, B = regression[0], regression[1]
        
        r = self.rank
        v_star = self.svd_v[0:r, 2:(self.svd_v.shape[1]-3)].T
        
        if init == 'train':
            x0 = v_star[0,0:r-1]
            forcing = v_star[:v_star.shape[0],r-1]
            forcing_t = self.t[:v_star.shape[0]]
            print('[sym] using train dataset mode')
            
        elif init == 'test':
            if self.test_mowin is None:
                print('[sym] WARNING: test dataset not initialized, I do it for you')
                self.compute_test_dataset()
                print('[mw] done')
                
            forcing = self.test_mowin[:,-1]  # using v_r as forcing term
            forcing_t = self.test_mowin_time[-len(forcing):]
            x0 = v_star[-1,0:r-1] # the initial condition is the last embedded state
            print('[sym] using test dataset mode')
            
        elif init == 'custom':
            print('[sym] taking custom parameters from args')
        
        sys = control.StateSpace(A, B, np.eye(r-1), 0*B)
        vsym, _, _ = cnt.lsim(sys, forcing, forcing_t, x0)
        
        self.vsym = { 'mode':init, 'vsym':vsym, 'time':forcing_t }
        return vsym, forcing_t
    
    
    def set_activity_criteria(self, f, args = {}):
        # Set an activity criteria for the forcing term.
        self.activity_criteria = f
        self.activity_criteria_args = args
    
    
    def plot_simulation(self, t0, t1):
        # Plots the simulation results from time t0 to t1.
        # Note: times are intended to be simulation time!
        
        route = self.vsym['mode']  # parse simulation type
        
        # retrieve simulation results
        ttp = self.vsym['time']
        v1_sym = self.vsym['vsym'][:,0]
        
        # take indexes of time intervals (note: simulation time!)
        idx = np.arange(int(t0/self.dt), int(t1/self.dt), dtype=int)
        
        # set behaviour depending on simulation type
        if route == 'train':
            v1_true = self.svd_v[0,:]
            v1_true_label = 'SVD'
            forcing = self.svd_v[self.rank-1,:]
        elif route == 'test':
            v1_true = self.test_mowin[:,0]
            v1_true_label = 'moving window (test dataset)'
            forcing = self.test_mowin[:,-1]
        else:
            print('ERROR: undefined simulation plot behaviour')
            return 0
        
        # do the plot, finally!
        fig, ax = plt.subplots(2, 1, figsize=(9,6))

        ax[0].plot(ttp[idx], v1_true[idx], lw=1.5, c='grey', label=v1_true_label)
        ax[0].plot(ttp[idx], v1_sym[idx], lw=1.5, c='gold', label='linear dynamics simulation')
        ax[0].set(ylabel='$v_1$')
        ax[0].tick_params(axis='y', rotation=45)
        ax[0].set_yticks([])
        ax[0].legend()
        
        if self.activity_criteria is None:
            # If there is no activity criteria, just show the forcing term.
            ax[1].plot(ttp[idx], forcing[idx], lw=1.5, c='grey', label='forcing term')
            
        else:
            # If there is an activity criteria, pass the forcing to the target function.
            # This function must return a boolean mask of true/false to code the activity
            # regions.
            active_mask = self.activity_criteria(forcing[idx], **self.activity_criteria_args)
            
            active = np.ma.masked_where(np.invert(active_mask), forcing[idx])
            unactive = np.ma.masked_where(active_mask, forcing[idx])
            
            ax[1].plot(ttp[idx], unactive, lw=1.5, c='silver', label='forcing term')
            ax[1].plot(ttp[idx], active, lw=1.5, c='red', label=' (activity)')
            
        ax[1].legend(loc='upper left')
        
        plt.xlabel('time [s]')
        plt.ylabel(r'   $v_r$         ', rotation=0)
        plt.suptitle('Forced linear dynamic system', fontweight='bold')
        plt.show()
    
    
    
    def workflow(self):
        checkpoint = 'init'
        
        self.build_Hankel()
        self.svd()
        self.set_sindy()
        self.regression()
        
        return checkpoint