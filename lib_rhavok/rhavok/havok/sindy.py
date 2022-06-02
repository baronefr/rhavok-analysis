
###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python module contains rountines to perform SINDy regression.
#
#  coder: Barone Francesco, last edit: 31 may 2022
#--------------------------------------------------------------------------
#  Open Access licence
#
#  reference paper: S. Brunton et al, 2017,
#                   Chaos as an intermittently forced linear system
#--------------------------------------------------------------------------

import numpy as np

def SINDy_linear_library(yin, nVars = None):
    
    #  Dev Note: this implementation is not complete! It allows to generate only
    # degree-one polynomial Theta. Further implementations might require higher order Poly!
    n = yin.shape[0]
    if nVars is None: nVars = yin.shape[1]
    yout = np.zeros( (n,1+nVars) )  # valid to poly order 1
    
    # poly order 0 (add bias term)
    yout[:,0] = np.ones((n))
    
    idx = 1
    # poly order 1
    for i in range(0, nVars):
        yout[:,idx] = yin[:,i]
        idx = idx+1
        
    # omitting higher poly order
    
    return yout



def sequential_threshold_least_squares(Theta, dXdt, thres, niter = 10, nleast=1):
    # SINDy optimization algorithm: sequential threshold least squares
    
    # initial guess: least squares
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
    # ref https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    
    for k in range(0, niter):       # repeat several times the thresholding
        mask_under_thres = (abs(Xi)<thres)
        Xi[mask_under_thres] = 0
        for j in range(0, nleast):  # repeat several times the least squares
            mask_above_thres = np.invert(mask_under_thres)
            # do again least squares
            Xi[mask_above_thres] = np.linalg.lstsq(Theta[:,mask_above_thres], 
                                                   dXdt, rcond=None )[0] 
    return Xi