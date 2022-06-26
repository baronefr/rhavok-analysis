
###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python module implements some more generic functions which are
#  wrapped by the library.
#
#  coder: Barone Francesco, last edit: 2 June 2022
#--------------------------------------------------------------------------
#  Open Access licence
#
#  reference paper: S. Brunton et al, 2017,
#                   Chaos as an intermittently forced linear system
#--------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


cmap = plt.get_cmap('gist_rainbow')

class externals:
    
    # plot embedded coordinates as spectrum
    def embedded_spectrum(x, t, v,             # data
                      limit, selection = None,      # what to plot
                      vf = lambda x: x,             # transformation function for v
                      figsize = (10,9), alpha = 1, ylim = [-.025,.025]): # aesthetics
        
        limit = np.arange(limit[0], limit[1])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1,2]}, figsize=figsize, sharex=True)
        ax1.plot(t[limit], x[limit], c='k')
        ax1.set_ylabel('x(t)')
        
        if selection is None:
            selection = np.arange(1,v.shape[1])
        elif isinstance(selection, int):
            if selection == -1:
                v = np.expand_dims(v, axis = 1)
                selection = [1]
            else: selection = np.arange(1,selection+1)
        elif isinstance(selection, tuple):
            selection = np.arange(selection[0]-1,selection[1])
        elif isinstance(selection, list):
            pass
        selection = list(selection)
        
        
        ax2.set_prop_cycle( cycler( color=[cmap(1.*i/len(selection)) for i in range(len(selection))] ))
        
        for ts in selection:
            ax2.plot(t[limit], vf(v[limit,ts-1]), label=str(ts+1), alpha=alpha)
        
        ax2.set_ylabel(r"$v_i(t)$")
        ax2.set_ylim(ylim)
        ax2.legend()
        return fig