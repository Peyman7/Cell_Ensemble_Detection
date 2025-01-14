# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:25:45 2022

@author: p.nazarirobati
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:35:24 2022

@author: p.nazarirobati

### This script calculates CrossCorrelation Histogram between a pair of spike trains
### Method for CrossCorrelation calculation: direct, mode: same

Inputs:
    Spikes1: spiketrain1 (list containing spiketrain1 firing), Unit must be in 0.1 ms
    Spikes2: spiketrain2 (list containing spiketrain1 firing), Unit must be in 0.1 ms
    bin_size: bin size for binning spiketrains (unit in 0.1 ms)
    t_start: session start time in 0.1 ms
    t_end: session end time in 0.1 ms
    lag: time lag for showing CrossCorrelation Histogram
    plot: if True, it returns the figure of CrossCorrelation

Outputs:
    Cross_corr_lag: histogram of CrossCorrelation in (-lag, lag) time
    peak_CCH: time of CrossCorrelation Histogram peak
    median_CCH: time of CrossCorrelation Histogram median

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elephant.conversion import BinnedSpikeTrain
import elephant.conversion as conv
import neo
import quantities as pq
from scipy import signal

##############

def CrossCorrelation_Histogram(Spikes1, Spikes2, Bin_size, t_start, t_end, lag, plot):
    
    if Spikes1 == Spikes2:
        
        Spikes = neo.SpikeTrain(Spikes1*pq.ms, t_start = t_start * pq.ms, t_stop = t_end * pq.ms)
        Qmatrix = conv.BinnedSpikeTrain(Spikes, bin_size=Bin_size * pq.ms, t_start = t_start * pq.ms, t_stop = t_end * pq.ms)
        Qmatrix_arr = Qmatrix.to_array()
        Bin_edges = Qmatrix.bin_edges

        sp1 = np.zeros((1,Qmatrix_arr[0].size))
        
        for s in range(sp1.size):
            sp1[0,s] = Qmatrix_arr[0,s]
        
        Cross_corr = signal.correlate(sp1[0], sp1[0], mode='same')
        Cross_corr_lag = Cross_corr[np.argmax(Cross_corr)-lag:np.argmax(Cross_corr) + lag+1]
        
        Cross_corr_lag[np.argmax(Cross_corr_lag)]=0
        
       # Cross_corr_histogram[0,0:lag]= Cross_corr_lag[0:lag]/[xx for xx in np.arange(lag,0,-1)]
       # Cross_corr_histogram[0,lag+1:2*lag+1]= Cross_corr_lag[lag+1:2*lag+1]/[xx for xx in np.arange(1,lag+1,1)]
    
    else:
        Spikes1 = neo.SpikeTrain(Spikes1*pq.ms, t_start = t_start * pq.ms, t_stop = t_end * pq.ms)
        Spikes2 = neo.SpikeTrain(Spikes2*pq.ms, t_start = t_start * pq.ms, t_stop = t_end * pq.ms)

        Qmatrix1 = conv.BinnedSpikeTrain(Spikes1, bin_size=Bin_size * pq.ms, t_start = t_start * pq.ms, t_stop = t_end * pq.ms)
        Qmatrix2 = conv.BinnedSpikeTrain(Spikes2, bin_size=Bin_size * pq.ms, t_start = t_start * pq.ms, t_stop = t_end * pq.ms)

        Qmatrix1_arr = Qmatrix1.to_array()
        Qmatrix2_arr = Qmatrix2.to_array()

        sp1 = np.zeros((1,Qmatrix1_arr[0].size))
        sp2 = np.zeros((1,Qmatrix1_arr[0].size))
        
        Cross_corr = signal.correlate(sp1[0], sp2[0], mode='same')
        
       # Cross_corr_histogram[0,0:lag]= Cross_corr_lag[0:lag]/[xx for xx in np.arange(lag,0,-1)]
       # Cross_corr_histogram[0,lag+1:2*lag+1]= Cross_corr_lag[lag+1:2*lag+1]/[xx for xx in np.arange(1,lag+1,1)]

        
        for s in range(sp1.size):
            sp1[0,s] = Qmatrix1_arr[0,s]
            
        for s in range(sp2.size):
            sp2[0,s] = Qmatrix2_arr[0,s]



    if plot==True:
        ### Plot Autocorrelation Historam
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.bar(np.arange(-lag,lag+1), Cross_corr_lag/100)
        #ax[1].bar(np.arange(-50,51), Cross_corr_histogram[0,:])
        ax.set_xlabel('msec (1 msec binsize)')
        ax.set_ylabel('Rate')
        fig.suptitle('Autocorrelation')

    #### CrossCorrelation Peak and Median location
    peak_CCH = np.argmax(Cross_corr_lag[lag+1:2*lag+1])
    median_CCH = (Cross_corr_lag[lag+1:2*lag+1].tolist()).index(np.percentile(Cross_corr_lag[lag+1:2*lag+1].tolist(),50,interpolation='nearest'))

    
    return Cross_corr_lag, peak_CCH, median_CCH



