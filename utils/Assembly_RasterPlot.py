# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:34:46 2023

@author: p.nazarirobati (p.nazarirobati@uleth.ca)

This script shows cell assemblys' neurons raster plot and the time that cell assembly is activated
"""

import numpy as np
import matplotlib.pyplot as plt
###################################################################

def Assembly_RasterPlot(spM, As_across_bins, Assembly_activity, t_start, t_end):

    eelements = As_across_bins['neurons']
    bin_size = As_across_bins['bin_size']
    alag = [int(xx/bin_size) for xx in As_across_bins['lags']]
    #AsemblyTimes = As_across_bins['times']
    
    AspM=[spM[xx] for xx in eelements]
    
    max_len = max([len(AspM[k]) for k in range(len(AspM))])

    #tb = np.linspace(t_start, t_end, len(Assembly_activity))
    tb = np.arange(t_start, t_end, bin_size)


    Asassembly=np.empty((len(AspM),max_len))
    Asassembly[:] = np.nan
    
    for i in range(len(AspM)):
        
        aus=np.hstack((np.zeros(alag[i]), Assembly_activity[0:len(Assembly_activity)-alag[i]]))
        activ_bins= tb[np.where(aus>0)[0]]
        #print(len(activ_bins))
        
        unit_as_spikes=[]
        
        for j in range(len(activ_bins)):
            spikes = [yy for yy in AspM[i] if yy>=activ_bins[j] and yy<=activ_bins[j]+bin_size]
            unit_as_spikes.append(spikes)
        #print(unit_as_spikes)
        
        unit_as_spikes_flatten = [item for sublist in unit_as_spikes for item in sublist]
        Asassembly[i,0:len(unit_as_spikes_flatten)]=unit_as_spikes_flatten
        #print(Asassembly[i,:])

    #### Raster plot of assembly's neurons and the time of assembly's activation
    fig, ax = plt.subplots(1,1, figsize=(10,4), facecolor='lightgray')
    
    for i in range(len(AspM)):
        aus=AspM[i]
        #print(np.isnan(aus))
        #aus[np.isnan(aus)]=[]
        #aus[np.argwhere(np.isnan(aus))] = []
        
        ax.scatter(aus,i*np.ones(len(aus)), c='blue', s=5, marker='o')
        #ax.eventplot(aus, colors='blue')
        
        aus2=Asassembly[i,:].tolist()
        #aus[np.isnan(aus)]=[];
        
        ax.scatter(aus2,i*np.ones(len(aus2)), c='red', s=5, marker='o')
        #ax.eventplot(aus2, colors='red')
        
    ax.set_yticks(range(len(eelements)))
    ax.set_yticklabels(eelements)

        
    ax.set_ylabel('neuron #')
    ax.set_xlabel('time (0.1 msec)')
        
    #ax.axvline(t_se1, linestyle='--', color='gray') 
    #ax.axvline(t_ss2, linestyle='--', color='gray')    
    ax.set_xlim([t_start - 10000, t_end + 10000])
    return fig