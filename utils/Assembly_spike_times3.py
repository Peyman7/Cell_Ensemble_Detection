# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:07:12 2023

@author: p.nazarirobati
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:44:18 2023

@author: p.nazarirobati
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:32:49 2023

@author: p.nazarirobati
"""

import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt, patches
from spiketrain import spiketrain
from epochs_data import epochs_data
from assemblies_data import assemblies_data
from assemblies_across_bins import assemblies_across_bins
from prunning_across_bins import prunning_across_bins
from elephant.conversion import BinnedSpikeTrain
import neo
import quantities as pq
import time

############
def Assembly_First_neuron_Spike_time(as_across_bins_cut, t_start, t_end, Spikes, Method):
      
    nneu = len(Spikes)
    for idx in range(len(as_across_bins_cut)):
            
        Assembly = as_across_bins_cut[idx]
        Assmbly_time_cut = [xx for xx in Assembly['times'] if xx>=t_start and xx<=t_end]
            
        if Method == 'first neuron':
            first_neuron_SPt = []
            #print(len(set(Assembly['lags'])))
            ### if assembly type is sequential
            if len(set(Assembly['lags']))>1:
                
                #print(len(set(Assembly['lags'])))
                neuron1 = Assembly['neurons'][0]
                
                Spikes_cut = np.array([xx for xx in Spikes[neuron1] if xx>=t_start and xx<=t_end])
                
               
                for tt in range(len(Assmbly_time_cut)):
                    #spt = [qq for qq in Spikes_cut if qq>=Assmbly_time_cut[tt] and qq<=Assmbly_time_cut[tt]+Bin]
                    spt_arg = Spikes_cut[(Spikes_cut>=Assmbly_time_cut[tt]) & (Spikes_cut<=Assmbly_time_cut[tt]+Bin)]

                    if len(spt_arg)==0:
                        print ("no spike found!")
                        first_neuron_SPt.append([])
                    else:
                        first_neuron_SPt.append(spt_arg[0])
            ### if assembly type is synchronous

            elif len(set(Assembly['lags']))==1:
                #print(len(set(Assembly['lags'])))
                
                #print('neuron id: ', Assembly['neurons'])

                for tt in range(len(Assmbly_time_cut)):
                    
                    spt_neurons = []
                    spt_arg = []
                    ### loop over neurons of assembly
                    for nn in Assembly['neurons']:
                       
                        #print('neuron id: ', nn)
                        Spikes_cut = np.array([xx for xx in Spikes[nn] if xx>=t_start and xx<=t_end])
                        #print(len(Spikes_cut))
                        spt_arg = Spikes_cut[(Spikes_cut>=Assmbly_time_cut[tt]) & (Spikes_cut<=Assmbly_time_cut[tt]+Bin)]
                        #print(spt_arg)
                        #spt = np.where(Spikes_cut>=Assmbly_time_cut[tt] and Spikes_cut<=Assmbly_time_cut[tt] + Bin)
                        
                        #spt = [qq for qq in Spikes_cut if qq>=Assmbly_time_cut[tt] and qq<=Assmbly_time_cut[tt]+Bin]
                        
                        if len(spt_arg)==0:
                            print ("no spike found!")
                      
                        else:
                            spt_neurons.append(spt_arg[0])
                    #print(spt_neurons)
                    #print(spt_neurons)
                    first_neuron_SPt.append(min(spt_neurons))
                print(first_neuron_SPt)
         
                        
            as_across_bins_cut[idx]['neuron1_spk'] = first_neuron_SPt
            as_across_bins_cut[idx]['times'] = Assmbly_time_cut

        
    return as_across_bins_cut


################# MAIN CODE RUNNING ################
BinSizes = [30, 50, 100, 250, 350, 500, 650, 750, 900, 1000] # 0.1ms unit
Bin = BinSizes[9]
temporal_resolution = 5 # unit in 0.1 msec
path_name = r"C:\Users\p.nazarirobati\Desktop\outputs\rr5\2015-04-15" 

spt_file = r"C:\Users\p.nazarirobati\Desktop\rr5\2.pkl"
with open(spt_file, 'rb') as f:
    spt_data = pickle.load(f)   # SpikeTrain Data

path_name2 = r"E:\assemblies_Russo\rr5\2015-04-15"
patterns = assemblies_data(path_name2)

Spikes = spiketrain(spt_file)

t_start = epochs_data(path_name)[4]
t_end = epochs_data(path_name)[5]

########

nneu = len(Spikes) # number of neurons
as_across_bins, as_across_bins_index = assemblies_across_bins(patterns, BinSizes)
as_across_bins_cut = [as_across_bins[xx] for xx in range(len(as_across_bins)) if as_across_bins[xx]['bin_size']==Bin]
print('total number of detected assemblies in Bin size ' + str(Bin/10) + ' ms:', len(as_across_bins_cut))

tic = time.time()
as_across_bins_cut2 = Assembly_First_neuron_Spike_time (as_across_bins_cut, t_start, t_end, Spikes, Method='first neuron')

toc = time.time()

total_running_time = toc - tic
x=g666
#### save results
filename = 'CAD_nTimes_rr5_2015-04_15_b1000.pkl'
with open (filename, 'wb') as ff:
    pickle.dump(as_across_bins_cut2, ff)





        
                
        
        
    
    

    
