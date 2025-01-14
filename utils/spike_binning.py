# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:31:18 2023

@author: p.nazarirobati (p.nazarirobati@uleth.ca)
"""
import numpy as np
import pandas as pd
import os
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
import neo
import matplotlib.pyplot as plt
from parula_colorbar import parula_colorbar
from matplotlib.colors import LinearSegmentedColormap
parula_map = LinearSegmentedColormap.from_list('parula', parula_colorbar())

#### data path 
SpikePath = r'E:\Database\rr5\2011-04-17_day04\spikes'
ReachPath = r'C:\Users\p.nazarirobati\Desktop\outputs\rr5\2015-04-17\epochs.treach.txt'

#### Paramters Initialization
t_start_shift = 10000 # trial starting time respecting to reach onset (unit in 0.1 msec)
t_end_shift = 10000 # trial ending time respecting to reach onset (unit in 0.1 msec)
bin_width = 250 # bin size for binning spike data (unit in 0.1 msec)
fr_threshold = 0.75 # threshold for removing low-firing rate neurons (unit in Hz) (optional if filtering low-quality neurons would be desired)
sigma = 0.015 # s.d for gaussian kernel (in case of gaussian smoothing if required)- optional


### Gaussian Kernel

x_for_kernel = np.arange(-t_start_shift/10000, t_end_shift/10000, bin_width/10000)
kernel = np.exp(-(x_for_kernel) ** 2 / (2 * sigma ** 2))
kernel_above_thresh = kernel > 0.0001
x_within_thresh = x_for_kernel[kernel_above_thresh]
#plt.plot(x_for_kernel, kernel)
finite_kernel = kernel[kernel_above_thresh]
finite_kernel = finite_kernel / finite_kernel.sum()
kernel_n_below_0 = int((len(finite_kernel) - 1) / 2.)


######
SpkData = []
SpkData_trial = []
### loading reach information and filtering desired trials
reach_info = pd.read_csv(ReachPath, sep='\s+') # read reaching information from epoch.reach.txt file

## one of following two line must be keplt comment
#reach_info_success = reach_info[(reach_info['success']==1) & (reach_info['issingle']==1)] # consdiering success trials with single attmept
reach_info_success = reach_info[(reach_info['issingle']==1)] # removing multiple attmept trials and considering only single-attempt trials

reach_info_success = reach_info_success.reset_index()

##### loading spike data
for root, dirs, files in os.walk(SpikePath):
    
    for file in sorted(files):
        if file.startswith('cell'):
            print(file)
            with open(os.path.join(root, file), 'r') as f1: 
                spike = f1.read()
            SpkData.append(spike)
            

for i in range(len(SpkData)):
                SpkData[i] = SpkData[i].split()
                for j in range(len(SpkData[i])):
                    SpkData[i][j] = int(float(SpkData[i][j]))/100  # convert to 0.1ms time stamps

##### extracting spike data around reaching times
for i in range(reach_info_success.shape[0]):
    
    reach_onset = reach_info_success.loc[i, 'tadvance']
    spike_trial = []             
    for j in range(len(SpkData)):
        spike_cut = [xx for xx in SpkData[j] if (xx>reach_onset - t_start_shift) and (xx<reach_onset + t_end_shift)]  
        spike_trial.append(spike_cut)
    SpkData_trial.append(spike_trial) # a list(length =num_trials) containing list(length = num_neurons) of spiking times around reaching time

#### binning spike data
spt_binned = np.zeros((len(SpkData), len(np.arange(-t_start_shift, t_end_shift, bin_width)), len(SpkData_trial))) # tensor including neurons temporal spike count per trial (size num_neurons x num_time points x num_trials)
sp_fr = np.zeros((len(SpkData) , len(SpkData_trial))) # array including neurons firing rate during reaching per each trial (size num_neurons x xnum_trials)
for i in range(reach_info_success.shape[0]):
        
    t_onset = reach_info_success.loc[i, 'tadvance']
    t_binned = np.arange(t_onset - t_start_shift, t_onset + t_end_shift, bin_width)
    
    for j in range(len(SpkData)):
        
        st = neo.SpikeTrain(SpkData_trial[i][j]*pq.ms, t_start =(t_onset - t_start_shift)* pq.ms ,t_stop = (t_onset + t_end_shift)*pq.ms)
        bst = BinnedSpikeTrain(st, t_start =(t_onset - t_start_shift)* pq.ms ,t_stop = (t_onset + t_end_shift)*pq.ms, bin_size=bin_width * pq.ms)
        bst_array = bst.to_array()
        spt_binned[j,:,i] = bst_array
        spike_count = np.sum(bst_array)
        sp_fr[j,i] = spike_count/((t_end_shift + t_start_shift)/10000)
        
        
sp_fr_avg = np.mean(sp_fr, axis=1) # average firing rate across trials
### removing low firing rate neurons (if we do not want to set such filtering, just comment it)
#spt_binned = spt_binned[np.where(sp_fr_avg>fr_threshold)[0],:] # removing low firing rate neurons

#### Creating Peri-event Time Histogram
peth = np.mean(spt_binned, axis=2)
peth_norm = np.zeros((np.shape(peth)[0], np.shape(peth)[1]))
peth_norm_sorted = np.zeros((np.shape(peth)[0], np.shape(peth)[1]))

for xx in range(np.shape(peth)[0]):
    peth_norm[xx,:] = (peth[xx,:] - np.mean(peth[xx,:]))/(np.std(peth[xx,:])+0.000000000000000001)

### creating template of spiking activity
peth_max_id = np.zeros((np.shape(peth_norm)[0],2))

for i in range(np.shape(peth_norm)[0]):
    max_sp = np.argmax(peth_norm[i,:])
    peth_max_id[i,0] = i
    peth_max_id[i,1] = max_sp

peth_max_id_sorted = peth_max_id[peth_max_id[:, 1].argsort(),:]

for i in range(np.shape(peth_norm)[0]):
    peth_norm_sorted[i,:] = peth_norm[int(peth_max_id_sorted[i,0]),:]
    #peth_norm_sorted[i,:] = np.convolve(peth_norm_sorted[i,:], finite_kernel, mode='same')


### Plot PETH results
ticks1 = np.linspace(-t_start_shift/10000, t_end_shift/10000, len(np.arange(-t_start_shift, t_end_shift, bin_width)))
ticks2 = np.arange(np.shape(spt_binned)[0])

fig, ax = plt.subplots(1,1, figsize =(4,4))
im=ax.pcolormesh(ticks1, ticks2,peth_norm_sorted, cmap=parula_map, vmin=-.5, vmax=2)    

ax.set_yticks(np.arange(0,np.shape(peth)[0],5), minor=False)
ax.set_yticklabels(np.arange(np.shape(peth)[0],0,-5), minor=False)

ax.set_xlabel('Time from reach onset (msec)', fontsize = 10)
ax.xaxis.labelpad = -2
ax.set_ylabel('Unit #', fontsize = 10)
ax.yaxis.labelpad = 0
ax.patch.set_edgecolor('black')  

ax.patch.set_linewidth('1.5') 
ax.set_title('Peri-Event Time Histogram (Bin_size=' + str(int(bin_width/10)) + 'msec)', fontweight = 'bold', fontsize = 8)

cbar = plt.colorbar(im,  pad = 0.005)
cbar.set_label('average spike count (z-scored)', rotation=90, fontsize = 10)
ax.axvline(x=0, color='lightgray', linestyle = '--')
plt.show()
        
