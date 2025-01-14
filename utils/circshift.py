# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:20:40 2023

@author: p.nazarirobati (p.nazarirobati@leth.ca)

### finding reach-modulated cell ensembles around reaching time (Karimi et al.2023, Jadhav et al. 2016)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from spiketrain import spiketrain
from epochs_data import epochs_data
from assemblies_data import assemblies_data
from assemblies_across_bins import assemblies_across_bins
from restyle_assembly_lags_time import restyle_assembly_lags_time
from prunning_across_bins import prunning_across_bins
from epochs_coloring import epochs_coloring
from matplotlib.colors import LinearSegmentedColormap
from parula_colorbar import parula_colorbar
from AssemblyActivity import AssemblyActivity
from cheetah_conversion import convert_cheetah_index
from cheetah_conversion import convert_cheetah_index2
import multiprocessing
from epochs_coloring import epochs_coloring
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score

##############


def AssemblyActivityEpochs(Assembly_activity, epochs):
    
    epochs_activity = np.empty((Assembly_activity.shape[0],1))
    epochs_activity[:] = np.nan
    for ii in range(epochs.shape[0]):
        
        #Activity_epoch = np.sum(Assembly_activity[:,int(epochs[ii,0]):int(epochs[ii,1])], axis=1)
        epochs_activity = np.hstack((epochs_activity,Assembly_activity[:,int(epochs[ii,0]):int(epochs[ii,1])]))

        #epochs_activity = epochs_activity+ Activity_epoch
    epochs_activity = np.delete(epochs_activity, 0, axis=1)    
    return epochs_activity
    
def AssemblyActivityEvent(Assembly_activity, epochs, shift1, shift2, bin_size, t_zero):
    
    #epochs_inter_interval = [epochs[i,1] - epochs[i-1,0] for i in range(1,epochs.shape[0])]
    
    event_activity = np.zeros((Assembly_activity.shape[0], int((shift1+shift2)/bin_size), epochs.shape[0]))
    print(event_activity.shape)
    
    for ii in range(epochs.shape[0]):
        
        if t_zero =='center':
            
            t_center = int((epochs[ii,0] + epochs[ii,1])/2)
        else:
            t_center = int(epochs[ii,0])
        
        
        if (t_center -int(shift1/bin_size)>0) and (t_center +int(shift2/bin_size)<Assembly_activity.shape[1]):
            
            event_activity[:,:,ii] = Assembly_activity[:,t_center - int(shift1/bin_size):t_center + int(shift2/bin_size)]
    
    return event_activity


def PETH_shuffled(peri_reach, max_shift, num_shuffling):
    

    sig_shuffled = np.zeros((peri_reach.shape[0], peri_reach.shape[1], num_shuffling))  # array including assemblies-shuffled condition  

    for perm_counter in range(num_shuffling):

        shift = np.round(4*max_shift*(np.random.rand(peri_reach.shape[2]) - .5),2) # amount of shift (up to two times of max-shift)
        shift = shift/freq
    
        sig_template = np.zeros(np.shape(peri_reach))
    
        for reach_counter in range(peri_reach.shape[2]):
        
            sig_template[:,:,reach_counter] = np.roll(peri_reach[:,:,reach_counter], int(shift[reach_counter]), axis=1) # randomly circular shifting
        
        sig_shuffled[:,:, perm_counter] = np.mean(sig_template, axis=2)  ## averaged across reach trials
        
    return sig_shuffled



########################################    
######### MAIN CODE ####################
########################################

### fixed parameters
BinSizes = [30, 50, 100, 250, 350, 500, 650, 750, 900, 1000] # 0.1ms unit

### paramters intialization
bin_size = BinSizes[4]
significant_threshold = 95
q_treshold = 0.9 # percentile to remove assemblies with total activation above it (all-time activated CAs)

### paramters intializations for files loading
day= '2015-04-16'
day_id = 3

####### loading files
path_name = r"E:\assemblies_Russo\rr5\\" + str(day)
spt_file = r"C:\Users\p.nazarirobati\Desktop\rr5\\" + str(day_id) + ".pkl" 
reach_info = pd.read_csv(r'E:\assemblies_Russo\rr5\\'+ str(day)+ '\epochs.treach.txt', sep='\s+')

reach_info = reach_info[reach_info['issingle']==1]
reach_info = reach_info[reach_info['treturn'] - reach_info['tadvance']>0]
spM = spiketrain(spt_file) # spike data (unbinned)

patterns = assemblies_data(path_name) # list of detected cell assemblies (Russo, 2017 main output)

freq = bin_size/10000 # CA activation resolution 
nneu = len(spM) # number of neurons


### reforming patterns based on As_Across_bins 
As_across_bins, As_across_bins_index = assemblies_across_bins(patterns, BinSizes)
#As_across_bins_pr, As_across_bins_pr_index = prunning_across_bins(As_across_bins, As_across_bins_index, nneu, criteria='biggest', th=0.7, style='pvalue')
print('total number of detected assemblies: ', len(As_across_bins))  

# picking assemblies detected in bin_size
As_across_bins_cut = [As_across_bins[xx] for xx in range(len(As_across_bins)) if As_across_bins[xx]['bin_size']==bin_size]

#### removing highly activated assemblies if total activation is more than 90th percentile

count_dist = np.array([len(As_across_bins_cut[xx]['times']) for xx in range(len(As_across_bins_cut))])
count_distQuantile = np.quantile(count_dist, q=q_treshold)
As_across_bins_cut = [As_across_bins_cut[xx] for xx in range(len(As_across_bins_cut)) if len(As_across_bins_cut[xx]['times'])<count_distQuantile]
#As_across_bins_pr, As_across_bins_index_pr  = prunning_across_bins(As_across_bins_cut, As_across_bins_index, nneu, criteria='distance', th=0.7, style='pvalue')

for jj in range(len(As_across_bins_cut)):
    
    if len(set(As_across_bins_cut[jj]['lags']))==1:
        As_across_bins_cut[jj]['neurons'] = sorted(As_across_bins_cut[jj]['neurons'])


### reading epoch timestamps data
epochs_ts = epochs_data(path_name)

t_start, t_end = epochs_ts['session'][0], epochs_ts['session'][1]
t_ss1, t_se1 = epochs_ts['sleep1'][0], epochs_ts['sleep1'][1] 
t_ss2, t_se2 = epochs_ts['sleep2'][0], epochs_ts['sleep2'][1] 
t_tks, t_tke = epochs_ts['task'][0], epochs_ts['task'][1] 

t_rem1 = epochs_ts['t_rem1']
t_rem2 = epochs_ts['t_rem2']
t_sws1 = epochs_ts['t_sws1']
t_sws2 = epochs_ts['t_sws2']

t_sleep1 = epochs_ts['t_sleep1']
t_sleep2 = epochs_ts['t_sleep2']



t_reaching = np.array(reach_info.iloc[:,3:6]) # reaching times stamps info

####################### converting epoch timestamps info from cheetah to vector starting from zero

tb = np.arange(t_start, t_end, BinSizes[9])


s1s_idx = np.where(tb>=t_ss1)[0][0]
s1e_idx = np.where(tb<=t_se1)[0][-1]

s2s_idx = np.where(tb>=t_ss2)[0][0]
s2e_idx = np.where(tb<=t_se2)[0][-1]

s1s_idx = np.where(tb>=t_ss1)[0][0]
s1e_idx = np.where(tb<=t_se1)[0][-1]

s2s_idx = np.where(tb>=t_ss2)[0][0]
s2e_idx = np.where(tb<=t_se2)[0][-1]

sws_s1 = convert_cheetah_index(tb, t_sws1)
sws_s2 = convert_cheetah_index(tb, t_sws2)

rem_s1 = convert_cheetah_index(tb, t_rem1)
rem_s2 = convert_cheetah_index(tb, t_rem2)

sleep1 = convert_cheetah_index(tb, t_sleep1)
sleep2 = convert_cheetah_index(tb, t_sleep2)


reach_sidx = np.where(tb>=t_tks)[0][0]
reach_eidx = np.where(tb<=t_tke)[0][-1]


reaching_idx = convert_cheetah_index2(tb, t_reaching)

### Assembly Activity across recording time

Assembly_activity = np.array(AssemblyActivity(As_across_bins_cut, spM, BinSizes, t_start, t_end, method ='full', LagChoice='begining'))
######################################################################################################
################### Finding reach-modulated assemblies (Karimi et.al 2023, Jadhav et al. 2016) #######
#### reaching analysis

num_trials = reaching_idx.shape[0]
pre = 2 # window osize before event onset (for shuffled condition)
post = 2 # window osize before event onset (for shuffled condition)

before_event = 0.5 # window size before event onset (for real condition)
after_event = 1 # window size after event onset (for real condition)

t_base = 1

num_shuffling = 1000 # number of permutation 
max_shift = 1

### Creating Peri-Reach time histogram (PETH)
peri_reach = np.zeros((Assembly_activity.shape[0], int((pre+post)/freq), num_trials))

midpoint = round(peri_reach.shape[1]/2)

for jj in range(num_trials):
    
    tpre = int(reaching_idx[jj,1] - int(pre/freq))
    tpost = int(reaching_idx[jj,1] + int(post/freq))

    peri_reach[:,:,jj] = Assembly_activity[:, tpre:tpost]
    
### shuffling PETHs structure using CIRCULAR SHIFT ######
#np.random.seed(42)    
#rng = np.random.default_rng(seed=0)
rng = np.random.default_rng(seed=0)

sig_shuffled = np.zeros((peri_reach.shape[0], peri_reach.shape[1], num_shuffling))  # array including assemblies-shuffled condition  
       
for perm_counter in range(num_shuffling):
    
    sig_template = np.zeros(np.shape(peri_reach))
    
    shift = np.round(4*max_shift*(rng.random(peri_reach.shape[2]) - .5),2)
    #print(np.mean(shift))
    shift = np.round(shift/freq)
    
    for reach_counter in range(peri_reach.shape[2]):
        sig_template[:,:,reach_counter] = np.roll(peri_reach[:,:,reach_counter], int(shift[reach_counter]), axis=1)
        
    sig_shuffled[:,:,perm_counter] = np.mean(sig_template, axis=2)    
    
sig = np.mean(peri_reach,axis=2)
#sig = sig[:,midpoint - round(before_event/freq):midpoint + round(before_event/freq)]

sig_after = sig[:, midpoint: midpoint + round(after_event/freq)]
sig_before = sig[:, midpoint - round(before_event/freq): midpoint]

#sig_shuffled = sig_shuffled[:,midpoint - round(before_event/freq):midpoint+round(before_event/freq), :]

sig_shuffled_after = sig_shuffled[:, midpoint: midpoint + round(after_event/freq)]
sig_shuffled_before = sig_shuffled[:, midpoint - round(before_event/freq): midpoint]
# creating a baseline to find positive/ negative modulation


#b1 = np.median(np.mean(peri_reach[:,0:midpoint - round((before_event)/freq)], axis=2), axis=1)
#b2 = np.median(np.mean(peri_reach[:,midpoint + round((after_event)/freq):peri_reach.shape[1]], axis=2), axis=1)

#b1 = np.median(np.mean(peri_reach[:,midpoint - round(before_event/freq) - round(t_base/freq):midpoint - round(before_event/freq),:], axis=2), axis=1)
#b2 = np.median(np.mean(peri_reach[:,midpoint + round(after_event/freq):midpoint + round(after_event/freq) + round(t_base/freq),:], axis=2), axis=1)

b1 = np.median(np.mean(peri_reach[:,midpoint - round(t_base/freq):midpoint,:], axis=2), axis=1)
b2 = np.median(np.mean(peri_reach[:,midpoint + round((after_event)/freq):midpoint + round((after_event)/freq) + round(t_base/freq),:], axis=2), axis=1)

#b1 = np.median(np.mean(peri_reach[:,midpoint - round(t_base/freq):midpoint,:], axis=2), axis=1)

baseline = (b1 + b2)/2
#baseline = b1

modulation_metric_after = np.mean((sig_after - np.mean(sig_shuffled_after,2))**2,1)  ## euclidean distance
modulation_metric_before = np.mean((sig_before - np.mean(sig_shuffled_before,2))**2,1)  ## euclidean distance

modulation_metric_shuffled_after = np.array([sig_shuffled_after[:,:,i] - np.mean(sig_shuffled_after, axis=2) for i in range(sig_shuffled_after.shape[2])])
modulation_metric_shuffled_after = np.reshape(modulation_metric_shuffled_after,np.shape(sig_shuffled_after))
modulation_metric_shuffled_after = np.mean(modulation_metric_shuffled_after**2, axis=1)

modulation_flag_after = modulation_metric_after > np.percentile(modulation_metric_shuffled_after,significant_threshold,1)

positive_flag_after = np.logical_and((np.mean(sig_after,axis=1) > baseline),  modulation_flag_after)
negative_flag_after = np.logical_and((np.mean(sig_after,axis=1) < baseline),  modulation_flag_after)



modulation_metric_shuffled_before = np.array([sig_shuffled_before[:,:,i] - np.mean(sig_shuffled_before, axis=2) for i in range(sig_shuffled_before.shape[2])])
modulation_metric_shuffled_before = np.reshape(sig_shuffled_before,np.shape(sig_shuffled_before))
modulation_metric_shuffled_before = np.mean(sig_shuffled_before**2, axis=1)

modulation_flag_before = modulation_metric_before > np.percentile(modulation_metric_shuffled_before,significant_threshold,1)

positive_flag_before = np.logical_and((np.mean(sig_before,axis=1) > baseline),  modulation_flag_before)
negative_flag_before = np.logical_and((np.mean(sig_before,axis=1) < baseline),  modulation_flag_before)


positive_correlated_after = np.where(positive_flag_after==True)[0]
negative_correlated_after = np.where(negative_flag_after==True)[0]

positive_correlated_before = np.where(positive_flag_before==True)[0]
negative_correlated_before = np.where(negative_flag_before==True)[0]




fig, ax = plt.subplots(1,4, figsize=(16,4), sharex=True, facecolor='lightgray', sharey=False)
axes = [ax[0], ax[0].twinx()]

axes[-1].set_frame_on(True)
axes[-1].patch.set_visible(False)

                
im1 = axes[0].imshow(np.mean(peri_reach[positive_correlated_after,:,:], axis=2), cmap='jet', aspect='auto') 
axes[1].plot(np.mean(np.mean(peri_reach[positive_correlated_after,:,:], axis=2),axis=0), color='white')
axes[1].set_ylim([0,.5])

ls_loc = list(np.arange(0,40,5))
axes[0].set_xticks(ls_loc)

lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)

axes = [ax[1], ax[1].twinx()]

im2 = axes[0].imshow(np.mean(peri_reach[negative_correlated_after,:,:], axis=2), cmap='jet', aspect='auto') 
axes[1].plot(np.mean(np.mean(peri_reach[negative_correlated_after,:,:], axis=2),axis=0), color='white')
axes[1].set_ylim([0,.5])

ls_loc = list(np.arange(0,40,5))
axes[0].set_xticks(ls_loc)

lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)

axes = [ax[2], ax[2].twinx()]

axes[0].set_xlabel('Time from grasp (sec)')
im3 = axes[0].imshow(np.mean(peri_reach[positive_correlated_before,:,:], axis=2), cmap='jet',  aspect='auto') 

axes[1].plot(np.mean(np.mean(peri_reach[positive_correlated_before,:,:], axis=2),axis=0), color='white')
axes[1].set_ylim([0,.5])


ls_loc = list(np.arange(0,40,5))
axes[0].set_xticks(ls_loc)

lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)


axes = [ax[3], ax[3].twinx()]

im4 = axes[0].imshow(np.mean(peri_reach[negative_correlated_before,:,:], axis=2), cmap='jet', aspect='auto') 
axes[1].plot(np.mean(np.mean(peri_reach[negative_correlated_before,:,:], axis=2),axis=0), color='white')
axes[1].set_ylim([0,.5])

ls_loc = list(np.arange(0,40,5))
axes[0].set_xticks(ls_loc)

lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)


axes[1].set_yticklabels([])


divider = make_axes_locatable(axes[0])

cax1 = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(im1, cax=cax1)

x=g6666


reach_correlated_CA_idx = np.where(positive_flag==True)[0]
reach_decorrelated_CA_idx = np.where(negative_flag==True)[0]

reach_modulated_idx = np.hstack([reach_correlated_CA_idx, reach_decorrelated_CA_idx])


#reach_modulated_idx = np.where(modulation_flag==True)[0]

reach_unmodulated_idx = np.setdiff1d(np.arange(modulation_flag.size), reach_modulated_idx)

### reach modulated assemblies (in as_across_bins format)
As_across_bins_modulated = [As_across_bins_cut[xx] for xx in list(reach_modulated_idx)]
As_across_bins_unmodulated = [As_across_bins_cut[xx] for xx in list(reach_unmodulated_idx)]

##############################################################################################
##############################################################################################

#km = TimeSeriesKMeans(n_clusters=4, metric='euclidean', max_iter_barycenter=100, init='k-means++', random_state =np.random.seed(seed=42))  # Similarity Measure: DTW/ EUCLIDEAN

#labels = km.fit_predict(np.mean(peri_reach[reach_modulated_idx,:,:], axis=2)) # list of time series labels after clsutering
#intertia = km.inertia_
#ls_silhoutte = []

#for j in range(2,11):
#    km = TimeSeriesKMeans(n_clusters=j, metric='euclidean', max_iter_barycenter=100, init='k-means++', random_state =np.random.seed(seed=42))  # Similarity Measure: DTW/ EUCLIDEAN
#    labels = km.fit_predict(np.mean(peri_reach[reach_modulated_idx,:,:], axis=2)) # list of time series labels after clsutering
#    sscore = silhouette_score(np.mean(peri_reach[reach_modulated_idx,:,:], axis=2), labels, metric='euclidean') # silhoutte score
#    ls_silhoutte.append(sscore)



#x=g6666
#############################
fig, ax = plt.subplots(1,3, figsize=(16,4), sharex=True, facecolor='lightgray', sharey=False)
axes = [ax[0], ax[0].twinx()]

axes[-1].set_frame_on(True)
axes[-1].patch.set_visible(False)

max_v = max(np.max(np.mean(peri_reach[reach_correlated_CA_idx,:,:])), np.max(np.mean(peri_reach[reach_decorrelated_CA_idx,:,:])))
                
im1 = axes[0].imshow(np.mean(peri_reach[reach_correlated_CA_idx,:,:], axis=2), cmap='jet', aspect='auto', vmax = max_v+0.01) 
axes[1].plot(np.mean(np.mean(peri_reach[reach_correlated_CA_idx,:,:], axis=2),axis=0), color='white')
axes[1].set_ylim([0,.5])

ls_loc = list(np.arange(0,40,5))
axes[0].set_xticks(ls_loc)

lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)

axes = [ax[1], ax[1].twinx()]

im2 = axes[0].imshow(np.mean(peri_reach[reach_decorrelated_CA_idx,:,:], axis=2), cmap='jet', aspect='auto', vmax = max_v+0.01) 
axes[1].plot(np.mean(np.mean(peri_reach[reach_decorrelated_CA_idx,:,:], axis=2),axis=0), color='white')
axes[1].set_ylim([0,.5])

ls_loc = list(np.arange(0,40,5))
axes[0].set_xticks(ls_loc)

lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)

axes = [ax[2], ax[2].twinx()]

im3 = axes[0].imshow(np.mean(peri_reach[reach_unmodulated_idx,:,:], axis=2), cmap='jet',  aspect='auto', vmax = max_v+0.01) 

axes[1].plot(np.mean(np.mean(peri_reach[reach_unmodulated_idx,:,:], axis=2),axis=0), color='white')
axes[1].set_ylim([0,.5])


ls_loc = list(np.arange(0,40,5))
axes[0].set_xticks(ls_loc)

lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)

axes[1].set_yticklabels([])

divider = make_axes_locatable(axes[0])

cax1 = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(im1, cax=cax1)

#cbar = plt.colorbar(im1, ax=axes[0], orientation='vertical', pad=0.15)
#cbar.set_label('(Re)activation strength', rotation=90, labelpad = 15)


ax[0].set_title('Reach-correlated assemblies')
ax[1].set_title('Reach-decorrelated assemblies')
ax[2].set_title('Reach-uncorrelated assemblies')

ax[1].set_xlabel('Time from reach onset (sec)')
ax[0].set_ylabel('CA#')
#filename1 = 'rr5_' + str(day) + '_reachCorrelated.pkl'
#filename2 = 'rr5_' + str(day) + '_reachDecorrelated.pkl'
#filename3 = 'rr5_' + str(day) + '_reachUnmodulated.pkl'




##############################################################
##############################################################
######## activity of reach-modulated CAs during sleep1 & sleep2

## activity across pre/post sleep recording
Activity_preM = Assembly_activity[reach_correlated_CA_idx,s1s_idx:s1e_idx]
Activity_postM = Assembly_activity[reach_correlated_CA_idx,s2s_idx:s2e_idx]


Activity_preUM = Assembly_activity[reach_unmodulated_idx,s1s_idx:s1e_idx]
Activity_postUM = Assembly_activity[reach_unmodulated_idx,s2s_idx:s2e_idx]


## activity across sleep epochs during pre/post training sleep
Activity_s1M = AssemblyActivityEpochs(Assembly_activity[reach_correlated_CA_idx,:], sleep1)
Activity_s2M = AssemblyActivityEpochs(Assembly_activity[reach_correlated_CA_idx,:], sleep2)

Activity_s1UM = AssemblyActivityEpochs(Assembly_activity[reach_unmodulated_idx,:], sleep1)
Activity_s2UM = AssemblyActivityEpochs(Assembly_activity[reach_unmodulated_idx,:], sleep2)
##

Avg_s1M = np.mean(Activity_s1M, axis=1)
Avg_s2M = np.mean(Activity_s2M, axis=1)


Avg_s1UM = np.mean(Activity_s1UM, axis=1)
Avg_s2UM = np.mean(Activity_s2UM, axis=1)












## smoothing signal
window = 5
Num_point = int((window*60)/(BinSizes[9]/10000))

sleep1_bin = np.reshape(Activity_preM[:,0:int(Activity_preM.shape[1]/Num_point)*Num_point], (Activity_preM.shape[0], Num_point, -1))
sleep1_binAvg = np.mean(sleep1_bin, axis=1)

sleep1um_bin = np.reshape(Activity_preUM[:,0:int(Activity_preUM.shape[1]/Num_point)*Num_point], (Activity_preUM.shape[0], Num_point, -1))
sleep1um_binAvg = np.mean(sleep1um_bin, axis=1)


s1Avg = np.mean(sleep1_binAvg, axis=0)
s1Std = np.std(sleep1_binAvg, axis=0)

s1umAvg = np.mean(sleep1um_binAvg, axis=0)
s1umStd = np.std(sleep1um_binAvg, axis=0)

sleep2_bin = np.reshape(Activity_postM[:,0:int(Activity_postM.shape[1]/Num_point)*Num_point], (Activity_postM.shape[0], Num_point, -1))
sleep2_binAvg = np.mean(sleep2_bin, axis=1)

s2Avg = np.mean(sleep2_binAvg, axis=0)
s2Std = np.std(sleep2_binAvg, axis=0)


sleep2um_bin = np.reshape(Activity_postUM[:,0:int(Activity_postUM.shape[1]/Num_point)*Num_point], (Activity_postUM.shape[0], Num_point, -1))
sleep2um_binAvg = np.mean(sleep2um_bin, axis=1)


s2umAvg = np.mean(sleep2um_binAvg, axis=0)
s2umStd = np.std(sleep2um_binAvg, axis=0)

s2umAvg = np.mean(sleep2um_binAvg, axis=0)
s2umStd = np.std(sleep2um_binAvg, axis=0)


### plot results
fig, ax2= plt.subplots(1,2, figsize=(8,4))
ax2[0].errorbar(np.linspace(0,180, sleep1_binAvg.shape[1]), s1Avg, s1Std, color='blue', label='pre-sleep', capsize=1.5)
ax2[0].errorbar(np.linspace(0,180, sleep2_binAvg.shape[1]), s2Avg, s2Std, color='red', label='post-sleep',  capsize=1.5)

ax2[0].legend()

ax2[1].errorbar(np.linspace(0,180, sleep1um_binAvg.shape[1]), s1umAvg, s1umStd, color='blue', label='pre-sleep(um)',  capsize=1.5)
ax2[1].errorbar(np.linspace(0,180, sleep2um_binAvg.shape[1]), s2umAvg, s2umStd, color='red', label='post-sleep(um)',  capsize=1.5)

ax2[0].set_ylim([0, 0.03])
ax2[1].set_ylim([0, 0.03])

ax2[0].set_title('Reach-decorrelated')
ax2[1].set_title('Reach-unmodulated')

plt.legend()























x=g44
############## plot results ##########
idx = 9


tbin1 = np.arange(t_ss1, t_se1, As_across_bins_modulated[idx]['bin_size'])
tbin2 = np.arange(t_ss2, t_se2, As_across_bins_modulated[idx]['bin_size'])

fig, ax = plt.subplots(1,3, figsize=(25,4))

ax[0].plot(tbin1[0:len(tbin1)-1], Activity_s1[idx,:])
ax[1].plot(np.mean(peri_reach[idx,:,:], axis=1))
ax[2].plot(tbin2[0:len(tbin2)-1], Activity_s2[idx,:])

ax[0] = epochs_coloring(ax[0], np.vstack((t_rem1, t_rem2)), np.max(Activity_s1[idx,:]), epoch='rem')
ax[0] = epochs_coloring(ax[0], np.vstack((t_sws1, t_sws2)), np.max(Activity_s1[idx:]), epoch='sws')


ax[2] = epochs_coloring(ax[2], np.vstack((t_rem1, t_rem2)), np.max(Activity_s2[idx,:]), epoch='rem')
ax[2] = epochs_coloring(ax[2], np.vstack((t_sws1, t_sws2)), np.max(Activity_s2[idx,:]), epoch='sws')



































x=g66666
sig_shuffled_post_reach = sig_shuffled[:,midpoint + np.arange(0,round(after_event/freq)),:]

sig_shuffled_pre_reach = sig_shuffled[:,midpoint + np.arange(-round(before_event/freq),0),:]


sig = np.mean(peri_reach,2)

baseline = np.median(sig[:,0:midpoint-round((before_event + after_event)/freq)], 1) 

sig_after_reach = sig[:,midpoint + np.arange(0,round(after_event/freq))]
sig_before_reach = sig[:,midpoint + np.arange(-round(before_event/freq),0)]


modulation_metric_after_reach = np.mean((sig_after_reach - np.mean(sig_shuffled_post_reach,2))**2,1)
modulation_metric_before_reach = np.mean((sig_before_reach - np.mean(sig_shuffled_pre_reach,2))**2,1)

sig_shuffled_post_reach = np.mean(sig_shuffled_post_reach, axis=2)
modulation_metric_shuffled_after_reach = sig_shuffled_post_reach - np.mean(sig_shuffled_post_reach, axis=1).reshape(sig_shuffled_post_reach.shape[0],1)


modulation_flag_after_reach = modulation_metric_after_reach > np.percentile(modulation_metric_shuffled_after_reach,sig_threshold,1)
modulation_flag_after_reach = (np.mean(sig_after_reach, axis=1)>baseline) & (modulation_flag_after_reach)



sig_shuffled_pre_reach = np.mean(sig_shuffled_pre_reach, axis=2)
modulation_metric_shuffled_before_reach = sig_shuffled_pre_reach - np.mean(sig_shuffled_pre_reach, axis=1).reshape(sig_shuffled_pre_reach.shape[0],1)


modulation_flag_before_reach = modulation_metric_before_reach > np.percentile(modulation_metric_shuffled_before_reach,sig_threshold,1)
modulation_flag_before_reach = (np.mean(sig_before_reach, axis=1)>baseline) & (modulation_flag_before_reach)

#### Activity during sleep
Activity_s1 = Assembly_activity[:,s1s_idx:s1e_idx]
Activity_s2 = Assembly_activity[:,s2s_idx:s2e_idx]