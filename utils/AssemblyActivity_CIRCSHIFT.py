# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:00:31 2023

@author: p.nazarirobati
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:20:40 2023

@author: p.nazarirobati (p.nazarirobati@leth.ca)

### finding reach-modulated cell ensembles around reaching time (Karimi et al.2023, Jadhav et al. 2016)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from tslearn.metrics import dtw
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from Assembly_RasterPlot import Assembly_RasterPlot
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from scipy import stats

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



def modulation_type(sig, before_event, after_event, baseline, modulation_flag):
    
    midpoint = round(sig.shape[1]/2)
    
    
    sig_before = sig[:,0:midpoint]
    sig_after = sig[:, midpoint:-1]
    
    positive_flag_before = np.logical_and((np.mean(sig_before,axis=1) > baseline),  modulation_flag)
    negative_flag_before = np.logical_and((np.mean(sig_before,axis=1) < baseline),  modulation_flag)
    
    positive_flag_after = np.logical_and((np.mean(sig_after,axis=1) > baseline),  modulation_flag)
    negative_flag_after = np.logical_and((np.mean(sig_after,axis=1) < baseline),  modulation_flag)
    
    
    r1 = np.logical_and(positive_flag_before, positive_flag_after)
    r2 = np.logical_and(positive_flag_before, negative_flag_after)
    
    r3 = np.logical_and(negative_flag_before, positive_flag_after)
    r4 = np.logical_and(negative_flag_before, negative_flag_after)
    
    arr = np.vstack((r1,r2, r3, r4))

    return arr



########################################    
######### MAIN CODE ####################
########################################
### paramters intializations for reading files
day= '2015-10-14'
day_id = 1

### fixed parameters
BinSizes = [30, 50, 100, 250, 350, 500, 650, 750, 900, 1000] # 0.1ms unit

### paramters intialization
bin_size = BinSizes[9]
significant_threshold = 95
q_treshold = 0.9 # percentile to remove assemblies with total activation above it (all-time activated CAs)
nneu_threshold = 2  # thresholding value to remove assemblies with members less than this value (should be 1 if it requires to keep all CAs)

pre = 3 # window size before event onset (for shuffled condition)
post = 3 # window size after event onset (for shuffled condition)

before_event = .5 # window size before event onset (for real condition)
after_event = 1 # window size after event onset (for real condition)

t_base = 2

num_shuffling = 1000 # number of permutation 
max_shift = 1.5

####### reading files info
path_name = r"E:\Master Project\cell assembly_Russo\assemblies_Russo\rr8\\" + str(day)
spt_file = r"E:\Master Project\spikes_data\rr8\\" + str(day_id) + ".pkl" 
reach_info = pd.read_csv(r'E:\Master Project\cell assembly_Russo\assemblies_Russo\rr8\\'+ str(day)+ '\epochs.treach.txt', sep='\s+')

####### loading data
spM = spiketrain(spt_file) # spike data (unbinned)
patterns = assemblies_data(path_name) # list of detected cell assemblies (Russo, 2017 main output)

####### filtering reach information data (removing multiple attempt trials and trials with negative duration)
# step-1: removing multiple-attempt trials
# step-2: removing False trials (negative duration)
# step-3: removing outlier trials (those with duration more than a threshold)
reach_info = reach_info[reach_info['issingle']==1]
reach_info = reach_info[reach_info['treturn'] - reach_info['tadvance']>0]
reach_info = reach_info[(reach_info['treturn'] - reach_info['tadvance']) < (reach_info['treturn'] - reach_info['tadvance']).quantile(q=significant_threshold/100)]

freq = bin_size/10000 # CA activation resolution 
nneu = len(spM) # number of neurons
### reforming patterns based on As_Across_bins  ###
As_across_bins, As_across_bins_index = assemblies_across_bins(patterns, BinSizes)
#As_across_bins_pr, As_across_bins_pr_index = prunning_across_bins(As_across_bins, As_across_bins_index, nneu, criteria='distance', th=0.3, style='signature')
#print('total number of detected assemblies: ', len(As_across_bins))  

### preprocessing detected assemblies  #####
## step-1: filtering CAs in a specific bin_size
## step-2: removing assemblies whose number of activation are more than q_threshold
## step-3: removing assemblies with members less than nneu_threshold

# step-1
As_across_bins_cut = [As_across_bins[xx] for xx in range(len(As_across_bins)) if As_across_bins[xx]['bin_size']==bin_size]
# step-2
count_dist = np.array([len(As_across_bins_cut[xx]['times']) for xx in range(len(As_across_bins_cut))])
count_distQuantile = np.quantile(count_dist, q=q_treshold)
# step-3
As_across_bins_cut = [As_across_bins_cut[xx] for xx in range(len(As_across_bins_cut)) if len(As_across_bins_cut[xx]['times'])<count_distQuantile]

############################################
### sorting zero-lag assemblies based on the neuron_id (optional and just for better understanding of data)
for jj in range(len(As_across_bins_cut)):
    
    if len(set(As_across_bins_cut[jj]['lags']))==1:
        As_across_bins_cut[jj]['neurons'] = sorted(As_across_bins_cut[jj]['neurons'])

As_across_bins_cut = [As_across_bins_cut[xx] for xx in range(len(As_across_bins_cut)) if len(As_across_bins_cut[xx]['neurons'])>nneu_threshold]

######## reading epoch timestamps data ############
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

t_wake1 = epochs_ts['t_wake1']
t_wake2 = epochs_ts['t_wake2']

t_reaching = np.array(reach_info.iloc[:,3:6]) # reaching times stamps info

####################### converting epoch timestamps info from cheetah to vector starting from zero

tb = np.arange(t_start, t_end, bin_size)

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

wake1 = convert_cheetah_index(tb, t_wake1)
wake2 = convert_cheetah_index(tb, t_wake2)

reaching_idx = convert_cheetah_index2(tb, t_reaching)
grasp_lag = reaching_idx[:,1] - reaching_idx[:,0] # LAG btw grasping time from reach onset

### Assembly Activity across recording time  ################################
Assembly_activity = np.array(AssemblyActivity(As_across_bins_cut, spM, BinSizes, t_start, t_end, method ='full', LagChoice='duration'))

######################################################################################################
############# Finding reach-modulated assemblies (Karimi et.al 2023, Jadhav et al. 2016) #############
######################################################################################################


### Creating Peri-Reach time histogram (PETH)
peri_reach = np.zeros((Assembly_activity.shape[0], int((pre+post)/freq), reaching_idx.shape[0]))

midpoint = round(peri_reach.shape[1]/2)

for jj in range(reaching_idx.shape[0]):
    
    tpre = int(reaching_idx[jj,0] - int(pre/freq))
    tpost = int(reaching_idx[jj,0] + int(post/freq))

    peri_reach[:,:,jj] = Assembly_activity[:, tpre:tpost]
           

### shuffling PETHs structure using CIRCULAR SHIFT ######
#np.random.seed(42)    
rng = np.random.default_rng(seed=0)

sig_shuffled = np.zeros((peri_reach.shape[0], peri_reach.shape[1], num_shuffling))  # array including assemblies-shuffled condition  
       
for perm_counter in range(num_shuffling):
    
    sig_template = np.zeros(np.shape(peri_reach))
    
    shift = np.round(4*max_shift*(rng.random(peri_reach.shape[2]) - .5),2)
    
    shift = np.round(shift/freq)
    #print(shift)
    
    for reach_counter in range(peri_reach.shape[2]):
        sig_template[:,:,reach_counter] = np.roll(peri_reach[:,:,reach_counter], int(shift[reach_counter]), axis=1)
        
    sig_shuffled[:,:,perm_counter] = np.mean(sig_template, axis=2)    
    
sig = np.mean(peri_reach,axis=2)
sig = sig[:,midpoint - round(before_event/freq):midpoint + round(after_event/freq)]

sig_shuffled = sig_shuffled[:,midpoint - round(before_event/freq):midpoint+round(after_event/freq), :]

### creating a baseline to find positive/ negative modulation

#b1 = np.median(np.mean(peri_reach[:,0:midpoint - round((before_event)/freq)], axis=2), axis=1)
#b2 = np.median(np.mean(peri_reach[:,midpoint + round((after_event)/freq):peri_reach.shape[1]], axis=2), axis=1)

b1 = np.median(np.mean(peri_reach[:,midpoint - round(before_event/freq) - round(t_base/freq):midpoint - round(before_event/freq),:], axis=2), axis=1)
b2 = np.median(np.mean(peri_reach[:,midpoint + round(after_event/freq):midpoint + round(after_event/freq) + round(t_base/freq),:], axis=2), axis=1)

#b1 = np.median(np.mean(peri_reach[:,midpoint - round(t_base/freq):midpoint,:], axis=2), axis=1)
#b2 = np.median(np.mean(peri_reach[:,midpoint + round((after_event/2)/freq):midpoint + round((after_event/2)/freq) + round(t_base/freq),:], axis=2), axis=1)


#baseline = (b1 + b2)/2
baseline = b1

modulation_metric_reach = np.mean((sig - np.mean(sig_shuffled,2))**2,1)  ## euclidean distance

modulation_metric_shuffled = np.array([sig_shuffled[:,:,i] - np.mean(sig_shuffled, axis=2) for i in range(sig_shuffled.shape[2])])
modulation_metric_shuffled = np.reshape(modulation_metric_shuffled,np.shape(sig_shuffled))
modulation_metric_shuffled = np.mean(modulation_metric_shuffled**2, axis=1)


Bonferroni_threshold = significant_threshold + (100-significant_threshold)/sig.shape[0]

modulation_flag = modulation_metric_reach > np.percentile(modulation_metric_shuffled, Bonferroni_threshold,1)  ### significant threshold is Bonferroni-corrected

positive_flag = np.logical_and((np.mean(sig,axis=1) > baseline),  modulation_flag)
negative_flag = np.logical_and((np.mean(sig,axis=1) < baseline),  modulation_flag)


reach_correlated_CA_idx = np.where(positive_flag == True)[0]
reach_uncorrelated_CA_idx = np.where(negative_flag == True)[0]

reach_modulated_idx = np.hstack([reach_correlated_CA_idx, reach_uncorrelated_CA_idx])
reach_unmodulated_idx = np.setdiff1d(np.arange(modulation_flag.size), reach_modulated_idx)

### reach modulated assemblies (in as_across_bins format)
As_across_bins_modulated = [As_across_bins_cut[xx] for xx in list(reach_modulated_idx)]
As_across_bins_unmodulated = [As_across_bins_cut[xx] for xx in list(reach_unmodulated_idx)]


As_across_bins_correlated_idx = [[0,j] for j in range(len(reach_correlated_CA_idx))]    
As_across_bins_correlated = [As_across_bins_cut[xx] for xx in list(reach_correlated_CA_idx)]

#As_across_bins_pr, As_across_bins_index_pr  = prunning_across_bins(As_across_bins_correlated, As_across_bins_correlated_idx, nneu, criteria='distance', th=0.3, style='signature')
#idx_pr = [As_across_bins_index_pr[i][1] for i in range(len(As_across_bins_pr))]

##############################################################################################
##############################################################################################
#### Normalizing PETHs before plotting results

peri_reach_norm = (np.mean(peri_reach,axis=2) - np.mean(np.mean(peri_reach,axis=2), axis=1).reshape(-1,1))/ (np.std(np.mean(peri_reach,axis=2), axis=1).reshape(-1,1)+0.000000001)

peri_reach_correlated = np.mean(peri_reach[reach_correlated_CA_idx,:,:], axis=2)
peri_reach_correlated = (peri_reach_correlated - np.mean(peri_reach_correlated, axis=1).reshape((-1,1)))/(np.std(peri_reach_correlated, axis=1).reshape(-1,1)+ 0.0000000001)

peri_reach_uncorrelated = np.mean(peri_reach[reach_uncorrelated_CA_idx,:,:], axis=2)
peri_reach_uncorrelated = (peri_reach_uncorrelated - np.mean(peri_reach_uncorrelated, axis=1).reshape((-1,1)))/(np.std(peri_reach_uncorrelated, axis=1).reshape(-1,1)+ 0.0000000001)

peri_reach_unmodulated = np.mean(peri_reach[reach_unmodulated_idx,:,:], axis=2)
peri_reach_unmodulated = (peri_reach_unmodulated - np.mean(peri_reach_unmodulated, axis=1).reshape((-1,1)))/(np.std(peri_reach_unmodulated, axis=1).reshape(-1,1)+ 0.0000000001)
#############################################

########### plot PETH results after shuffling ######
plt.style.use('default')
font = {'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 10}
mpl.rc('font', **font)

fig, ax = plt.subplots(1,3, figsize=(6,4), sharex=True, sharey=False, dpi=80)
axes = [ax[0], ax[0].twinx()]


axes[-1].set_frame_on(True)
axes[-1].patch.set_visible(False)

#max_v = .9*max(np.max(peri_reach_correlated), np.max(peri_reach_uncorrelated), np.max(peri_reach_unmodulated))
#min_v = .9*min(np.min(peri_reach_correlated), np.min(peri_reach_uncorrelated), np.min(peri_reach_unmodulated))
max_v = .9*np.max(peri_reach_norm)
min_v = np.min(peri_reach_norm)


im1 = axes[0].imshow(peri_reach_correlated, cmap='jet', aspect='auto', vmin = min_v-.01 , vmax = max_v+0.01) 
axes[0].axvline(x=int((pre/(bin_size/10000)+ grasp_lag).mean()), color='lightgray', linestyle = '--')

axes[1].plot(np.mean(peri_reach_correlated,axis=0), color='white')
axes[1].fill_between(np.arange(peri_reach.shape[1]), np.mean(peri_reach_correlated,axis=0) - np.std(peri_reach_correlated,axis=0),  np.mean(peri_reach_correlated,axis=0) + np.std(peri_reach_correlated,axis=0), color='gray', alpha = 0.5)

axes[1].set_ylim([-5,5])

ls_loc = list(np.arange(0,peri_reach.shape[1],100))
axes[0].set_xticks(ls_loc)
lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)


axes = [ax[1], ax[1].twinx()]


im2 = axes[0].imshow(peri_reach_uncorrelated, cmap='jet', aspect='auto',  vmin = min_v-.01 ,vmax = max_v+0.01) 
axes[0].axvline(x=int((pre/(bin_size/10000)+ grasp_lag).mean()), color='lightgray', linestyle = '--')

axes[1].plot(np.mean(peri_reach_uncorrelated,axis=0), color='white')
axes[1].fill_between(np.arange(peri_reach.shape[1]), np.mean(peri_reach_uncorrelated,axis=0) - np.std(peri_reach_uncorrelated,axis=0), np.mean(peri_reach_uncorrelated,axis=0) + np.std(peri_reach_uncorrelated,axis=0), color='gray', alpha = 0.5)

axes[1].set_ylim([-5,5])

ls_loc = list(np.arange(0,peri_reach.shape[1],10))
axes[0].set_xticks(ls_loc)
lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)

axes = [ax[2], ax[2].twinx()]

im3 = axes[0].imshow(peri_reach_unmodulated, cmap='jet',  aspect='auto',  vmin = min_v-.01 ,vmax = max_v+0.01) 
axes[0].axvline(x=int((pre/(bin_size/10000)+ grasp_lag).mean()), color='lightgray', linestyle = '--')

axes[1].plot(np.mean(peri_reach_unmodulated,axis=0), color='white')
axes[1].fill_between(np.arange(peri_reach.shape[1]), np.mean(peri_reach_unmodulated,axis=0) - np.std(peri_reach_unmodulated,axis=0), np.mean(peri_reach_unmodulated,axis=0) + np.std(peri_reach_unmodulated,axis=0), color='gray', alpha = 0.5)
axes[1].set_ylim([-5,5])


ls_loc = list(np.arange(0,peri_reach.shape[1],10))
axes[0].set_xticks(ls_loc)
lst = [xx*bin_size/10000-pre for xx in ls_loc]
axes[0].set_xticklabels(lst)

axes[1].set_yticklabels([])
axes[1].tick_params(right = False)
#axes[1].set_yticslabel([])

divider = make_axes_locatable(axes[0])

cax1 = divider.append_axes("right", size="5%", pad=0.05)

cbar = fig.colorbar(im1, cax=cax1)

#cbar = plt.colorbar(im1, ax=axes[0], orientation='vertical', pad=0.15)
cbar.set_label('PETH activation strength (z-scored)', rotation=90, labelpad = 15)


ax[0].set_title('Reach-activated assemblies')
ax[1].set_title('Reach-inactivated assemblies')
ax[2].set_title('Reach-unmodulated assemblies')

ax[1].set_xlabel('Time from reach onset (sec)')
ax[0].set_ylabel('CA#')
#filename1 = 'rr5_' + str(day) + '_reachCorrelated.pkl'
#filename2 = 'rr5_' + str(day) + '_reachDecorrelated.pkl'
#filename3 = 'rr5_' + str(day) + '_reachUnmodulated.pkl'




x=g666
##############################################################
##############################################################
######## activity of reach-modulated CAs during sleep1 & sleep2

## activity across pre/post sleep recording
Activity_preM = Assembly_activity[reach_modulated_idx,s1s_idx:s1e_idx]
Activity_postM = Assembly_activity[reach_modulated_idx,s2s_idx:s2e_idx]


Activity_preUM = Assembly_activity[reach_unmodulated_idx,s1s_idx:s1e_idx]
Activity_postUM = Assembly_activity[reach_unmodulated_idx,s2s_idx:s2e_idx]


## activity across sleep epochs during pre/post training sleep periods
Activity_s1M = AssemblyActivityEpochs(Assembly_activity[reach_modulated_idx,:], sleep1)
Activity_s2M = AssemblyActivityEpochs(Assembly_activity[reach_modulated_idx,:], sleep2)

Activity_s1UM = AssemblyActivityEpochs(Assembly_activity[reach_unmodulated_idx,:], sleep1)
Activity_s2UM = AssemblyActivityEpochs(Assembly_activity[reach_unmodulated_idx,:], sleep2)


Activity_w1M = AssemblyActivityEpochs(Assembly_activity[reach_modulated_idx,:], wake1)
Activity_w2M = AssemblyActivityEpochs(Assembly_activity[reach_modulated_idx,:], wake2)

Activity_w1UM = AssemblyActivityEpochs(Assembly_activity[reach_unmodulated_idx,:], wake1)
Activity_w2UM = AssemblyActivityEpochs(Assembly_activity[reach_unmodulated_idx,:], wake2)
##

Avg_s1M = np.mean(Activity_s1M, axis=1)
Avg_s2M = np.mean(Activity_s2M, axis=1)


Avg_s1UM = np.mean(Activity_s1UM, axis=1)
Avg_s2UM = np.mean(Activity_s2UM, axis=1)

Avg_w1M = np.mean(Activity_w1M, axis=1)
Avg_w2M = np.mean(Activity_w2M, axis=1)

Avg_w1UM = np.mean(Activity_w1UM, axis=1)
Avg_w2UM = np.mean(Activity_w2UM, axis=1)

### finding reactivation candidates

S1S2_metric = 100 * (Avg_s2M - Avg_s1M)/(Avg_s1M)
S2W2_metric = 100 * (Avg_s2M - Avg_w2M)/(Avg_w2M)

S1W1_metric = 100 * (Avg_s1M - Avg_w1M)/(Avg_w1M)

#reactivation_CA = np.logical_and((S1S2_metric>np.quantile(S1S2_metric, q=.95)), (S2W2_metric>np.quantile(S2W2_metric, q=.95)))
reactivation_CA = S1S2_metric>np.quantile(S1S2_metric, q=.95)
Preactivation_CA = np.logical_and((S1S2_metric<np.quantile(S1S2_metric, q=.1)), (S1W1_metric>np.quantile(S1W1_metric, q=.95)))

#x=g666
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


ca_idx = 2
fig, ax= plt.subplots(2,1, figsize=(12,4))

tbin1 = np.arange(0, ((t_se1-t_ss1)/10000)/60, ((bin_size)/10000)/60)
ax[0].plot(tbin1[0:len(tbin1)-1],Activity_preM[ca_idx,:], color='blue', marker = 'o')

ax[0] = epochs_coloring(ax[0], ((t_sws1-t_ss1)/10000)/60, max(np.max(Activity_preM[ca_idx,:]), np.max(Activity_postM[ca_idx,:])), epoch='sws')
ax[0] = epochs_coloring(ax[0], ((t_rem1-t_ss1)/10000)/60, max(np.max(Activity_preM[ca_idx,:]), np.max(Activity_postM[ca_idx,:])), epoch='rem')
ax[0] = epochs_coloring(ax[0], ((t_wake1-t_ss1)/10000)/60, max(np.max(Activity_preM[ca_idx,:]), np.max(Activity_postM[ca_idx,:])), epoch='wake')
ax[0].set_title('Pre-training sleep')

ax[0].set_xlim([-.5,180.5])

tbin2 = np.arange(0, ((t_se2-t_ss2)/10000)/60, ((As_across_bins_cut[0]['bin_size'])/10000)/60)
ax[1].plot(tbin2[0:len(tbin2)-1],Activity_postM[ca_idx,:], color='blue', marker = 'o')

ax[1] = epochs_coloring(ax[1], ((t_sws2-t_ss2)/10000)/60, max(np.max(Activity_preM[ca_idx,:]), np.max(Activity_postM[ca_idx,:])), epoch='sws')
ax[1] = epochs_coloring(ax[1], ((t_rem2-t_ss2)/10000)/60, max(np.max(Activity_preM[ca_idx,:]), np.max(Activity_postM[ca_idx,:])), epoch='rem')
ax[1] = epochs_coloring(ax[1], ((t_wake2-t_ss2)/10000)/60, max(np.max(Activity_preM[ca_idx,:]), np.max(Activity_postM[ca_idx,:])), epoch='wake')
ax[1].set_title('Post-training sleep')

ax[1].set_xlim([-.5,180.5])

##### reactivation candidate


time_axis = np.arange(s2s_idx, s2e_idx)
epoch_status = np.zeros_like(time_axis, dtype=str)

epoch_start_times = [sws_s2[:, 0], rem_s2[:, 0], wake2[:, 0]]
epoch_end_times = [sws_s2[:, 1], rem_s2[:, 1], wake2[:, 1]]
epoch_labels = ['S', 'R', 'W']
#epoch_labels = [0, 1, 2]

    
for start_times, end_times, label in zip(epoch_start_times, epoch_end_times, epoch_labels):
    for start_time, end_time in zip(start_times, end_times):
        # Convert the epoch start and end times to indices in the time axis
        start_index = int(start_time - s2s_idx) 
        end_index = int(end_time - s2s_idx)

        # Assign the epoch label to the corresponding time range
        epoch_status[start_index:end_index] = str(label)


sleep_df = pd.DataFrame({'Activity':Activity_postM[0, :], 'SleepStage': epoch_status, 'Assembly':np.arange(Activity_postM.shape[1])})

dummy_sleepstage = pd.get_dummies(epoch_status, columns=['SleepStage'], drop_first=True)
data_with_dummies = pd.concat([sleep_df, dummy_sleepstage], axis=1)


X = data_with_dummies[['S', 'R']]  # Select the relevant predictor variables
X = sm.add_constant(X)  # Add a constant term
y = data_with_dummies['Activity']  # Select the response variable

model = sm.ZeroInflatedPoisson(y, X, inflation = 'probit')  
results = model.fit()  # Fit the model
print(results.summary())


x=g66
assembly_data = []
activity_data = []
stage_data = []

for ii in range(Activity_postM.shape[0]):
    # Extract the activity and stage information for the current assembly
    activities = Activity_postM[ii,:]
    stages = epoch_status
    
    # Append the assembly, activity, and stage data to the respective lists
    assembly_data.extend([ii] * len(activities))
    activity_data.extend(activities)
    stage_data.extend(stages)


data = pd.DataFrame({'Assembly': assembly_data, 'Activity': activity_data, 'Stage': stage_data})

model_formula = "Activity ~ Stage + (1 | Assembly)"
model = smf.mixedlm(model_formula, data=data, groups=data["Assembly"])
result = model.fit()

# Print the model summary
print(result.summary())


x=g6666
#x=g55
### plot results
fig, ax2= plt.subplots(1,2, figsize=(8,4))
ax2[0].errorbar(np.linspace(0,180, sleep1_binAvg.shape[1]), s1Avg, s1Std, color='blue', label='pre-sleep', capsize=1.5)
ax2[0].errorbar(np.linspace(0,180, sleep2_binAvg.shape[1]), s2Avg, s2Std, color='red', label='post-sleep',  capsize=1.5)

ax2[0].legend()

ax2[1].errorbar(np.linspace(0,180, sleep1um_binAvg.shape[1]), s1umAvg, s1umStd, color='blue', label='pre-sleep(um)',  capsize=1.5)
ax2[1].errorbar(np.linspace(0,180, sleep2um_binAvg.shape[1]), s2umAvg, s2umStd, color='red', label='post-sleep(um)',  capsize=1.5)

ax2[0].set_ylim([0, 0.3])
ax2[1].set_ylim([0, 0.3])

ax2[0].set_title('Reach-modulated')
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