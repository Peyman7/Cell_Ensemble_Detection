# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:12:45 2022

@author: p.nazarirobati
"""

from __future__ import division, print_function, unicode_literals
import quantities as pq
import numpy as np
from scipy.io import loadmat
import neo
#import pandas as pd
import pickle
from neo import io
import quantities as pq
#from elephant.cell_assembly_detection import cell_assembly_detection
from elephant.conversion import BinnedSpikeTrain
#from parfor import parfor
import math
import multiprocessing
#from functools import partial
#from joblib import Parallel, delayed
import time
import warnings
from scipy.stats import f
import elephant.conversion as conv
from elephant.utils import deprecated_alias
import itertools
import copy
import os
import sys
import math
import time
import warnings
from scipy.stats import f
from elephant.utils import deprecated_alias
import zipfile
################################ RUSSO CAD METHOD ##################################

###############################SPIKE TRAIN NEO OBJECT##################################################
def spike_train2(spM, t_ss, t_es):
    
    spT = [neo.SpikeTrain(spM[i]*pq.ms, t_start = t_ss*pq.ms, t_stop = t_es*pq.ms) for i in range(len(spM))]   
    return spT
   
#######################################CAD PARALLEL###################################################################    
def CAD_p(spM):
    
    ### Create Q-matrix and CAD dictionaries
    hist_dict= {"Q-Matrix":[], "bin_size":[]}
    patt_list = []
    
    #### Parameters
    spike_train = spM['spike']
    t_ss = spM['t_st']
    t_es = spM['t_en']
    rat_num = spM['rat']
    rec_date = spM['date_rec']
    max_lags = spM['lag']
    bin_size = spM['bin']
    print('starting CAD on bin size: ', bin_size)
    print('start time: ', t_ss)
    print('end time: ', t_es)
    ### Binning spike train neo object
    spT_bin = conv.BinnedSpikeTrain(spike_train, bin_size= bin_size*pq.ms,t_start= t_ss*pq.ms, t_stop = t_es*pq.ms) # binned spike train (Input for Russo, 2017 CAD method)

    ### Russo, 2017 Cell Assembly Detection
    

    print('saving file:', rat_num, rec_date, str(bin_size))
    file_name_CAD_s = "BinnedSP_" + rat_num + "_" + rec_date + "_" + str(bin_size) + ".pkl"                    
    open_file = open(file_name_CAD_s, "wb")
    pickle.dump(spT_bin, open_file)
    open_file.close() 
    #patt_list.append(patterns)
   
    # save detected assemblies at each bin size in separate files.
    #file_name_CAD = "CAD_" + rat_num + "_" + rec_date + ".pkl"                    
    #open_file = open(file_name_CAD, "wb")
    #pickle.dump(patt_list, open_file)
    #open_file.close() 
    return spT_bin    


###########################PARALLEL RUN ###################################################################
    
def par(sp_list):
  
  results = CAD_p(sp_list)
  return results

##################################MAIN CODE FOR READING DATA #############################################
def main_assembly_detection_cc(runNum):
    
    
    # Read input file
    filename = str(runNum) + ".zip"
    print('input file name: ', filename)
    path = os.getcwd()
    #print(os.getcwd())
    #path_f = os.chdir('~/project/6000201/peymannr/Data/rr8')
    
    for root, dirs, files in os.walk(path):
      for file in files: 
        if file.startswith(filename):
          with zipfile.ZipFile(filename, 'r') as zip:
            #zips = zip.extractall()
              with zip.open(str(runNum)+".pkl") as myfile:
            #data_zip = zip.extractall()
                ##with open(myfile, 'rb') as q:
                  data = pickle.load(myfile)
        #print("Input file name: ", myfile)
    return data

############################################### MAIN CODE FOR RUNNING ####################################

data = main_assembly_detection_cc(int(os.getenv('SLURM_ARRAY_TASK_ID',0)))
print(len(data))
###parameters
bin_size = [30, 50, 100, 250, 350, 500, 650, 750, 900, 1000]   ## in NMSA unit (0.1 ms)
max_lags = 10

### data extraction
sp = data['spikes'] # spike train data
t_s = data['times session'][0,0] # t_start
t_e = data['times session'][0,1] # t_end
rat_n = data['rat_num'] # rat
date = data['date'] # date of recording
print('total numer of neurons: ', len(sp))
binL = len(bin_size)

# Create neo.core spike train
spT = spike_train2(sp, t_s, t_e)
splist = dict([('spike',spT), ('t_st',t_s) , ('t_en', t_e), ('rat',rat_n), ('date_rec', date), ('lag', max_lags), ('bin', bin_size)])
#size_list = len(split)
print('total length of splist:', len(splist))

# Parallel run

ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
print('number of available cpus: ', ncpus)
tic = time.time()
splist1 = splist
results = CAD_p(splist)
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))


