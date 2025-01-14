# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:49:24 2023

@author: p.nazarirobati

preprocessing sleep events based as follows:
    - removing epochs with length above a specific threshold (up_threshold)
    - joining epochs with Inter Event Internval less than low_threshold as one epoch
"""
import numpy as np

def event_preprocessing(epochs, low_threshold, up_threshold):
    
    epochs_diff = np.diff(epochs)
    
    if up_threshold>0:
        
        ss = np.where(epochs_diff<up_threshold)[0]
        epochs_new = epochs[ss,:]
    
    if low_threshold>0:
        epochs_interval = [epochs_new[i,1] - epochs_new[i-1,0] for i in range(1,epochs_new.shape[0])]
        #print(epochs_interval[0])
        low_interval = [xx for xx in range(len(epochs_interval)) if epochs_interval[xx]<=low_threshold]
        
        print(low_interval)
        for jj in range(len(low_interval)):
            
            epochs_new[low_interval[jj]-1,1] = epochs_new[low_interval[jj],1]
        epochs_new = np.delete(epochs_new, low_interval, axis=0)
    
    return epochs_new