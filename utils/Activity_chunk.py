# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:20:55 2023

@author: p.nazarirobati
"""
import numpy as np

def Activity_chunk(idx_chunk,chunk_vector, Activity_epoch, BinSize):
    
    ls_chunk = [] 

    for ii in range(chunk_vector.size-1):
        ix = np.where((idx_chunk[:,1]<=chunk_vector[ii+1]) & (idx_chunk[:,1]>chunk_vector[ii]))
        ls_chunk.append(ix[0])

    Activity_tot = []
    
    for k in range(len(ls_chunk)):
        
        activity_chunk = np.zeros((Activity_epoch.shape[0],1))
        
        for xx in ls_chunk[k]:
            activity_chunk = np.hstack([activity_chunk, Activity_epoch[:,idx_chunk[xx,0]:idx_chunk[xx,1]]])

        activity_chunk_sum = np.sum(activity_chunk, axis=1)/(activity_chunk.shape[1]* BinSize/10000)
        #activity_chunk_sum = np.sum(activity_chunk, axis=1)

        #activity_chunk_sum = np.median(activity_chunk, axis=1)

        
        Activity_tot.append(activity_chunk_sum)
        
    return np.array(Activity_tot).T