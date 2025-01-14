# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:21:06 2023

@author: p.nazarirobati(p.nazarirobati@uleth.ca)

### converting cheetah timestamps to vector started from zero
"""
import numpy as np

def convert_cheetah_index(tb, epochs):
    
    epoch_indx = np.zeros(np.shape(epochs))

    for i in range(epochs.shape[0]):
    
        st = np.where(tb>=epochs[i,0])[0][0]
        se = np.where(tb<=epochs[i,1])[0][-1]
    
        epoch_indx[i,:] = [int(st), int(se)]

    return epoch_indx



def convert_cheetah_index2(tb, epochs_so):
    
    epoch_indx = np.zeros(np.shape(epochs_so))

    for i in range(epochs_so.shape[0]):
    
        st = np.where(tb>=epochs_so[i,0])[0][0]
        sm = np.where(tb>=epochs_so[i,1])[0][0]
        se = np.where(tb>=epochs_so[i,2])[0][0]
    
        epoch_indx[i,:] = [int(st), int(sm),int(se)]

    return epoch_indx