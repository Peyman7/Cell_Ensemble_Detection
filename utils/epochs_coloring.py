# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:38:48 2023

@author: p.nazarirobati
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.collections import PatchCollection


def epochs_coloring(ax, time_epochs, max_Assembly_activity, epoch):
    patches = []
    
    for jj in range(np.shape(time_epochs)[0]):
        
        x_axis = time_epochs[jj,0]
        
        y_axis = time_epochs[jj,1] - time_epochs[jj,0]
       
        patches.append(matplotlib.patches.Rectangle((x_axis, 0), y_axis, max_Assembly_activity+2 ))            
    
    
    
    if epoch =='rem':
        p = PatchCollection(patches, color='indianred', alpha=0.5, edgecolor='none')
    
    elif epoch =='sws':
        p = PatchCollection(patches, color='green', alpha=0.5, edgecolor='none')
    elif epoch =='spindle':
        p = PatchCollection(patches, color='yellow', alpha=0.5, edgecolor='none')
    elif epoch =='swr':
        p = PatchCollection(patches, color='orange', alpha=0.5, edgecolor='none')
    elif epoch =='wake':
        p = PatchCollection(patches, color='lightgray', alpha=0.5, edgecolor='none')
        

    ax.add_collection(p)
    
    return ax
