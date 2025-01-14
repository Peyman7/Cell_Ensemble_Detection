# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:59:40 2022

@author: p.nazarirobati
"""
import numpy as np
from restyle_assembly_lags_time import restyle_assembly_lags_time

def assemblies_across_bins(assembly, BinSizes):
    
    ## Parameters Initialization
    empty=1
    e=0
    NN = sum([len(assembly[i]) for i in range(len(assembly))])
    as_across_bins = [{}]*(sum([len(assembly[i]) for i in range(len(assembly))]))
    as_across_bins_index = [[]]*(sum([len(assembly[i]) for i in range(len(assembly))]))
    
    # Test first temporal bin
    while empty:
        
        if len(assembly[e])>0:
            nAss = len(assembly[e])    
            as_across_bins[0:nAss]=assembly[e]
            for i in range(nAss):
                as_across_bins[i]['bin_size']=BinSizes[e]
                as_across_bins_index[i]=[e,i]
            j=nAss
            identity= range(nAss)
            id=nAss
            empty=0
        else:
            e=e+1 
        #print(j)  
    for gg in range(e+1,len(BinSizes)):
        if len(assembly[gg])>0:
            #print('bin: ', BinSizes[gg])
            nAss = len(assembly[gg])
            #print('num_assm: ', nAss)
            for i in range(nAss):
                if j<NN:
                    as_across_bins[j] = assembly[gg][i]
                    as_across_bins_index[j] = [gg, i]
                #id = id + 1
                #identity[j] = id
                    j=j+1        
    # Reorder lags and shift assembly's occurrence  
    as_across_bins, as_across_bins_index = restyle_assembly_lags_time(as_across_bins, as_across_bins_index)
    return as_across_bins, as_across_bins_index
            
