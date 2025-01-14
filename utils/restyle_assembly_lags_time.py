# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:58:59 2022

@author: p.nazarirobati
"""
import numpy as np

def restyle_assembly_lags_time(A, A_index):
    
    As_restyled= []
    As_restyled_index = A_index
    numAss = len(A)
    
    for i in range(numAss):
        #print(i)
        dictt = {}
        llag = A[i]['lags']
        llag.insert(0, 0)
        lag_s = np.sort(llag)
        lag_s_idx = np.argsort(llag)
        minlag = min(lag_s)
        dictt= {'neurons':[A[i]['neurons'][j] for j in lag_s_idx], 'lags': lag_s - minlag, 'times':A[i]['times'] + minlag ,
                'pvalue':A[i]['pvalue'][-1], 'signature':A[i]['signature'][-1][1], 'bin_size':A[i]['bin_size']}
        As_restyled.append(dictt.copy())
    
    return As_restyled, As_restyled_index    