# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 21:15:16 2022

@author: p.nazarirobati
"""

import numpy as np

def autocorr(xx):
    
    acorr = np.correlate(xx, xx, mode='full')
    acorr = acorr[acorr.size//2:]
    
    acorr_norm = acorr/max(acorr)
    
    return acorr_norm