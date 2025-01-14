# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:10:56 2022

@author: p.nazarirobati
"""
import pickle
import os

def spiketrain(filename):
    with open(filename, 'rb') as q:
        data = pickle.load(q)
    spikes = data['spikes']
    t_start = data['times session'][0,0]
    t_end = data['times session'][0,1]
    nneurons = len(spikes)
    for i in range(nneurons):
        spikes[i] = [x for x in spikes[i] if ((x>=t_start) and (x<=t_end))]

    return spikes