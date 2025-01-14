# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:08:52 2022

@author: p.nazarirobati
"""

def raster_visualization(spike_train_file, t_start, t_end):
    with open(spike_train_file, 'rb') as q:
        spikes = pickle.load(q)
    nneurons = len(spikes)
    for i in range(nneurons):
        spikes[i] = [x for x in spikes[i] if ((x>=t_start) and (x<=t_end))]
    rasterplot(spikes, s=1.5, c='black')
    plt.show()