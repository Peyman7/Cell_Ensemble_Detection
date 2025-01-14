# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:09:15 2022

@author: p.nazarirobati
"""
import numpy as np
import matplotlib.pyplot as plt
import neo
import elephant 
from viziphant import rasterplot
from viziphant.rasterplot import eventplot
from viziphant.events import add_event
import quantities as pq
import pickle
from collections import defaultdict

def assemblies_time_visualization(spike_train_file, patterns, t_start, t_end, ttsk1, ttsk2, circle_sizes, colors=None):
    with open(spike_train_file, 'rb') as q:
        data = pickle.load(q)
    nneurons = len(data['spikes'])
    #print(nneurons)
    #ttsk1 = 120758412.94
    #ttsk2 = 138625271.01
    #tsk_binned = np.linspace(ttsk1, ttsk2, 60).tolist()
    #print(tsk_binned)
    
    spikes = data['spikes']
    for i in range(nneurons):
        spikes[i] = [x for x in spikes[i] if ((x>=t_start) and (x<=t_end))]
    
    spiketrains = [neo.SpikeTrain(spikes[i]*pq.ms, t_start = t_start*pq.ms, t_stop = t_end*pq.ms) for i in range(len(spikes))]   # Create neo.core spike train 

    if isinstance(patterns, dict):
       patterns = [patterns]
    
    event = neo.Event([ttsk1, ttsk2]*pq.ms, labels=['task_start', 'task_end'])
    axes = rasterplot.rasterplot(spiketrains, color='lightblue', s=circle_sizes[0])
    add_event(axes, event)
    #### 
    #for i in range(np.shape(tsk_binned)[0]-1):
    #    event1 = neo.Event([tsk_binned[i], tsk_binned[i+1]]*pq.ms, labels=['s', 'e'])
    #    add_event(axes, event1)
    units = spiketrains[0].units
    #print(units)
    time_scalar = units.rescale('ms').item()
    patterns_overlap = defaultdict(lambda: defaultdict(list))

    if colors is None:
        # +1 is necessary
        cmap = plt.cm.get_cmap("hsv", len(patterns) + 1)
        colors = np.array([cmap(i) for i in range(len(patterns))])
    elif not isinstance(colors, (list, tuple, np.ndarray)):
        raise TypeError("'colors' must be a list of colors")
    elif len(colors) != len(patterns):
        raise ValueError("The length of 'colors' must match the length of "
                         "the input 'patterns'.")

    for pattern_id, pattern in enumerate(patterns):
        times_ms = pattern['times']*pq.ms.magnitude.astype(int)
        lagg = pattern['lags']
        t_lg =[]
        t_lg_al = []
        for t in times_ms:
            for lg in lagg:
                t_lg = [(lg+ t) for lg in lagg]
            t_lg_al. append(t_lg)

        lgs = []
        neur= []

        for i in range(len(times_ms)):
            lgs.append(t_lg_al[i])
            neur.append(pattern['neurons'])
        patts = [neur, lgs]
        axes.scatter(patts[1], patts[0], 
        c=[colors[pattern_id]], s=circle_sizes[1])
        
    axes.set_yticks(np.arange(nneurons), minor=False)
    rows_label = list(range(nneurons))
    
    axes.set_ylabel('Neuron #')

    axes.yaxis.set_label_coords(-0.05, 0.5)
    axes.set_yticklabels(rows_label, minor=False, fontsize=6)
    #axes.set_xlim([23107273.75, 23108273.79])
    axes.set_xlim([t_start-1000000, t_end+1000000])
    axes.set_ylim([-0.5, nneurons])

    #axes.margins(0.2,2)
    plt.show()
    return axes