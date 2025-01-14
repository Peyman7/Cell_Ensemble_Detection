# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:37:07 2022

@author: p.nazarirobati
### This script analyzes spike train data, detects cell assemblies, and visualizes activation during task trials.
"""

# Importing Libraries
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from spiketrain import spiketrain
from epochs_data import epochs_data
from assemblies_data import assemblies_data
from assemblies_across_bins import assemblies_across_bins
from prunning_across_bins import prunning_across_bins

# Functions for Analysis
def load_data(file_path, file_type='pickle'):
    """Load data from a file."""
    if file_type == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_type == 'csv':
        return pd.read_csv(file_path, sep='\s+')
    else:
        raise ValueError("Unsupported file type. Use 'pickle' or 'csv'.")

def sort_activation_times(activation_times, assembly_indices):
    """Sort activation times and their corresponding assembly indices."""
    sort_idx = np.argsort(activation_times)
    sorted_times = [activation_times[i] for i in sort_idx]
    sorted_indices = [assembly_indices[i] for i in sort_idx]
    return sorted_times, sorted_indices

def filter_activation_times(activation_times, assembly_indices, t_start, t_end):
    """Filter activation times within a given range."""
    filtered_times = [t for t in activation_times if t_start <= t <= t_end]
    filtered_indices = [
        assembly_indices[i] for i, t in enumerate(activation_times) if t_start <= t <= t_end
    ]
    return filtered_times, filtered_indices

def plot_raster_with_task(reach_times, spikes, nneu, t_start, t_end, assembly_data, bin_size):
    """Plot raster plot of spike trains with task annotations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background for task trials
    for _, row in reach_times.iterrows():
        color = '#90EE90' if row['success'] == 1 else 'lightblue'
        ax.add_patch(
            patches.Rectangle(
                (row['tadvance'], 0),
                row['treturn'] - row['tadvance'],
                nneu + 1,
                facecolor=color,
                alpha=0.5
            )
        )
    
    # Spike raster
    for neuron_idx, neuron_spikes in enumerate(spikes):
        filtered_spikes = [t for t in neuron_spikes if t_start <= t <= t_end]
        ax.scatter(filtered_spikes, [neuron_idx] * len(filtered_spikes), s=2, color='gray')
    
    # Assembly raster
    cmap = plt.get_cmap('gnuplot')
    unique_idx = np.unique(assembly_data['indices'])
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_idx))]

    for i, (time, idx) in enumerate(zip(assembly_data['times'], assembly_data['indices'])):
        loc = np.where(unique_idx == idx)[0][0]
        neurons = assembly_data['assemblies'][idx]['neurons']
        lags = assembly_data['assemblies'][idx]['lags']
        neuron_firing_times = [time + lag for lag in lags]

        for neuron, firing_time in zip(neurons, neuron_firing_times):
            neuron_spikes = [spike for spike in spikes[neuron] if firing_time <= spike <= firing_time + bin_size]
            ax.scatter(neuron_spikes, [neuron] * len(neuron_spikes), s=10, color=colors[loc])

    ax.set_xlim([t_start - 10000, t_end + 10000])
    ax.set_ylim([-0.5, nneu + 0.5])
    ax.set_xlabel("Time (0.1 ms)", fontweight="bold")
    ax.set_ylabel("Neuron #", fontweight="bold")
    plt.tight_layout()
    plt.show()

# Main Code
if __name__ == "__main__":
    # Parameters
    bin_sizes = [30, 50, 100, 250, 350, 500, 650, 750, 900, 1000]
    bin_size = bin_sizes[5]
    t_before, t_after = 10000, 10000
    path_name = r"C:\Users\p.nazarirobati\Desktop\2015-10-14"
    reachfile_path = r"C:\Users\p.nazarirobati\Desktop\outputs\rr8\2015-10-14\epochs.treach.txt"
    spt_file = r"C:\Users\p.nazarirobati\Desktop\rr8\1.pkl"
    reach_idx_file = r"C:\Users\p.nazarirobati\Desktop\Analysis\Reaching_CA_idx\rr8_ReachingCA_idx.pkl"

    # Load data
    reach_times = load_data(reachfile_path, file_type='csv')
    spt_data = load_data(spt_file, file_type='pickle')
    reach_idx = load_data(reach_idx_file, file_type='pickle')
    reach_idx_day = reach_idx[0]

    # Initialize spike train and assembly data
    spikes = spiketrain(spt_file)
    patterns = assemblies_data(path_name)

    t_start, t_end = epochs_data(path_name)[0], epochs_data(path_name)[1]
    t_tks, t_tke = epochs_data(path_name)[6], epochs_data(path_name)[7]

    # Assemblies across bins
    nneu = len(spikes)
    as_across_bins, as_across_bins_index = assemblies_across_bins(patterns, bin_sizes)
    as_across_bins_pr, _ = prunning_across_bins(
        as_across_bins, as_across_bins_index, nneu, criteria="biggest", th=0.7, style="pvalue"
    )
    as_across_bins_cut = [
        as_across_bins_pr[i] for i in range(len(as_across_bins_pr)) if as_across_bins_pr[i]['bin_size'] == bin_size
    ]
    print(f"Total detected assemblies in Bin size {bin_size / 10} ms: {len(as_across_bins_cut)}")

    # Activation times
    activation_times = []
    assembly_indices = []

    for i, assembly in enumerate(as_across_bins_cut):
        activation_times += assembly['times'].tolist()
        assembly_indices += [i] * len(assembly['times'])

    sorted_times, sorted_indices = sort_activation_times(activation_times, assembly_indices)
    filtered_times, filtered_indices = filter_activation_times(sorted_times, sorted_indices, t_tks, t_tke)

    # Assembly data
    assembly_data = {
        'times': filtered_times,
        'indices': filtered_indices,
        'assemblies': as_across_bins_cut
    }

    # Plotting
    plot_raster_with_task(reach_times, spikes, nneu, t_start, t_end, assembly_data, bin_size)
