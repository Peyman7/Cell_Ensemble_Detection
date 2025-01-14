# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:25:41 2022

@author: p.nazarirobati (p.nazarirobati@uleth.ca)
### This script performs 1) trial average analysis on the assemblies activity 2) clustering activities time series 3)  sleep activity analysis
### List of funtions:
    
    - assemblies_trial_average -> get the average activity across trials
        - Input:
            data: assemblies temporal activity per each trial (np.array MxNxP, M:number of assemblies, N: number of timepoints, P:number of trials)
        - Output:
            y_avg: average activity across trials (np.array MxN)
    
    - assemblies_preprocessing -> performs preprocesing including low-quality signals removal and smoothing using LOWESS method
        - Inputs:
            y_avg: assemblies average activity across trials (np.array MxN)
            Threshold: a value to remove time series with peak less than it
            smooth_fraction: smoothing ratio in LOWESS
        - Outputs:
            y_avg_norm: time series after removal (np.array QxN)
            orig_idx -> original index of rows remianed after thresholding (index of assemblise remianed after thresholding)
    
    - assemblies_clustering -> performs Kmeans clustering to find similiar-shaped assemblies
        - Inputs:
            y_avg_norm: preprocessed array of assemblies activity (np.array QxN)
            cluster_count: number of clusters for performing kmeans
            metric: distance metric used for clsutering ("Euclidean", "dtw")
        - Outputs:
            labels: label of assemblies after clustering (np.array (Qx1))
    
    - clustering_plot -> plots assemlies activity based on their labels
        - Inputs:
            cluster_count: number of clusters
            labels: label assigned to each assembly (np.array Qx1)
            y_avg_norm: preprocessed array of assemblies activity (np.array QxN)
            x: time range data points (np.array 1xN)
            starting_time: starting time label for x axis
            end_time: ending time label for x axis
        - Outputs:
            figure shows clusters members dynamics(gray color) + average activity in each cluster (red color)
"""

####### Importing Libraries #######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tsmoothie.smoother import LowessSmoother
from assemblies_across_bins import assemblies_across_bins
from assemblies_data import assemblies_data
from spiketrain import spiketrain
from parula_colorbar import parula_colorbar
from matplotlib.colors import LinearSegmentedColormap

# Functions for Assemblies Analysis
def assemblies_trial_average(data):
    """Compute average activity across trials."""
    return np.mean(data, axis=2)

def assemblies_preprocessing(y_avg, threshold, smooth_fraction):
    """Preprocess assemblies activity with thresholding and smoothing."""
    valid_idx = [i for i, row in enumerate(y_avg) if np.max(row) > threshold]
    y_filtered = y_avg[valid_idx]
    smoother = LowessSmoother(smooth_fraction=smooth_fraction, iterations=1)
    y_smoothed = smoother.smooth(y_filtered).smooth_data
    y_normalized = TimeSeriesScalerMeanVariance().fit_transform(y_smoothed)
    return np.squeeze(y_normalized), valid_idx

def assemblies_clustering(y_avg_norm, cluster_count, metric):
    """Cluster assemblies using KMeans."""
    km = TimeSeriesKMeans(n_clusters=cluster_count, metric=metric, random_state=42)
    return km.fit_predict(y_avg_norm)

def clustering_plot(cluster_count, labels, y_avg_norm, x, start_time, end_time):
    """Plot clusters and their average dynamics."""
    fig, axs = plt.subplots(1, cluster_count, figsize=(15, 4))
    fig.suptitle("Clusters - KMeans")

    for cluster_idx in range(cluster_count):
        cluster_data = y_avg_norm[labels == cluster_idx]
        avg_cluster = np.mean(cluster_data, axis=0)
        std_cluster = np.std(cluster_data, axis=0)

        axs[cluster_idx].plot(x, avg_cluster, color='red')
        axs[cluster_idx].fill_between(x, avg_cluster - std_cluster, avg_cluster + std_cluster, color='red', alpha=0.3)
        axs[cluster_idx].set_title(f"Cluster {cluster_idx + 1}")
        axs[cluster_idx].axvline(x=0, color='green', linestyle='--')
        axs[cluster_idx].axvline(x=-start_time, color='blue', linestyle='--')
        axs[cluster_idx].axvline(x=end_time, color='blue', linestyle='--')

    plt.tight_layout()
    plt.show()

def clusters_average(y_avg_norm, labels, cluster_count):
    """Calculate average dynamics for each cluster."""
    avg_activity = np.zeros((cluster_count, np.shape(y_avg_norm)[1]))
    for cluster_idx in range(cluster_count):
        cluster_data = y_avg_norm[labels == cluster_idx]
        avg_activity[cluster_idx, :] = np.mean(cluster_data, axis=0) if len(cluster_data) > 0 else 0
    return avg_activity

def assemblies_membership_id(labels, cluster_count, orig_idx):
    """Assign original indices to cluster members."""
    idx_members = []
    for cluster_idx in range(cluster_count):
        cluster_members = [orig_idx[i] for i in np.where(labels == cluster_idx)[0]]
        idx_members.append(cluster_members)
    return idx_members

def neurons_membership(as_across_bins_cut, cluster_count, idx_members, nneu):
    """Analyze neuron membership in clusters."""
    neurons_members = [[] for _ in range(cluster_count)]
    for cluster_idx in range(cluster_count):
        cluster_data = [as_across_bins_cut[i]['neurons'] for i in idx_members[cluster_idx]]
        neurons_members[cluster_idx] = [neuron + 1 for sublist in cluster_data for neuron in sublist]
    return neurons_members

def assemblies_clusters_sleep(sleep_info_cut, epoch_duration, labels, cluster_count, idx_members):
    """Analyze cluster activity during sleep phases."""
    sleep_freq_activation_cc = []
    for cluster_idx in range(cluster_count):
        cluster_assemblies = sleep_info_cut.iloc[idx_members[cluster_idx], :]
        normalized = cluster_assemblies[['REM1', 'REM2', 'SWS1', 'SWS2']].div(epoch_duration.loc[0, ['tm_rem1', 'tm_rem2', 'tm_sws1', 'tm_sws2']] / 10000, axis=1)
        sleep_freq_activation_cc.append(normalized)
    return sleep_freq_activation_cc

def save_results(filename_prefix, **results):
    """Save results to .npy files."""
    for key, value in results.items():
        np.save(f"{filename_prefix}_{key}.npy", value)


# Main Code
if __name__ == "__main__":
    # Input Parameters
    days_id = [
        '2015-04-14', '2015-04-15', '2015-04-16',
        '2015-04-17', '2015-04-18', '2015-04-19'
    ]
    idx = 2
    dd = days_id[idx]

    # Load Data
    data = np.load(f'C:/path_to_data/rr5_AssemblyActivity_{dd}_b5100.npy', allow_pickle=True)
    reach_info = pd.read_csv(f'C:/path_to_outputs/rr5/{dd}/epochs.treach.txt', sep='\s+')
    sleep_info = pd.read_csv(f'C:/path_to_sleep/rr5_{dd}_assemblies_sleep.csv')
    epoch_duration = pd.read_csv(f'C:/path_to_duration/rr5_{dd}_epochs_duration.csv')

    # Parameters
    threshold = 0.02
    smooth_fraction = 0.15
    cluster_count = 4
    time_range = 2.5

    # Process Data
    y_avg = assemblies_trial_average(data)
    x = np.linspace(-time_range, time_range, y_avg.shape[1])
    y_avg_norm, orig_idx = assemblies_preprocessing(y_avg, threshold, smooth_fraction)
    labels = assemblies_clustering(y_avg_norm, cluster_count, metric='euclidean')

    # Visualization
    clustering_plot(cluster_count, labels, y_avg_norm, x, start_time=1, end_time=2)

    # Save Results
    save_results(f"rr5_{dd}", y_avg_norm=y_avg_norm, labels=labels, orig_idx=orig_idx)