# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:37:30 2023

@author: p.nazarirobati (p.nazarirobati@uleth.ca)

### This script computes the activation profile of assemblies across recording time.

Inputs:
    - As_across_bins: Structure containing assembly information.
    - spM: Raw spike train timestamps data.
    - BinSizes: Vector of bin sizes to be tested.
    - t_start: Recording start time (0.1 ms).
    - t_end: Recording end time (0.1 ms).
    - method: Computation method ("full", "partial", "combined", "binary").
    - LagChoice: Lag handling method ("begining", "duration").

Output:
    - Assembly_activity: List containing the activation profile of assemblies across recording time.
"""

# Importing Required Libraries
import numpy as np
import neo
import quantities as pq
from elephant.conversion import BinnedSpikeTrain

# Main Function
def AssemblyActivity(As_across_bins, spM, BinSizes, t_start, t_end, method, LagChoice):
    """
    Computes the activation profile of assemblies across recording time.

    Parameters:
    - As_across_bins: Assembly information.
    - spM: Raw spike train data.
    - BinSizes: Vector of bin sizes to test.
    - t_start, t_end: Start and end times of the recording.
    - method: "full", "partial", "combined", or "binary".
    - LagChoice: "begining" or "duration".

    Returns:
    - Assembly_activity: Activation profiles of assemblies across time.
    """
    # Validate Inputs
    if method not in ['full', 'partial', 'combined', 'binary']:
        raise ValueError("Invalid computation method. Choose from 'full', 'partial', 'combined', 'binary'.")
    if LagChoice not in ['begining', 'duration']:
        LagChoice = 'begining'  # Default LagChoice

    Assembly_activity = []  # Output list

    # Iterate through assemblies
    for i, assembly in enumerate(As_across_bins):
        print(f"Processing Assembly {i}")
        bin_size = assembly['bin_size']
        assembly_times = assembly['times']
        neurons = assembly['neurons']
        lags = assembly['lags']
        lags_scaled = [int(lag / bin_size) for lag in lags]

        # Bin spikes
        spT = [
            neo.SpikeTrain(spM[neuron] * pq.ms, t_start=t_start * pq.ms, t_stop=t_end * pq.ms)
            for neuron in neurons
        ]
        spT_bin = BinnedSpikeTrain(spT, bin_size=bin_size * pq.ms, t_start=t_start * pq.ms, t_stop=t_end * pq.ms).to_array()

        # Convert assembly times to binary format
        assembly_times_binned = BinnedSpikeTrain(
            neo.SpikeTrain(assembly_times * pq.ms, t_start=t_start * pq.ms, t_stop=t_end * pq.ms),
            bin_size=bin_size * pq.ms,
            t_start=t_start * pq.ms,
            t_stop=t_end * pq.ms,
        ).to_array()

        # Initialize shifted spike bins
        len_times = assembly_times_binned.shape[1]
        Sbin_shifted = np.full(spT_bin.shape, np.nan)

        for j, lag in enumerate(lags_scaled):
            if lag == 0:
                Sbin_shifted[j, :] = spT_bin[j, :]
            elif lag > 0:
                Sbin_shifted[j, :-lag] = spT_bin[j, lag:]
            else:
                Sbin_shifted[j, -lag:] = spT_bin[j, :len_times + lag]

        # Compute activity based on the selected method
        if method == 'full':
            Sbin_shifted[:, np.isnan(np.sum(Sbin_shifted, axis=0))] = 0
            activity = np.min(Sbin_shifted, axis=0)

        elif method == 'binary':
            Sbin_shifted[:, np.isnan(np.sum(Sbin_shifted, axis=0))] = 0
            activity = np.where(np.min(Sbin_shifted, axis=0) >= 1, 1, 0)

        elif method in ['combined', 'partial']:
            max_rate = int(np.max(spT_bin))
            Sbin_p = [(spT_bin >= level + 1).astype(int) for level in range(max_rate)]
            Asactivity = np.zeros((max_rate, spT_bin.shape[1]))

            for level in range(max_rate):
                for j, lag in enumerate(lags_scaled):
                    if lag == 0:
                        Sbin_shifted[j, :] = Sbin_p[level][j, :]
                    elif lag > 0:
                        Sbin_shifted[j, :-lag] = Sbin_p[level][j, lag:]
                    else:
                        Sbin_shifted[j, -lag:] = Sbin_p[level][j, :len_times + lag]

                Asactivity[level, :] = np.sum(Sbin_shifted, axis=0)
                Asactivity[level, Asactivity[level, :] == 1] = 0

            activation_total = np.sum(Asactivity, axis=0)
            activation_total[np.isnan(activation_total)] = 0

            if method == 'combined':
                activity = activation_total / (len(neurons) ** 5)
            else:  # partial
                activity = activation_total / len(neurons)

        # Modify activity based on LagChoice
        if LagChoice == 'begining':
            Assembly_activity.append(activity)
        elif LagChoice == 'duration':
            activity_lag = activity
            for kk in range(1, lags_scaled[-1] + 1):
                arr_lag = np.hstack((np.zeros(kk), activity[:-kk]))
                activity_lag = np.vstack((activity_lag, arr_lag))
            Assembly_activity.append(np.sum(activity_lag, axis=0) if lags_scaled[-1] > 0 else activity_lag)

    return Assembly_activity
