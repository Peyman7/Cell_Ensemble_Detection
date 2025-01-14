# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:10:52 2022

@author: p.nazarirobati 

This script computes the Peri-Spindle Time Histogram of Sharp-Wave Ripples (SWRs).

Inputs:
    - Bin resolution: Temporal resolution of sleep times binning.
    - Bin size: For creating peri-event time histograms.
    - SWR threshold: Maximum duration for SWRs.
    - Rat IDs and days for analysis.
    - Sleep and event epoch data.

Outputs:
    - Peri-Spindle Event Time Histograms for each rat and day.
    - Statistics on SWRs, spindles, and k-complex spindles.
"""

# Importing Required Libraries
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from scipy import signal

# Global Parameters
BIN_RESOLUTION = 0.0005  # Unit: seconds
BIN_SIZE = 0.01  # Unit: seconds
N = 1  # Time window around spindle onset (Unit: seconds)
THRESHOLD = 0.15  # Unit: seconds
RATS = ['rr5', 'rr6', 'rr8', 'rr9', 'rr7']
DAYS_IDX = [
    '2016-04-20', '2016-04-21', '2016-04-22', '2016-04-23', '2016-04-24',
    '2016-04-25', '2016-04-26', '2016-04-27', '2016-04-28', '2016-04-29',
    '2016-05-01', '2016-05-02', '2016-05-03', '2016-05-04', '2016-05-05'
]
F_SAMPLING = int(1 / BIN_RESOLUTION)  # Sampling frequency (Hz)

# Functions


def load_epochs(file_path, reshape_factor):
    """
    Load epochs from a file and reshape them.

    Parameters:
    - file_path: Path to the file.
    - reshape_factor: Factor to reshape the data.

    Returns:
    - Reshaped epoch data.
    """
    with open(file_path, 'r') as file:
        data = np.fromstring(file.read(), dtype=np.float64, sep=' ')
    return np.reshape(data, (int(len(data) / reshape_factor), reshape_factor))


def bin_events(events, sleep_binned, f_sampling):
    """
    Bin events based on sleep epochs.

    Parameters:
    - events: Event data (e.g., spindles, SWRs).
    - sleep_binned: Binned sleep times.
    - f_sampling: Sampling frequency.

    Returns:
    - Binned events.
    """
    binned_events = np.zeros((1, len(sleep_binned)))
    for event in events:
        indices = np.where(sleep_binned >= event[0])[0]
        if indices.size > 0:
            binned_events[0, indices[0]] = 1
    return binned_events


def calculate_peth(spindles_in_swr, bin_count):
    """
    Calculate the Peri-Spindle Event Time Histogram.

    Parameters:
    - spindles_in_swr: Spindles binned within SWR epochs.
    - bin_count: Number of bins.

    Returns:
    - Mean and standard deviation of the PETH.
    """
    split_splits = [np.hsplit(spindle, bin_count) for spindle in spindles_in_swr]
    split_sums = [[np.sum(segment) for segment in splits] for splits in split_splits]
    peth_mean = np.mean(split_sums, axis=0)
    peth_std = np.std(split_sums, axis=0)
    return peth_mean, peth_std


def plot_peth(peth_mean, time_range, rat, day, output_path):
    """
    Plot the Peri-Spindle Event Time Histogram.

    Parameters:
    - peth_mean: Mean PETH data.
    - time_range: Time range for the histogram.
    - rat: Rat ID.
    - day: Day of recording.
    - output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(time_range, peth_mean, width=BIN_SIZE, align='edge', color='blue', edgecolor='blue')
    ax.axvline(x=0, linestyle='--', color='red', linewidth=1)
    ax.set_xlabel('Time from SWR onset (sec)')
    ax.set_ylabel('Average spindle count')
    ax.set_title(f'PETH for {rat}, {day}')
    plt.savefig(output_path)
    plt.close(fig)


# Main Execution Loop
def main():
    results = {'num_swr': [], 'num_spindle': [], 'num_kspindle': []}
    rat = RATS[3]  # Specify the rat to analyze
    y_all_mean, y_all_std = [], []

    for day in DAYS_IDX:
        spindle_path = f'C:/Users/p.nazarirobati/Desktop/outputs/{rat}/{day}'
        epoch_path = f'C:/Users/p.nazarirobati/Desktop/epochs duration/{rat}/{rat}_{day}_epochs_duration.csv'

        sp_epoch = pd.read_csv(epoch_path)
        sleep_epochs = load_epochs(f'{spindle_path}/epochs.sleep2', reshape_factor=2)

        spindle = load_epochs(f'{spindle_path}/epochs.tslvs2', reshape_factor=2) / 10000
        kspindle = load_epochs(f'{spindle_path}/epochs.tslvsk2', reshape_factor=2) / 10000
        swr = load_epochs(f'{spindle_path}/epochs.tsspw2', reshape_factor=2) / 10000

        results['num_spindle'].append(spindle.shape[0])
        results['num_kspindle'].append(kspindle.shape[0])

        spindles_sorted = kspindle[kspindle[:, 0].argsort()]
        sleep_binned = np.arange(sleep_epochs[0, 0] / 10000, sleep_epochs[0, 1] / 10000, BIN_RESOLUTION)

        spindles_binned = bin_events(spindles_sorted, sleep_binned, F_SAMPLING)
        swr_binned = bin_events(swr, sleep_binned, F_SAMPLING)

        swr_epochs_idx = [
            idx for idx in np.where(swr_binned == 1)[1]
            if F_SAMPLING * N <= idx < len(swr_binned[0]) - F_SAMPLING * N
        ]
        spindles_in_swr = np.array([spindles_binned[0, idx - F_SAMPLING * N:idx + F_SAMPLING * N]
                                    for idx in swr_epochs_idx])

        bin_count = int(2 * N / BIN_SIZE)
        peth_mean, peth_std = calculate_peth(spindles_in_swr, bin_count)
        y_all_mean.append(peth_mean)
        y_all_std.append(peth_std)

        plot_peth(
            peth_mean,
            np.arange(-N, N, BIN_SIZE),
            rat,
            day,
            output_path=f'{rat}_{day}_peth_plot.png'
        )

        results['num_swr'].append(len(spindles_in_swr))

    # Save Results
    with open(f'{rat}_results.pkl', 'wb') as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    main()
