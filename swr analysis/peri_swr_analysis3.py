# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:30:50 2023

@author: p.nazarirobati (p.nazarirobati@uleth.ca)

This script analyzes assemblies' activity around events (e.g., SWR) and random shuffled events.

Inputs:
    - Temporal resolution, Gaussian kernel parameters, and cluster profiles.
    - Rat-specific and day-specific data for analysis.

Outputs:
    - Peri-ripple activity plots for clusters across days.
    - Normalized activity profiles compared to shuffled random events.

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats.distributions import t as tdist

# Constants
NUM_SHUFFLING = 1000  # Number of shuffled data
TEMPORAL_RESOLUTION = 20  # 0.1 ms
T_SHIFT1 = 0.15  # Time before event (seconds)
T_SHIFT2 = 0.15  # Time after event (seconds)
SIGMA = 5  # Gaussian kernel std for smoothing
CATEGORY = ['pre-reach activated', 'reach activated', 'reach inactivated', 'post-reach activated']
GAUSSIAN_SIZE = int(np.ceil(6 * SIGMA)) + 1
RATS = ['rr5', 'rr6', 'rr8', 'rr9', 'rr7']
DAYS_IDX = [
    '2015-04-14', '2015-04-15', '2015-04-16', '2015-04-17', '2015-04-18',
    '2015-04-19', '2015-04-20', '2015-04-21', '2015-04-22', '2015-04-23',
    '2015-04-24', '2015-04-25', '2015-04-26', '2015-04-27', '2015-04-28'
]
NUM_CLUSTERS = 4

# Functions


def gaussian_filter1d(size, sigma):
    """Create a 1D Gaussian filter."""
    filter_range = np.linspace(-int(size / 2), int(size / 2), size)
    return np.exp(-filter_range**2 / (2 * sigma**2))


def load_cluster_members(rat, day_id):
    """
    Load assemblies ID per cluster for a given rat and day.

    Parameters:
    - rat: Rat ID.
    - day_id: Day ID.

    Returns:
    - Assemblies member IDs.
    """
    filepath = f'C:/Users/p.nazarirobati/Desktop/Analysis/trial-average profile/Clustering/{rat}/idx_members/{rat}_idx_members_{day_id}.npy'
    return np.load(filepath, allow_pickle=True)


def load_event_activity(filepath):
    """
    Load assemblies' activity around specific events.

    Parameters:
    - filepath: Path to the pickle file.

    Returns:
    - Data containing assemblies' activity.
    """
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def calculate_peri_event_norm(activity, random_events, num_shuffling):
    """
    Normalize assemblies' activity relative to random events.

    Parameters:
    - activity: Activity around specific events.
    - random_events: Activity around shuffled events.
    - num_shuffling: Number of shuffled iterations.

    Returns:
    - Normalized peri-event activity.
    """
    tensor_random = np.array([random_events[i] for i in range(num_shuffling)])
    avg_random = np.mean(tensor_random, axis=0)
    std_random = np.std(tensor_random, axis=0)

    norm_activity = (activity - avg_random) / (std_random + 1e-12)

    # Cap extreme outliers
    norm_activity[norm_activity > 20] = 0
    return norm_activity


def plot_peri_event_activity(
    t, peri_event_activity, num_days, num_clusters, category, output_path
):
    """
    Plot peri-event activity for clusters across days.

    Parameters:
    - t: Time vector.
    - peri_event_activity: Normalized activity data.
    - num_days: Number of days.
    - num_clusters: Number of clusters.
    - category: List of cluster categories.
    - output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(num_clusters, num_days, figsize=(25, 15), sharex=True, sharey=True)

    for i in range(num_days):
        for j in range(num_clusters):
            activity = peri_event_activity[i][j]
            if activity is not None:
                mean_activity = np.mean(activity, axis=0)
                std_activity = np.std(activity, axis=0)

                smoothed_activity = np.convolve(mean_activity, gaussian_filter1d(GAUSSIAN_SIZE, SIGMA), mode='same')

                ax[j, i].plot(t, smoothed_activity, color='blue')
                ax[j, i].fill_between(
                    t, smoothed_activity - std_activity, smoothed_activity + std_activity,
                    color='blue', alpha=0.35, edgecolor='white'
                )
                ax[j, i].axvline(x=0, linestyle='--', linewidth=0.5, color='green')

            if j == int(num_clusters / 2):
                ax[j, 0].set_ylabel('Averaged z-scored peri-event activity', fontsize=8)
            if i == int(num_days / 2):
                ax[0, i].set_title(f'Day {i + 1}', fontsize=8)
                ax[j, i].set_title(category[j], fontsize=10)

    fig.text(0.4, 0.065, 'Time relative to event center (sec)')
    plt.savefig(output_path)
    plt.show()


# Main Code
def main():
    rat = RATS[0]
    num_days = len(DAYS_IDX)
    cluster_members = [load_cluster_members(rat, day_id) for day_id in DAYS_IDX]

    # Load event-triggered data
    event_activity_path = f'C:/Users/p.nazarirobati/Desktop/Analysis/swr_activation/swr/swr_avg/{rat}_Assembly_SWRAvg.pkl'
    event_activity = load_event_activity(event_activity_path)['CaActivity_swrAvg']

    # Load shuffled data
    random_event_data = []
    for i in range(NUM_SHUFFLING):
        random_activity_path = (
            f'C:/Users/p.nazarirobati/Desktop/Analysis/swr_activation/random_events/rand_avg/{rat}/{rat}_Assembly_randAvg{i + 1}.pkl'
        )
        random_event_data.append(load_event_activity(random_activity_path)['CaActivity_randAvg'])

    # Normalize activity and organize data
    peri_event_activity = []
    for day_idx in range(num_days):
        activity = np.array(event_activity[day_idx])
        random_events = [random_event_data[itr][day_idx] for itr in range(NUM_SHUFFLING)]

        normalized_activity = calculate_peri_event_norm(activity, random_events, NUM_SHUFFLING)
        peri_event_activity.append([normalized_activity[cluster] for cluster in cluster_members[day_idx]])

    # Time vector
    time_vector = np.linspace(-T_SHIFT1, T_SHIFT2, normalized_activity.shape[1])

    # Plot
    output_path = f'{rat}_peri_event_activity_plot.png'
    plot_peri_event_activity(time_vector, peri_event_activity, num_days, NUM_CLUSTERS, CATEGORY, output_path)


if __name__ == "__main__":
    main()
