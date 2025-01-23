# Preprocess actual calcium data
'''
This script preprocesses the calcium data before training. It performs two essential steps:

- low-pass filtering the calcium signal
- Inferring the time delay between image presentation and calcium recording

The low-pass filter is a butterworth filter.

The inference of delay happens by performing linear regression from images onto calcium signal and choosing that delay which leads to the smallest MSE on average across all ROIs.
'''
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import matplotlib as mpl
import scipy
import os

# Get the home directory
home_directory = os.path.expanduser("~")

# Set the path to the data directory where results will be stored
base_dir = f'{home_directory}/GitRepos/jaxley_rgc/deistler_our_data_and_morph'
assert os.path.exists(base_dir), f'{base_dir} does not exist.'
  
all_dfs = pd.read_pickle(f"{base_dir}/results/data/setup.pkl")

cell_id = "2020-07-08_1"



def get_stim_and_calcium(one_roi):
    """Interpolates the noise stimulus and align it temporally with calcium."""
            
    noise_times = one_roi["Triggertimes_noise"]
    calcium_at_roi = one_roi["Traces0_raw_noise"]
    times_calcium = one_roi["Tracetimes0_noise"]

    # Throw out calcium from before the noise.
    condition = times_calcium > np.min(noise_times)
    calcium_at_roi = calcium_at_roi[condition]
    times_calcium = times_calcium[condition]

    # Throw out calcium from after 200ms after the last noise trigger.
    condition = times_calcium < np.max(noise_times) + 0.2
    calcium_at_roi = calcium_at_roi[condition]
    times_calcium = times_calcium[condition]
    
    interpolated_noise = np.zeros((noise_stimulus.shape[0], noise_stimulus.shape[1], len(times_calcium)))  # np.interp(times_calcium, joined_times, joined_stim)
    for i in range(15):
        for j in range(20):
            interpolator = scipy.interpolate.interp1d(noise_times, noise_stimulus[i, j], kind="zero", fill_value="extrapolate")
            interpolated_noise[i, j, :] = interpolator(times_calcium)

    # Align calcium and stimulus.
    temporal_delay_steps = 10
    calcium_at_roi = calcium_at_roi[temporal_delay_steps:]
    times_calcium = times_calcium[temporal_delay_steps:]
    interpolated_noise = interpolated_noise[:, :, :-temporal_delay_steps]

    return interpolated_noise, calcium_at_roi, times_calcium, noise_times

def lowpass_calcium(one_roi):
    noise_times = one_roi["Triggertimes_noise"]
    calcium_at_roi = one_roi["Traces0_raw_noise"]
    times_calcium = one_roi["Tracetimes0_noise"]

    # Throw out calcium from before the noise.
    condition = times_calcium > np.min(noise_times)
    calcium_at_roi = calcium_at_roi[condition]
    times_calcium = times_calcium[condition]

    # Throw out calcium from after 400ms after the last noise trigger.
    condition = times_calcium < np.max(noise_times) + 0.4
    calcium_at_roi = calcium_at_roi[condition]
    times_calcium = times_calcium[condition]

    mean_calcium = np.mean(calcium_at_roi)
    std_calcium = np.std(calcium_at_roi)
    raw_calcium = (calcium_at_roi - mean_calcium) / std_calcium
    sos = signal.butter(4, 7.0, 'low', fs=31.25, output='sos')
    filtered_calcium = signal.sosfilt(sos, raw_calcium)
    return filtered_calcium, raw_calcium, times_calcium, noise_times


def delay_and_subsample_calcium(calcium, times_calcium, noise_times, noise_stimulus,temporal_delay_steps: int = 5):
    interpolated_noise = np.zeros((noise_stimulus.shape[0], noise_stimulus.shape[1], len(times_calcium)))
    noise_index = np.arange(noise_stimulus.shape[2])
    noise_times = noise_times[:1500]
    interpolator = scipy.interpolate.interp1d(noise_times, noise_index, kind="zero", fill_value="extrapolate")
    interpolated_index = interpolator(times_calcium)

    calcium = calcium[temporal_delay_steps:]
    times_calcium = times_calcium[temporal_delay_steps:]
    interpolated_noise_index = interpolated_index[:-temporal_delay_steps]

    diff_X = np.concatenate([np.diff(interpolated_noise_index), [0]]) > 0.0
    last_calcium_per_image = calcium[diff_X]
    last_time_per_image = times_calcium[diff_X]

    return last_calcium_per_image, last_time_per_image

def main():
    all_dfs.head()
    one_morph = all_dfs[all_dfs["cell_id"] == cell_id]

    # Read stimulus information.
    file = h5py.File(f"{base_dir}/data/noise.h5", 'r+')
    noise_stimulus = noise_stimulus = file["NoiseArray3D"][()]

    counter = 0
    for df in one_morph[:1].iterrows():
        index = df[0]
        one_roi = df[1]
        filtered_calcium, raw_calcium, times_calcium, noise_times = lowpass_calcium(one_roi)
        subsampled_calcium, subsampled_times = delay_and_subsample_calcium(
            filtered_calcium, 
            times_calcium, 
            noise_times,
            noise_stimulus, 
            temporal_delay_steps=5
        )

    # raw vs filtered calcium    
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    _ = ax.plot(times_calcium[:200], filtered_calcium[:200])
    _ = ax.plot(times_calcium[:200], raw_calcium[:200])
    _ = ax.set_xlabel("Time (seconds)")
    _ = ax.set_ylim([-6, 6])
    plt.savefig(f"{base_dir}/results/figs/raw_vs_filtered_ca.png", dpi=200, bbox_inches="tight")



    ### Low-pass filter
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    _ = ax.plot(times_calcium[:200], filtered_calcium[:200])
    _ = ax.scatter(times_calcium[:200], filtered_calcium[:200])

    for i in range(33):
        _ = ax.axvline(noise_times[i], c="gray", alpha=0.3)

    _ = ax.plot(subsampled_times[:30], subsampled_calcium[:30])
    _ = ax.scatter(subsampled_times[:30], subsampled_calcium[:30])

    _ = ax.set_xlabel("Time (seconds)")
    _ = ax.set_ylim([-6, 6])

    plt.savefig(f"{base_dir}/results/figs/raw_filtered_ca_stim_trigger.png", dpi=200, bbox_inches="tight")


    ### Perform linear regression onto the last calcium value in order to identify the optimal time lag
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from scipy.spatial.distance import correlation
    all_accs_across_rois = []
    shifts = np.arange(2, 14)
    for j, df in enumerate(one_morph.iterrows()):
        one_roi = df[1]
        filtered, _, times_calcium, noise_times = lowpass_calcium(one_roi)
        all_accs = []
        for i in shifts:
            subsampled_calcium, subsampled_times = delay_and_subsample_calcium(
                filtered, 
                times_calcium, 
                noise_times,
                noise_stimulus, 
                temporal_delay_steps=i
            )

            data = noise_stimulus.reshape((300, 1500)).T[:1498]
            target = subsampled_calcium[:1498]

            clf = LinearRegression()
            shuffle = KFold(n_splits=5, shuffle=True, random_state=0)
            scores = cross_val_score(
                clf, data, target, cv=shuffle, scoring="neg_mean_absolute_error", verbose=0
            )
            cross_val_acc = np.mean(scores)
            all_accs.append(cross_val_acc)
        print("j", j, "acc", np.asarray(all_accs) + 1.0)
        all_accs_across_rois.append(all_accs)



    average_acc = np.mean(all_accs_across_rois, axis=0) + 1.0
    print("\n\naverage_acc", average_acc)
    best_shift = shifts[np.argmax(average_acc)]
    print("BEST SHIFT: ", best_shift)

    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    _ = ax.plot(shifts, average_acc)
    plt.savefig(f"{base_dir}/results/figs/acc_vs_shifts.png", dpi=200, bbox_inches="tight")


    # Save calcium labels (low-pass filtered and time-delayed)
    inds = []
    cell_ids = []
    rec_ids = []
    roi_ids = []
    labels = []

    for df in one_morph.iterrows():
        index = df[0]
        one_roi = df[1]
        roi_ids.append(one_roi["roi_id"])
        rec_ids.append(one_roi["rec_id"])
        cell_ids.append(one_roi["cell_id"])
        inds.append(index)

        filtered, _, times_calcium, noise_times = lowpass_calcium(one_roi)
        subsampled_calcium, subsampled_times = delay_and_subsample_calcium(
            filtered, 
            times_calcium, 
            noise_times,
            noise_stimulus, 
            temporal_delay_steps=best_shift
        )
        labels.append(subsampled_calcium[:1498].tolist())
    labels = pd.DataFrame().from_dict(
        {
            "inds": inds,
            "cell_id": cell_ids,
            "rec_id": rec_ids,
            "roi_id": roi_ids,
            "ca": labels, 
        }
    )
    labels.to_pickle(f"{base_dir}/results/data/labels_lowpass_{cell_id}.pkl")


if __name__ == "__main__":
    main()
    print("Done.")


