import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

def plot_epoch_average_across_gestures(recorded_data, recorded_labels, epoch_index, sampling_rate=500):
    """
    Plots the raw averaged signal for each of the 6 gestures in a 2x3 grid.
    The x-axis is in milliseconds and the y-axis in microvolts (µV).
    
    Args:
        recorded_data (np.ndarray): Shape (n_epochs, window_length, channels).
        recorded_labels (np.ndarray): Array-like of shape (n_epochs,). Expected to be strings.
        epoch_index (int): The epoch number to plot for each gesture.
        sampling_rate (int): Sampling rate in Hz.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    gesture_list = [
        "relaxed",
        "clenched_fist",
        "open_palm",
        "upwards_rotation",
        "downwards_rotation",
        "left_rotation"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    
    for i, gesture in enumerate(gesture_list):
        gesture_epochs = recorded_data[recorded_labels == gesture]
        n_epochs = gesture_epochs.shape[0]
        ax = axes[i]
        
        if n_epochs == 0:
            ax.text(0.5, 0.5, f"No data for\n{gesture}", ha='center', va='center', fontsize=12)
            ax.set_title(gesture)
            continue
        if epoch_index >= n_epochs:
            ax.text(0.5, 0.5, f"Only {n_epochs} epochs available", ha='center', va='center', fontsize=12)
            ax.set_title(gesture)
            continue
        
        epoch = gesture_epochs[epoch_index]
        avg_signal = np.mean(epoch, axis=1)
        window_length = epoch.shape[0]
        time_axis = np.arange(window_length) / sampling_rate * 1000  # ms
        
        ax.plot(time_axis, avg_signal, label="Raw", color="tab:blue")
        ax.set_title(f"{gesture} (Epoch {epoch_index})")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc="upper right")
    
    for ax in axes[-3:]:
        ax.set_xlabel("Time (ms)")
    
    fig.suptitle(f"Raw Average Signal (across channels) for Epoch {epoch_index}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def filter_epoch(epoch, sampling_rate=500):
    """
    Applies a 4th order Butterworth bandpass filter (4.5–100 Hz)
    followed by a notch filter at 60 Hz to the given epoch.
    
    Args:
        epoch (np.ndarray): Raw epoch data with shape (window_length, channels).
        sampling_rate (int): Sampling frequency in Hz.
    
    Returns:
        np.ndarray: Filtered epoch.
    """
    nyq = sampling_rate / 2.0
    low = 4.5 / nyq
    high = 100 / nyq
    b, a = butter(4, [low, high], btype='band')
    bandpassed = filtfilt(b, a, epoch, axis=0)
    
    notch_freq = 60
    norm_notch = notch_freq / nyq
    Q = 30
    b_notch, a_notch = iirnotch(norm_notch, Q)
    filtered_epoch = filtfilt(b_notch, a_notch, bandpassed, axis=0)
    
    return filtered_epoch

def plot_epoch_average_across_gestures_filtered(recorded_data, recorded_labels, epoch_index, sampling_rate=500):
    """
    Plots the filtered averaged signal for each of the 6 gestures in a 2x3 grid.
    The x-axis is in milliseconds and the y-axis in microvolts (µV).
    
    Args:
        recorded_data (np.ndarray): Shape (n_epochs, window_length, channels).
        recorded_labels (np.ndarray): Array-like of shape (n_epochs,). Expected to be strings.
        epoch_index (int): The epoch number to plot for each gesture.
        sampling_rate (int): Sampling rate in Hz.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    gesture_list = [
        "relaxed",
        "clenched_fist",
        "open_palm",
        "upwards_rotation",
        "downwards_rotation",
        "left_rotation"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    
    for i, gesture in enumerate(gesture_list):
        gesture_epochs = recorded_data[recorded_labels == gesture]
        n_epochs = gesture_epochs.shape[0]
        ax = axes[i]
        
        if n_epochs == 0:
            ax.text(0.5, 0.5, f"No data for\n{gesture}", ha='center', va='center', fontsize=12)
            ax.set_title(gesture)
            continue
        if epoch_index >= n_epochs:
            ax.text(0.5, 0.5, f"Only {n_epochs} epochs available", ha='center', va='center', fontsize=12)
            ax.set_title(gesture)
            continue
        
        epoch = gesture_epochs[epoch_index]
        filtered_epoch = filter_epoch(epoch, sampling_rate)
        avg_signal = np.mean(filtered_epoch, axis=1)
        window_length = epoch.shape[0]
        time_axis = np.arange(window_length) / sampling_rate * 1000  # ms
        
        ax.plot(time_axis, avg_signal, label="Filtered", color="tab:orange")
        ax.set_title(f"{gesture} (Epoch {epoch_index})")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc="upper right")
    
    for ax in axes[-3:]:
        ax.set_xlabel("Time (ms)")
    
    fig.suptitle(f"Filtered Average Signal (across channels) for Epoch {epoch_index}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# -----------------------------
# Main Script
# -----------------------------
with open("emg_gestures.pkl", "rb") as f:
    recorded_data, recorded_labels = pickle.load(f)

recorded_data = np.array(recorded_data)       # Expected shape: (n_epochs, window_length, channels)
recorded_labels = np.array(recorded_labels)     # Expected to be strings

try:
    epoch_index = int(input("Enter epoch index: "))
except ValueError:
    print("Please enter a valid integer for the epoch index.")
    exit(1)

# Create both figures.
fig_raw = plot_epoch_average_across_gestures(recorded_data, recorded_labels, epoch_index, sampling_rate=500)
fig_filtered = plot_epoch_average_across_gestures_filtered(recorded_data, recorded_labels, epoch_index, sampling_rate=500)

# Display both figures simultaneously.
plt.show(block=False)
input("Press Enter to exit...")