from scipy import signal
import numpy as np
import time
import matplotlib.pyplot as plt

def filter_emg_data(data, fs):
    """
    Filters EMG data with a 60 Hz notch filter (Q=30) and a 4th order Butterworth bandpass filter (10–200 Hz)

    Parameters
      data: Input data array (n_samples, n_channels)
      fs: Sampling rate in Hz

    Returns
      Filtered data (n_samples, n_channels)
    """
    notch_freq = 60.0
    Q = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    
    low_fc = 10.0
    high_fc = 200.0
    order = 4
    nyq = 0.5 * fs
    low = low_fc / nyq
    high = high_fc / nyq
    b_band, a_band = signal.butter(order, [low, high], btype='band')
    
    filtered_data = np.empty_like(data)
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        filtered_channel = signal.filtfilt(b_notch, a_notch, channel_data)
        filtered_channel = signal.filtfilt(b_band, a_band, filtered_channel)
        filtered_data[:, ch] = filtered_channel
    return filtered_data

def compute_fft(data, fs):
    """
    Computes the FFT of the data along the time axis

    Parameters
      data: Input data array (n_samples, n_channels)
      fs: Sampling rate in Hz

    Returns
      fft_data: FFT of the data
      freqs: Frequency bins corresponding to the FFT
    """
    fft_data = np.fft.rfft(data, axis=0)
    freqs = np.fft.rfftfreq(data.shape[0], d=1/fs)
    return fft_data, freqs

def calibrate(state, num_samples, num_points_per_sample, num_channels, board_shim):
    """
    Collects num_samples of data for a given gesture

    Parameters
      state: Gesture state (e.g. "rest", "flex", etc)
      num_samples: Number of samples to collect
      num_points_per_sample: Number of data points per sample
      num_channels: Number of channels in the data
      board_shim: Object to interact with the board

    Returns
      Collected samples (num_samples, num_points_per_sample, num_channels)
    """
    samples = np.zeros((num_samples, num_points_per_sample, num_channels))
    print("Do", state, "pose for", num_samples, "samples in...")
    time.sleep(1)
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("GO!")
    
    for sample_num in range(num_samples):
        points = np.zeros((num_points_per_sample, num_channels))
        for pt in range(num_points_per_sample):
            while board_shim.get_board_data_count() < 1:
                time.sleep(0.001)
            sample = board_shim.get_board_data(1)
            sample = sample[:num_channels]
            points[pt] = sample.squeeze()
        print(f"{sample_num}/{num_samples}")
        samples[sample_num] = points
    return samples

def classify(sample, centroids):
    """
    Classifies a feature vector by finding the nearest centroid

    Parameters
      sample: Feature vector to classify
      centroids: Array of centroid vectors (n_classes, feature_dim)

    Returns
      Index of the nearest centroid
    """
    min_distance = float("inf")
    label = -1
    for i in range(centroids.shape[0]):
        dist = np.linalg.norm(centroids[i] - sample)
        if dist < min_distance:
            min_distance = dist
            label = i
    return label

def plot_data_per_class(raw_data, fs, class_label=""):
    """
    Plots raw and filtered EMG data and their FFTs for each channel

    Parameters
      raw_data: Array (n_samples, n_channels) with the raw EMG data
      fs: Sampling rate in Hz
      class_label: Label for the class (optional)

    Returns
      A matplotlib figure with the plots
    """
    n_samples, n_channels = raw_data.shape
    filtered_data = filter_emg_data(raw_data, fs)
    
    fig, axs = plt.subplots(n_channels, 4, figsize=(16, 4 * n_channels))
    if n_channels == 1:
        axs = np.array([axs])
    for ch in range(n_channels):
        time_axis = np.arange(n_samples) / fs * 1000
        raw_channel = raw_data[:, ch]
        filt_channel = filtered_data[:, ch]
        fft_raw = np.fft.rfft(raw_channel)
        freqs_raw = np.fft.rfftfreq(n_samples, d=1/fs)
        fft_filt = np.fft.rfft(filt_channel)
        freqs_filt = np.fft.rfftfreq(n_samples, d=1/fs)
        
        axs[ch, 0].plot(time_axis, raw_channel, color="tab:blue")
        axs[ch, 0].set_title(f"Ch {ch+1} Raw")
        axs[ch, 0].set_xlabel("Time (ms)")
        axs[ch, 0].set_ylabel("Amplitude")
        
        axs[ch, 1].plot(time_axis, filt_channel, color="tab:orange")
        axs[ch, 1].set_title(f"Ch {ch+1} Filt")
        axs[ch, 1].set_xlabel("Time (ms)")
        axs[ch, 1].set_ylabel("Amplitude")
        
        axs[ch, 2].plot(freqs_raw, np.abs(fft_raw), color="tab:blue")
        axs[ch, 2].set_title(f"Ch {ch+1} Raw FFT")
        axs[ch, 2].set_xlabel("Freq (Hz)")
        axs[ch, 2].set_ylabel("Magnitude")
        
        axs[ch, 3].plot(freqs_filt, np.abs(fft_filt), color="tab:orange")
        axs[ch, 3].set_title(f"Ch {ch+1} Filt FFT")
        axs[ch, 3].set_xlabel("Freq (Hz)")
        axs[ch, 3].set_ylabel("Magnitude")
    fig.suptitle(f"EMG Data for Class: {class_label}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig