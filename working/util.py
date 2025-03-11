from scipy import signal
import numpy as np
import time
import matplotlib.pyplot as plt

def filter_emg_data(data, fs):
    """
    Filters EMG data using a 60 Hz notch filter (Q=30) and a 4th order Butterworth bandpass filter (10–200 Hz).

    Parameters:
      data : numpy.ndarray
          Input data (n_samples, n_channels).
      fs : float
          Sampling rate in Hz.

    Returns:
      filtered_data : numpy.ndarray
          Filtered data (n_samples, n_channels).
    """
    # Notch Filter at 60 Hz
    notch_freq = 60.0
    Q = 30.0  # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    
    # 4th Order Butterworth Bandpass Filter (10 - 200 Hz)
    low_fc = 10.0
    high_fc = 200.0
    order = 4
    nyq = 0.5 * fs
    low = low_fc / nyq
    high = high_fc / nyq
    b_band, a_band = signal.butter(order, [low, high], btype='band')
    
    # Filter each channel individually
    filtered_data = np.empty_like(data)
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        filtered_channel = signal.filtfilt(b_notch, a_notch, channel_data)
        filtered_channel = signal.filtfilt(b_band, a_band, filtered_channel)
        filtered_data[:, ch] = filtered_channel
        
    return filtered_data

def compute_fft(data, fs):
    """
    Computes the FFT of the given data along the time axis (axis=0).

    Parameters:
      data : numpy.ndarray
          Input data (n_samples, n_channels).
      fs : float
          Sampling rate in Hz.

    Returns:
      fft_data : numpy.ndarray
          The FFT of the input data.
      freqs : numpy.ndarray
          The corresponding frequency bins.
    """
    fft_data = np.fft.rfft(data, axis=0)
    freqs = np.fft.rfftfreq(data.shape[0], d=1/fs)
    return fft_data, freqs

def calibrate(state, num_samples, num_points_per_sample, num_channels, board_shim):
    if num_samples == 0 or num_points_per_sample == 0:
        raise ValueError("Error: Samples set to 0")
    
    samples = np.zeros((num_samples, num_channels))
    print("Do", state, "pose for", num_samples, "samples in...")
    time.sleep(1)
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("GO!")
    
    sample_num = 0
    while sample_num < num_samples:
        num_points = 0
        points = np.zeros((num_points_per_sample, num_channels))
        
        while num_points < num_points_per_sample:
            while board_shim.get_board_data_count() < 1:
                time.sleep(0.001)
            sample = board_shim.get_board_data(1)
            sample = sample[:num_channels]
            points[num_points] = sample.squeeze()
            num_points += 1
        points = filter_emg_data(points, 500)
        print(f"{sample_num}/{num_samples}")
        mean_of_points = np.mean(np.abs(points), axis=0)
        samples[sample_num] = mean_of_points
        sample_num += 1
            
    return samples

def classify(sample, centroids):
    min_distance = float("inf")
    state = -1
    for i in range(centroids.shape[0]):
        dist = np.linalg.norm(centroids[i] - sample)
        if dist < min_distance:
            min_distance = dist
            state = i
    return state

def plot_data_per_class(raw_data, fs, class_label=""):
    """
    Plots a figure for one raw data sample for a given class.
    The figure has one row per channel and four columns:
      Column 1: Raw time series for that channel.
      Column 2: Filtered time series for that channel.
      Column 3: Frequency spectrum (FFT magnitude) of raw data.
      Column 4: Frequency spectrum (FFT magnitude) of filtered data.

    Parameters:
      raw_data : numpy.ndarray
          Array of shape (n_samples, n_channels).
      fs : float
          Sampling rate in Hz.
      class_label : str, optional
          Label for the class (displayed in the figure title).

    Returns:
      fig : matplotlib.figure.Figure
          The generated figure.
    """
def plot_data_per_class(raw_data, fs, class_label=""):
    """
    Plots a figure for one raw data sample for a given class.
    The figure has one row per channel and four columns:
      Column 1: Raw time series for that channel.
      Column 2: Filtered time series for that channel.
      Column 3: Frequency spectrum (FFT magnitude) of raw data.
      Column 4: Frequency spectrum (FFT magnitude) of filtered data.

    Parameters:
      raw_data : numpy.ndarray
          Array of shape (n_samples, n_channels).
      fs : float
          Sampling rate in Hz.
      class_label : str, optional
          Label for the class (displayed in the figure title).

    Returns:
      fig : matplotlib.figure.Figure
          The generated figure.
    """
    n_samples, n_channels = raw_data.shape
    # Filter the entire data
    filtered_data = filter_emg_data(raw_data, fs)
    
    fig, axs = plt.subplots(n_channels, 4, figsize=(16, 4 * n_channels))
    if n_channels == 1:
        axs = np.array([axs])
    
    for ch in range(n_channels):
        time_axis = np.arange(n_samples) / fs * 1000  # in milliseconds
        raw_channel = raw_data[:, ch]
        filt_channel = filtered_data[:, ch]
        
        # Compute FFT for raw channel
        fft_raw = np.fft.rfft(raw_channel)
        freqs_raw = np.fft.rfftfreq(n_samples, d=1/fs)
        
        # Compute FFT for filtered channel
        fft_filt = np.fft.rfft(filt_channel)
        freqs_filt = np.fft.rfftfreq(n_samples, d=1/fs)
        
        # Plot raw time series
        axs[ch, 0].plot(time_axis, raw_channel, color="tab:blue")
        axs[ch, 0].set_title(f"Ch {ch+1} Raw")
        axs[ch, 0].set_xlabel("Time (ms)")
        axs[ch, 0].set_ylabel("Amplitude")
        
        # Plot filtered time series
        axs[ch, 1].plot(time_axis, filt_channel, color="tab:orange")
        axs[ch, 1].set_title(f"Ch {ch+1} Filtered")
        axs[ch, 1].set_xlabel("Time (ms)")
        axs[ch, 1].set_ylabel("Amplitude")
        
        # Plot FFT magnitude of raw channel
        axs[ch, 2].plot(freqs_raw, np.abs(fft_raw), color="tab:blue")
        axs[ch, 2].set_title(f"Ch {ch+1} Raw FFT")
        axs[ch, 2].set_xlabel("Frequency (Hz)")
        axs[ch, 2].set_ylabel("Magnitude")
        
        # Plot FFT magnitude of filtered channel
        axs[ch, 3].plot(freqs_filt, np.abs(fft_filt), color="tab:orange")
        axs[ch, 3].set_title(f"Ch {ch+1} Filtered FFT")
        axs[ch, 3].set_xlabel("Frequency (Hz)")
        axs[ch, 3].set_ylabel("Magnitude")
        
    fig.suptitle(f"EMG Data for Class: {class_label}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig