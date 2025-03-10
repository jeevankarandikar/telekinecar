from scipy import signal
import numpy as np
import time

def filter_emg_data(data, fs, fft=False):
    """
    Filters EMG data using a 60 Hz notch filter (Q=30) and a 4th order Butterworth bandpass filter (10–200 Hz).

    Parameters:
      data : numpy.ndarray
          Input data (n_samples, n_channels).
      fs : float
          Sampling rate in Hz.
      fft : bool, optional
          If True, returns FFT data with frequency bins.

    Returns:
      If fft is False:
          Filtered data (n_samples, n_channels).
      If fft is True:
          Tuple of (fft_data, freqs).
    """

    # Notch Filter at 60Hz
    notch_freq = 60.0
    Q = 30.0 # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    
    # 4th Order Butterworth Bandpass Filter (10 - 200 Hz)
    low_fc = 10.0
    high_fc = 200.0
    order = 4
    nyq = 0.5 * fs
    low = low_fc / nyq
    high = high_fc / nyq
    b_band, a_band = signal.butter(order, [low, high], btype='band')
    
    # Filter Each Channel
    filtered_data = np.empty_like(data)
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        filtered_channel = signal.filtfilt(b_notch, a_notch, channel_data)
        filtered_channel = signal.filtfilt(b_band, a_band, filtered_channel)
        filtered_data[:, ch] = filtered_channel
    
    if fft:
        # rfft returns only real frequencies
        fft_data = np.fft.rfft(filtered_data, axis=0)
        freqs = np.fft.rfftfreq(filtered_data.shape[0], d=1/fs)
        return fft_data, freqs
    else:
        return filtered_data


def calibrate(state,num_samples,num_points_per_sample,num_channels, board_shim):

    if num_samples == 0 or num_points_per_sample == 0:
        raise ValueError("Error: Samples set to 0")
    

    samples = np.zeros((num_samples,num_channels))
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
        points = np.zeros((num_points_per_sample,num_channels))

        while num_points < num_points_per_sample:
            while board_shim.get_board_data_count() < 1:
                time.sleep(0.001)
            sample = board_shim.get_board_data(1)
            sample = sample[:num_channels]
            points[num_points] = sample.squeeze()
            num_points += 1
        points = filter_emg_data(points,500)
        print(f"{sample_num}/{num_samples}")
        mean_of_points = np.mean(np.abs(points),axis=0)
        samples[sample_num] = mean_of_points
        sample_num += 1
            
    return samples

def classify(sample,centroids):
    min_distance = float("inf")
    state = -1
    for i in range(centroids.shape[0]):
        dist = np.linalg.norm(centroids[i] - sample)
        if dist < min_distance:
            min_distance = dist
            state = i
    return state