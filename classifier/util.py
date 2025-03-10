from scipy import signal
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
import numpy as np
import time

def notch(data,samping_rate):
    #TODO
    pass

def band_pass(data,sampling_rate):
    #TODO
    pass

def fft():
    #TODO
    pass

def filter_emg_data(data, fs):
    """
    Apply a notch filter at 60 Hz and a bandpass filter from 10 Hz to 200 Hz.
    
    data: numpy array of shape (n_samples, n_channels)
    fs: sampling rate (Hz)
    """
    # --- Notch filter (60 Hz) ---
    notch_freq = 60.0  # Frequency to be removed from signal (Hz)
    Q = 30.0           # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    
    # --- Bandpass filter (10 Hz - 200 Hz) ---
    lowcut = 10.0
    highcut = 200.0
    order = 2
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b_band, a_band = signal.butter(order, [low, high], btype='band')
    
    # Apply filters channel-by-channel using zero-phase filtering.
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        # Apply notch filter
        channel_data = signal.filtfilt(b_notch, a_notch, channel_data)
        # Apply bandpass filter
        channel_data = signal.filtfilt(b_band, a_band, channel_data)
        filtered_data[:, ch] = channel_data
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