# filtering.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def filter_epoch(epoch, sampling_rate=500):
    """
    Applies a 4th‑order Butterworth bandpass filter (4.5–100 Hz)
    followed by a 60 Hz notch filter to the given epoch.

    Args:
        epoch (np.ndarray): Raw epoch data with shape (window_length, channels).
        sampling_rate (int): Sampling frequency in Hz.

    Returns:
        np.ndarray: Filtered epoch.
    """
    nyq = sampling_rate / 2.0
    # Bandpass filter from 4.5 to 100 Hz.
    low = 4.5 / nyq
    high = 100 / nyq
    b, a = butter(4, [low, high], btype='band')
    bandpassed = filtfilt(b, a, epoch, axis=0)

    # Notch filter at 60 Hz.
    notch_freq = 60
    norm_notch = notch_freq / nyq
    Q = 30  # Quality factor; adjust as needed.
    b_notch, a_notch = iirnotch(norm_notch, Q)
    filtered_epoch = filtfilt(b_notch, a_notch, bandpassed, axis=0)

    return filtered_epoch
