from utils import calibrate, classify, filter_emg_data, plot_data_per_class, compute_fft
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters
num_samples_per_class = 100
num_points_per_sample = 50  # Must be >= 27 for the IIR filter padding
train_ratio = 0.8
num_classes = 3
num_channels = 4
states = {0: "rest", 1: "flex", 2: "spiderman"}
fs = 500

def main():
    """
    Main flow for calibration, training, and inference of EMG data
    """
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)
    
    board_shim.prepare_session()
    board_shim.start_stream(1)
    print("Connected to MindRove board and started stream")
    
    train_size = int(num_samples_per_class * train_ratio)
    test_size = num_samples_per_class - train_size

    # Collect calibration data for each gesture
    # Each sample has shape (num_points_per_sample, num_channels)
    raw_data_all = np.zeros((num_classes, num_samples_per_class, num_points_per_sample, num_channels))
    for i in range(num_classes):
        print(f"Calibrating gesture '{states[i]}'")
        samples = calibrate(state=states[i],
                            num_samples=num_samples_per_class,
                            num_points_per_sample=num_points_per_sample,
                            num_channels=num_channels,
                            board_shim=board_shim)
        raw_data_all[i] = samples
        input(f"Calibration for gesture '{states[i]}' complete. Press Enter to continue to the next gesture")

    # Split data into training and testing sets
    train_data_raw = np.zeros((num_classes, train_size, num_points_per_sample, num_channels))
    test_data_raw  = np.zeros((num_classes, test_size, num_points_per_sample, num_channels))
    for i in range(num_classes):
        rng = np.random.default_rng()
        indices = rng.choice(num_samples_per_class, size=num_samples_per_class, replace=False)
        train_data_raw[i] = raw_data_all[i][indices[:train_size]]
        test_data_raw[i]  = raw_data_all[i][indices[train_size:]]
    
    # Build features by averaging absolute values over the time axis
    train_feats_raw_time = np.zeros((num_classes, train_size, num_channels))
    test_feats_raw_time  = np.zeros((num_classes, test_size, num_channels))
    train_feats_raw_freq = np.zeros((num_classes, train_size, num_channels))
    test_feats_raw_freq  = np.zeros((num_classes, test_size, num_channels))
    train_feats_filt_time = np.zeros((num_classes, train_size, num_channels))
    test_feats_filt_time  = np.zeros((num_classes, test_size, num_channels))
    train_feats_filt_freq = np.zeros((num_classes, train_size, num_channels))
    test_feats_filt_freq  = np.zeros((num_classes, test_size, num_channels))
    
    # Process training samples
    for i in range(num_classes):
        for j in range(train_size):
            sample = train_data_raw[i, j]  # shape (num_points_per_sample, num_channels)
            # Raw time feature: average absolute value per channel
            train_feats_raw_time[i, j] = np.mean(np.abs(sample), axis=0)
            # Raw frequency feature: average absolute FFT per channel
            fft_raw = np.fft.rfft(sample, axis=0)
            train_feats_raw_freq[i, j] = np.mean(np.abs(fft_raw), axis=0)
            # Filtered time feature
            sample_filt = filter_emg_data(sample, fs)
            train_feats_filt_time[i, j] = np.mean(np.abs(sample_filt), axis=0)
            # Filtered frequency feature
            fft_filt = np.fft.rfft(sample_filt, axis=0)
            train_feats_filt_freq[i, j] = np.mean(np.abs(fft_filt), axis=0)
    
    # Process testing samples
    for i in range(num_classes):
        for j in range(test_size):
            sample = test_data_raw[i, j]
            test_feats_raw_time[i, j] = np.mean(np.abs(sample), axis=0)
            fft_raw = np.fft.rfft(sample, axis=0)
            test_feats_raw_freq[i, j] = np.mean(np.abs(fft_raw), axis=0)
            sample_filt = filter_emg_data(sample, fs)
            test_feats_filt_time[i, j] = np.mean(np.abs(sample_filt), axis=0)
            fft_filt = np.fft.rfft(sample_filt, axis=0)
            test_feats_filt_freq[i, j] = np.mean(np.abs(fft_filt), axis=0)
    
    # Compute centroids from training features for each gesture
    centroids_raw_time = np.zeros((num_classes, num_channels))
    centroids_raw_freq = np.zeros((num_classes, num_channels))
    centroids_filt_time = np.zeros((num_classes, num_channels))
    centroids_filt_freq = np.zeros((num_classes, num_channels))
    for i in range(num_classes):
        centroids_raw_time[i] = np.mean(train_feats_raw_time[i], axis=0)
        centroids_raw_freq[i] = np.mean(train_feats_raw_freq[i], axis=0)
        centroids_filt_time[i] = np.mean(train_feats_filt_time[i], axis=0)
        centroids_filt_freq[i] = np.mean(train_feats_filt_freq[i], axis=0)
    
    # Evaluate classification accuracy using Euclidean distance
    def evaluate_accuracy(test_feats, centroids):
        correct = 0
        total = 0
        for i in range(num_classes):
            for feat in test_feats[i]:
                dists = np.linalg.norm(centroids - feat, axis=0)
                pred = np.argmin(dists)
                if pred == i:
                    correct += 1
                total += 1
        return correct / total
    
    acc_raw_time = evaluate_accuracy(test_feats_raw_time, centroids_raw_time)
    acc_raw_freq = evaluate_accuracy(test_feats_raw_freq, centroids_raw_freq)
    acc_filt_time = evaluate_accuracy(test_feats_filt_time, centroids_filt_time)
    acc_filt_freq = evaluate_accuracy(test_feats_filt_freq, centroids_filt_freq)
    
    print("Classification Accuracies:")
    print(f"Raw Time Series: {acc_raw_time*100:.2f}%")
    print(f"Raw Frequency Series: {acc_raw_freq*100:.2f}%")
    print(f"Filtered Time Series: {acc_filt_time*100:.2f}%")
    print(f"Filtered Frequency Series: {acc_filt_freq*100:.2f}%")
    time.sleep(2)
    
    # Plot calibration data for each gesture
    fig, axs = plt.subplots(num_classes, 4, figsize=(20, 5*num_classes))
    for i in range(num_classes):
        # Concatenate samples along the time axis
        gesture_samples = raw_data_all[i]  # shape (num_samples, num_points, num_channels)
        concatenated = np.concatenate(gesture_samples, axis=0)
        # Average across channels to get a single time series
        avg_raw_time_series = np.mean(concatenated, axis=1)
        # Create time axis in ms
        t = np.arange(concatenated.shape[0]) / fs * 1000
        
        # Compute FFT of raw time series
        fft_raw = np.fft.rfft(avg_raw_time_series)
        freqs_raw = np.fft.rfftfreq(avg_raw_time_series.shape[0], d=1/fs)
        
        # Filter concatenated data and compute FFT
        concatenated_filt = filter_emg_data(concatenated, fs)
        avg_filt_time_series = np.mean(concatenated_filt, axis=1)
        fft_filt = np.fft.rfft(avg_filt_time_series)
        freqs_filt = np.fft.rfftfreq(avg_filt_time_series.shape[0], d=1/fs)
        
        axs[i, 0].plot(t, avg_raw_time_series, color="blue")
        axs[i, 0].set_title(f"{states[i]} Raw Time")
        axs[i, 2].plot(freqs_raw, np.abs(fft_raw), color="orange")
        axs[i, 2].set_title(f"{states[i]} Raw Freq")
        axs[i, 1].plot(t, avg_filt_time_series, color="blue")
        axs[i, 1].set_title(f"{states[i]} Filt Time")
        axs[i, 3].plot(freqs_filt, np.abs(fft_filt), color="orange")
        axs[i, 3].set_title(f"{states[i]} Filt Freq")
    plt.tight_layout()
    plt.show()
    
    # Inference loop
    print("\nStarting inference. Press Ctrl+C to exit")
    try:
        while True:
            # Collect a new sample from the board
            sample = np.zeros((num_points_per_sample, num_channels))
            for pt in range(num_points_per_sample):
                while board_shim.get_board_data_count() < 1:
                    time.sleep(0.001)
                s = board_shim.get_board_data(1)
                s = s[:num_channels]
                sample[pt] = s.squeeze()
            
            # Compute features for the new sample
            feat_raw_time = np.mean(np.abs(sample), axis=0)
            fft_raw = np.fft.rfft(sample, axis=0)
            feat_raw_freq = np.mean(np.abs(fft_raw), axis=0)
            sample_filt = filter_emg_data(sample, fs)
            feat_filt_time = np.mean(np.abs(sample_filt), axis=0)
            fft_filt = np.fft.rfft(sample_filt, axis=0)
            feat_filt_freq = np.mean(np.abs(fft_filt), axis=0)
            
            # Classify by comparing features to centroids
            label_raw_time = np.argmin(np.linalg.norm(centroids_raw_time - feat_raw_time, axis=1))
            label_raw_freq = np.argmin(np.linalg.norm(centroids_raw_freq - feat_raw_freq, axis=1))
            label_filt_time = np.argmin(np.linalg.norm(centroids_filt_time - feat_filt_time, axis=1))
            label_filt_freq = np.argmin(np.linalg.norm(centroids_filt_freq - feat_filt_freq, axis=1))
            
            # Map label indices to gesture names
            pred_raw_time = states[label_raw_time]
            pred_raw_freq = states[label_raw_freq]
            pred_filt_time = states[label_filt_time]
            pred_filt_freq = states[label_filt_freq]
            
            print("Inference Results -> Raw Time:", pred_raw_time,
                  " | Raw Freq:", pred_raw_freq,
                  " | Filt Time:", pred_filt_time,
                  " | Filt Freq:", pred_filt_freq)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Exiting inference loop")

if __name__ == "__main__":
    main()
