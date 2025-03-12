from utils import calibrate, classify, filter_emg_data, plot_data_per_class
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters
num_samples_per_class = 100
num_points_per_sample = 30  # Must be >= 27 for the IIR filter padding.
train_ratio = 0.8
num_classes = 3
num_channels = 4
states = {0: "rest", 1: "flex", 2: "spiderman"}
fs = 500

def main():
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)
    
    board_shim.prepare_session()
    board_shim.start_stream(1)
    print("Connected to MindRove board and started stream.")
    
    train_size = int(num_samples_per_class * train_ratio)
    test_size = num_samples_per_class - train_size

    # Calibration: collect full raw time-series samples for each gesture
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
        input(f"Calibration for gesture '{states[i]}' complete. Press Enter to continue to the next gesture...")

    # Split into training and testing sets
    train_data_raw = np.zeros((num_classes, train_size, num_points_per_sample, num_channels))
    test_data_raw  = np.zeros((num_classes, test_size, num_points_per_sample, num_channels))
    for i in range(num_classes):
        rng = np.random.default_rng()
        indices = rng.choice(num_samples_per_class, size=num_samples_per_class, replace=False)
        train_data_raw[i] = raw_data_all[i][indices[:train_size]]
        test_data_raw[i]  = raw_data_all[i][indices[train_size:]]
    
    # Build four datasets by computing features from each sample
    # For each sample (shape: [num_points, num_channels]), extract a feature vector by averaging absolute values over time.
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
            sample = train_data_raw[i, j]  # shape: (num_points_per_sample, num_channels)
            # Raw time
            train_feats_raw_time[i, j] = np.mean(np.abs(sample), axis=0)
            # Raw frequency
            fft_raw = np.fft.rfft(sample, axis=0)
            train_feats_raw_freq[i, j] = np.mean(np.abs(fft_raw), axis=0)
            # Filtered time
            sample_filt = filter_emg_data(sample, fs)
            train_feats_filt_time[i, j] = np.mean(np.abs(sample_filt), axis=0)
            # Filtered frequency
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
    
    # Compute centroids for datasets
    centroids_raw_time = np.zeros((num_classes, num_channels))
    centroids_raw_freq = np.zeros((num_classes, num_channels))
    centroids_filt_time = np.zeros((num_classes, num_channels))
    centroids_filt_freq = np.zeros((num_classes, num_channels))
    for i in range(num_classes):
        centroids_raw_time[i] = np.mean(train_feats_raw_time[i], axis=0)
        centroids_raw_freq[i] = np.mean(train_feats_raw_freq[i], axis=0)
        centroids_filt_time[i] = np.mean(train_feats_filt_time[i], axis=0)
        centroids_filt_freq[i] = np.mean(train_feats_filt_freq[i], axis=0)
    
    # Classification accuracy for datasets
    def evaluate_accuracy(test_feats, centroids):
        correct = 0
        total = 0
        for i in range(num_classes):
            for feat in test_feats[i]:
                dists = np.linalg.norm(centroids - feat, axis=1)
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
    #print(f"Raw Time Series: {acc_raw_time*100:.2f}%")
    #print(f"Raw Frequency Series: {acc_raw_freq*100:.2f}%")
    print(f"Filtered Time Series: {acc_filt_time*100:.2f}%")
    print(f"Filtered Frequency Series: {acc_filt_freq*100:.2f}%")
    time.sleep(2)
    
    # Generate 4x3 plots
    fig, axs = plt.subplots(num_classes, 4, figsize=(20, 5*num_classes))
    for i in range(num_classes):
        avg_sample = np.mean(np.concatenate((train_data_raw[i], test_data_raw[i]), axis=0), axis=0)
        # Averaging across channels
        avg_raw_time = np.mean(avg_sample, axis=1)
        sample_filt = filter_emg_data(avg_sample, fs)
        avg_filt_time = np.mean(sample_filt, axis=1)
        fft_raw, freqs_raw = compute_fft(avg_sample, fs)
        avg_raw_freq = np.mean(np.abs(fft_raw), axis=1)
        fft_filt, freqs_filt = compute_fft(sample_filt, fs)
        avg_filt_freq = np.mean(np.abs(fft_filt), axis=1)
        
        t = np.arange(num_points_per_sample) / fs * 1000  # time axis in ms
        
        axs[i, 0].plot(t, avg_raw_time)
        axs[i, 0].set_title(f"{states[i]} Raw Time")
        axs[i, 1].plot(freqs_raw, avg_raw_freq)
        axs[i, 1].set_title(f"{states[i]} Raw Freq")
        axs[i, 2].plot(t, avg_filt_time)
        axs[i, 2].set_title(f"{states[i]} Filt Time")
        axs[i, 3].plot(freqs_filt, avg_filt_freq)
        axs[i, 3].set_title(f"{states[i]} Filt Freq")
    
    plt.tight_layout()
    plt.show()

     # --- Inference Loop ---
    print("\nStarting inference. Press Ctrl+C to exit.")
    try:
        while True:
            # Collect one new sample from the board.
            sample = np.zeros((num_points_per_sample, num_channels))
            for pt in range(num_points_per_sample):
                while board_shim.get_board_data_count() < 1:
                    time.sleep(0.001)
                s = board_shim.get_board_data(1)
                s = s[:num_channels]
                sample[pt] = s.squeeze()
            
            # Compute features for each representation.
            feat_raw_time = np.mean(np.abs(sample), axis=0)
            fft_raw = np.fft.rfft(sample, axis=0)
            feat_raw_freq = np.mean(np.abs(fft_raw), axis=0)
            sample_filt = filter_emg_data(sample, fs)
            feat_filt_time = np.mean(np.abs(sample_filt), axis=0)
            fft_filt = np.fft.rfft(sample_filt, axis=0)
            feat_filt_freq = np.mean(np.abs(fft_filt), axis=0)
            
            # Classify for each dataset by computing Euclidean distance to the centroids.
            label_raw_time = np.argmin(np.linalg.norm(centroids_raw_time - feat_raw_time, axis=1))
            label_raw_freq = np.argmin(np.linalg.norm(centroids_raw_freq - feat_raw_freq, axis=1))
            label_filt_time = np.argmin(np.linalg.norm(centroids_filt_time - feat_filt_time, axis=1))
            label_filt_freq = np.argmin(np.linalg.norm(centroids_filt_freq - feat_filt_freq, axis=1))
            
            # Map label indices to gesture names.
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
        print("Exiting inference loop.")

if __name__ == "__main__":
    main()
