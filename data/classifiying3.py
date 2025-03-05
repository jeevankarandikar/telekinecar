import pickle
import time
import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from scipy.signal import butter, filtfilt, iirnotch

def filter_epoch(epoch, sampling_rate=500):
    """
    Applies a 4th‑order Butterworth bandpass filter (4.5–100 Hz)
    followed by a notch filter at 60 Hz to the given epoch.
    
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

#########################
# Training the classifier
#########################
def train_classifier(data_file="emg_gestures.pkl"):
    """
    Loads recorded data and trains a gesture classifier using only three gestures:
    clenched_fist, relaxed, and open_palm. Each epoch is filtered and then flattened into
    a feature vector. Performs 5-fold cross-validation on the training set and prints the CV scores.
    
    Returns:
        clf: Trained classifier.
        scaler: Fitted StandardScaler.
        le: Fitted LabelEncoder.
    """
    # Load recorded data and labels.
    with open(data_file, "rb") as f:
        recorded_data, recorded_labels = pickle.load(f)
    
    recorded_data = np.array(recorded_data)       # shape: (n_epochs, window_length, channels)
    recorded_labels = np.array(recorded_labels)     # Expected to be strings
    
    # Use only the allowed gestures.
    allowed_gestures = ["clenched_fist", "relaxed", "open_palm"]
    mask = np.isin(recorded_labels, allowed_gestures)
    recorded_data = recorded_data[mask]
    recorded_labels = recorded_labels[mask]
    
    # Apply filtering to each epoch and flatten into feature vectors.
    X = np.array([filter_epoch(epoch, sampling_rate=500).flatten() for epoch in recorded_data])
    
    # Encode gesture labels as numbers.
    le = LabelEncoder()
    y = le.fit_transform(recorded_labels)
    
    # Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the classifier.
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=5000, random_state=42)
    
    # Perform 5-fold cross-validation on the training set.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score: {:.2f}%".format(np.mean(cv_scores)*100))
    
    # Train the classifier on the full training set.
    clf.fit(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Save the trained model, scaler, and label encoder.
    with open("gesture_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("gesture_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("gesture_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    return clf, scaler, le

#########################
# Real-time inference
#########################
def real_time_inference(clf, scaler, le, sampling_rate=500, model_input_len=100):
    """
    Starts real-time inference using the MindRove board.
    Acquires raw data in windows (epochs), applies filtering,
    flattens the filtered epoch, scales it, and predicts the gesture.
    
    Args:
        clf: Trained classifier.
        scaler: Fitted StandardScaler.
        le: Fitted LabelEncoder.
        sampling_rate (int): Sampling rate in Hz.
        model_input_len (int): Number of samples in an epoch.
    """
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)
    
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        print("Real-time inference started. Press Ctrl+C to stop.")
        
        while True:
            # Wait until the board has enough samples.
            required_samples = model_input_len
            while board_shim.get_board_data_count() < required_samples:
                time.sleep(0.005)
            
            # Retrieve a window of data.
            raw_data = board_shim.get_board_data(required_samples)
            # For a 4-channel device, assume the first 4 channels are EMG.
            emg_data = np.array(raw_data[:4]).T  # shape: (model_input_len, channels)
            
            # Filter the epoch.
            filtered_emg = filter_epoch(emg_data, sampling_rate)
            
            # Flatten the filtered epoch.
            feature_vector = filtered_emg.flatten().reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Predict the gesture.
            pred_numeric = clf.predict(feature_vector_scaled)[0]
            pred_label = le.inverse_transform([pred_numeric])[0]
            print(f"Predicted Gesture: {pred_label}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Real-time inference stopped by user.")
    except Exception as e:
        print("Error during real-time inference:", e)
    finally:
        board_shim.stop_stream()
        board_shim.release_session()

#########################
# Main script
#########################
if __name__ == "__main__":
    choice = input("Enter 'train' to train the model or 'infer' to run real-time inference: ").strip().lower()
    
    if choice == "train":
        clf, scaler, le = train_classifier()
        print("Training complete. Model saved as 'gesture_classifier.pkl'.")
    elif choice == "infer":
        with open("gesture_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("gesture_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("gesture_label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        real_time_inference(clf, scaler, le, sampling_rate=500, model_input_len=100)
    else:
        print("Invalid choice. Please enter 'train' or 'infer'.")