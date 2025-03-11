from util import calibrate, filter_emg_data
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

num_samples_per_class = 100
num_points_per_sample = 50
train_ratio = 0.8
test_ratio = 0.2
num_classes = 3
num_channels = 4
states = {0: "rest", 1: "flex", 2: "spiderman"}

def main():
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)
    
    board_shim.prepare_session()
    board_shim.start_stream(1)
    print("Connected to MindRove board and started stream.")
    
    train_size = int(num_samples_per_class * train_ratio)
    test_size = num_samples_per_class - train_size

    train_data = np.zeros((num_classes, train_size, num_channels))
    test_data = np.zeros((num_classes, test_size, num_channels))
    train_labels = []
    test_labels = []

    # Calibration Loop
    for i in range(num_classes):
        class_data = calibrate(state=states[i], num_samples=num_samples_per_class, 
                               num_points_per_sample=num_points_per_sample, 
                               num_channels=num_channels, board_shim=board_shim)
        
        rng = np.random.default_rng()
        indices = rng.choice(num_samples_per_class, size=num_samples_per_class, replace=False)

        for j in range(train_size):
            train_data[i, j] = class_data[indices[j]]
            train_labels.append(i)

        for j in range(test_size):
            test_data[i, j] = class_data[indices[train_size + j]]
            test_labels.append(i)

    # We dont use centroids since for these models we are trying to plot a decision boundry. dont need mean like RNN
    # Reshape Data for Training
    X_train = train_data.reshape(-1, num_channels)
    X_test = test_data.reshape(-1, num_channels)
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # Initialize classifiers
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')

    # Train classifiers
    rf_classifier.fit(X_train, y_train)
    svm_classifier.fit(X_train, y_train)

    # Predictions
    rf_pred = rf_classifier.predict(X_test)
    svm_pred = svm_classifier.predict(X_test)

    # Print Accuracy
    print(f"Random Forest Test Accuracy: {round(accuracy_score(y_test, rf_pred) * 100, 2)}%")
    print(f"SVM Test Accuracy: {round(accuracy_score(y_test, svm_pred) * 100, 2)}%")
    
    time.sleep(2)

    # Real-Time Inference Loop
    model = rf_classifier  # Choose which model to use for inference

    while True:
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
        current_sample = np.mean(np.abs(points), axis=0).reshape(1, -1)
        
        pred = model.predict(current_sample)[0]
        print(f"Predicted State: {states[pred]}")

if __name__ == "__main__":
    main()