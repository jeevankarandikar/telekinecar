from util import calibrate, classify, filter_emg_data
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import numpy as np
import time

num_samples_per_class = 100
num_points_per_sample = 50
train_ratio = 0.8
test_ratio = 0.2
num_classes = 3
num_channels = 4
states = {0: "rest", 1 : "flex", 2: "spiderman"}

def main():

    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)
    
    board_shim.prepare_session()
    board_shim.start_stream(1)
    print("Connected to MindRove board and started stream.")
    
    train_size = int(num_samples_per_class*train_ratio)
    test_size = num_samples_per_class-train_size

    train_data = np.zeros((num_classes, train_size, num_channels))
    test_data = np.zeros((num_classes, test_size, num_channels))

    # Calibration Loop
    for i in range(num_classes):
        class_data = calibrate(state=states[i], num_samples=num_samples_per_class, num_points_per_sample=num_points_per_sample, num_channels=num_channels, board_shim=board_shim)
        
        rng = np.random.default_rng()
        indices = rng.choice(num_samples_per_class, size=num_samples_per_class, replace=False)

        for j in range(len(indices[:train_size])):
            train_data[i,j] = class_data[indices[j]]

        for j in range(len(indices[train_size:])):
            test_data[i,j] = class_data[indices[j]]

    # Calculate Centroids
    centroids = np.zeros((num_classes,num_channels))
    for i in range(num_classes):
        centroids[i] = np.mean(train_data[i], axis=0)

    # Test Set
    errors = 0
    total = 0
    for state in states:
        for sample in test_data[state]:
            pred = classify(sample, centroids)
            if pred != state:
                errors += 1
            total += 1

    print(f"Test Accuracy: {round((errors/total) * 100, 4)}%")
    time.sleep(2)

    # Inference
    while True:
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
        current_sample = np.mean(np.abs(points),axis=0)
        
        pred = classify(current_sample,centroids)
        print(pred)
    
if __name__ == "__main__":
    main()