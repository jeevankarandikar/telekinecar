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
            points[num_points] = np.abs(sample[:num_channels].squeeze())
            num_points += 1
        print(sample_num + "/" + num_samples)
        mean_of_points = np.mean(points, axis=0)
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