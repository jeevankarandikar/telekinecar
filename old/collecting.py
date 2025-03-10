import pickle
import time
import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

def collect_emg_data(
    recording_time_sec=8,
    sampling_rate=500,
    model_input_len=100,
    overlap_frac=10,
    output_file="emg_gestures.pkl"
):
    # Define gesture labels.
    gesture_names = {
        0: "relaxed",
        1: "clenched_fist",
        2: "open_palm",
        3: "upwards_rotation",
        4: "downwards_rotation",
        5: "left_rotation"
    }
    
    recorded_data = []
    recorded_labels = []
    
    # Initialize MindRove board (assumed 4-channel EMG device).
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)
    
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        print("Connected to MindRove board and started stream.")
        
        # Warm-up phase: wait up to warmup_time seconds for data to arrive.
        warmup_time = 10  # seconds
        print(f"Warming up for {warmup_time} seconds...")
        warmup_start = time.time()
        while time.time() - warmup_start < warmup_time:
            if board_shim.get_board_data_count() > 0:
                # Flush some initial data.
                board_shim.get_board_data(sampling_rate)
                break
        if board_shim.get_board_data_count() == 0:
            print("No data stream received from the board. Please check the connection.")
            return
        
        # Loop through each gesture.
        for gesture_id in range(6):
            gesture_name = gesture_names[gesture_id]
            input(f"\nPrepare for gesture '{gesture_name}' (using your right hand). Press Enter to record for {recording_time_sec} seconds...")
            
            # Flush any residual data.
            board_shim.get_board_data()
            
            # Wait briefly until some data is available (up to 5 seconds).
            wait_start = time.time()
            while board_shim.get_board_data_count() < sampling_rate:
                if time.time() - wait_start > 5:
                    print(f"No data received for gesture '{gesture_name}'. Skipping this gesture.")
                    break
            else:
                # Ensure the buffer holds enough samples for the recording period.
                required_samples = recording_time_sec * sampling_rate
                while board_shim.get_board_data_count() < required_samples:
                    time.sleep(0.01)
                
                # Retrieve raw data.
                raw_data = board_shim.get_board_data(required_samples)
                # For a 4-channel device, assume the first 4 channels are EMG.
                # raw_data is organized as a list where each element corresponds to a channel.
                emg_data = np.array(raw_data[:4]).T  # Shape: (samples, channels)
                # No preprocessing is applied; we use the raw data directly.
                
                # Segment the continuous data into overlapping windows.
                gesture_samples = []
                for i in range(0, len(emg_data) - model_input_len, overlap_frac):
                    sample = emg_data[i:i + model_input_len]
                    gesture_samples.append(sample)
                
                recorded_data.extend(gesture_samples)
                recorded_labels.extend([gesture_name] * len(gesture_samples))
                print(f"Recorded {len(gesture_samples)} samples for gesture '{gesture_name}'.")
        
        # Save the collected data and labels.
        with open(output_file, "wb") as f:
            pickle.dump((recorded_data, recorded_labels), f)
        print(f"\nData collection complete. Saved to '{output_file}'.")
    
    except Exception as e:
        print("Error during data collection:", e)
    
    finally:
        board_shim.stop_stream()
        board_shim.release_session()
        print("Disconnected from MindRove board.")

if __name__ == "__main__":
    collect_emg_data()