'''
This script plots data from an input file with the same format as imu_temporal_aligner.py.
The sampling rate is 500 Hz. It also plots the envelope (rectified + RMS smoothed) of the first 16 channels.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================== CONFIGURATION ==============================
input_file = r"C:\Users\chloe\OneDrive\Desktop\swallow EMG\data\participants\1\extracted signals\water 20\12.txt"
sampling_rate = 500  # Hz
rms_window_sec = 0.05  # RMS smoothing window in seconds
# ===========================================================================

def load_data(file_path):
    """Load data from a comma-delimited text file, assuming no header."""
    try:
        data = pd.read_csv(file_path, delimiter=',', header=None)
        return data.values
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def create_envelope(data, sampling_rate, rms_window_sec):
    """Create RMS envelope for the first 16 channels."""
    # Extract first 16 channels
    first_16_channels = data[:, :16]
    
    # Rectify the data (take absolute value)
    rectified_data = np.abs(first_16_channels)
    
    # Calculate RMS envelope
    window_samples = int(rms_window_sec * sampling_rate)
    if window_samples < 1:
        window_samples = 1
    
    # Use rolling window to calculate Root Mean Square for each channel
    rms_data = np.zeros_like(first_16_channels)
    
    for ch in range(16):
        # Calculate rolling RMS for each channel
        for i in range(len(data)):
            start_idx = max(0, i - window_samples // 2)
            end_idx = min(len(data), i + window_samples // 2 + 1)
            window_data = rectified_data[start_idx:end_idx, ch]
            rms_data[i, ch] = np.sqrt(np.mean(window_data ** 2))
    
    return rms_data

def plot_data(data, file_path, sampling_rate, rms_window_sec):
    """Plot data with envelopes for the first 16 channels."""
    n_channels = data.shape[1]
    n_samples = data.shape[0]
    
    # Create time axis in seconds
    time = np.arange(n_samples) / sampling_rate
    
    # Create envelope for first 16 channels
    envelope_data = create_envelope(data, sampling_rate, rms_window_sec)
    
    # Create subplots
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 1.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    # Plot each channel
    for ch in range(n_channels):
        ax = axes[ch]
        
        if ch < 16:
            # First 16 channels: plot original signal and envelope
            ax.plot(time, data[:, ch], linewidth=0.5, color='#003fffff', label='Original' if ch == 0 else "")
            ax.plot(time, envelope_data[:, ch], linewidth=0.7, color='red', label='Envelope' if ch == 0 else "")
        else:
            # Last 6 channels: plot only original signal
            ax.plot(time, data[:, ch], linewidth=0.5, color='#003fffff')
        
        ax.set_ylabel(f'{ch+1}', fontsize=8)
        ax.tick_params(axis='y', labelleft=False, which='both', length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.margins(y=0.1)
    
    # Set x-axis label and title
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    
    # Add single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    # Adjust layout
    plt.subplots_adjust(left=0.42, right=0.58, top=0.88, bottom=0.1)
    plt.show()

def main():
    # Use configured input file
    file_path = input_file
    
    if not file_path:
        print("No input file specified in configuration. Please set input_file path.")
        return
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please check the input_file path in the configuration section.")
        return
    
    try:
        # Load data
        print(f"Loading data from: {file_path}")
        data = load_data(file_path)
        
        if data is None or data.size == 0:
            print("No data found in file.")
            return
        
        print(f"Data loaded successfully. Shape: {data.shape}")
        
        # Plot data
        plot_data(data, file_path, sampling_rate, rms_window_sec)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return

if __name__ == "__main__":
    main()
