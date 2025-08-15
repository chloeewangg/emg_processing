import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
import glob

# ============================ CONFIGURATION ==============================

INPUT_FILE = r"C:\Users\chloe\Documents\FreeBCI_GUI\Recordings\Stream_2025_07_30_112301\1.txt"

# Filter parameters
SAMPLING_RATE = 500  
LOW_CUTOFF = 20  
HIGH_CUTOFF = 200  
TRANSITION_WIDTH_PERCENT = 20  

# Data parameters
NUM_HEADER_ROWS = 6
NUM_PROCESSED_CHANNELS = 16  
TOTAL_CHANNELS = 22  

# Plotting parameters
EXCLUDE_SECONDS = 1  

CHANNELS_TO_PLOT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] 

# ==========================================================================

def create_bandpass_filter(low_cutoff, high_cutoff, sampling_rate, transition_width_percent=20):
    """
    Create a bandpass filter with specified transition width.
    
    args:
        low_cutoff (float): lower cutoff frequency in Hz
        high_cutoff (float): upper cutoff frequency in Hz
        sampling_rate (float): sampling rate in Hz
        transition_width_percent (float): transition width as percentage of cutoff frequencies
    
    returns:
        b, a (tuple): filter coefficients
    """
    # Calculate transition widths
    low_transition = low_cutoff * transition_width_percent / 100
    high_transition = high_cutoff * transition_width_percent / 100
    
    # Calculate normalized frequencies
    low_norm = (low_cutoff - low_transition) / (sampling_rate / 2)
    high_norm = (high_cutoff + high_transition) / (sampling_rate / 2)
    
    # Ensure frequencies are within valid range
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(0.001, min(high_norm, 0.999))
    
    # Create bandpass filter
    b, a = butter(4, [low_norm, high_norm], btype='band')
    return b, a

def create_notch_filter(notch_freq, sampling_rate, quality_factor=30):
    """
    Create a notch filter to remove power line interference.
    
    args:
        notch_freq (float): frequency to notch out in Hz
        sampling_rate (float): sampling rate in Hz
        quality_factor (float): quality factor of the notch filter
    
    returns:
        b, a (tuple): filter coefficients
    """
    b, a = iirnotch(notch_freq, quality_factor, sampling_rate)
    return b, a

def trim_leading_zeros(data):
    """
    Trim leading zeros from the data.
    
    args:
        data (numpy.ndarray): input data array
    
    returns:
        numpy.ndarray: data with leading zeros removed
    """
    # Find the first non-zero row
    non_zero_rows = np.any(data != 0, axis=1)
    first_non_zero = np.where(non_zero_rows)[0]
    
    if len(first_non_zero) > 0:
        start_idx = first_non_zero[0]
        return data[start_idx:]
    else:
        return data

def apply_filters(data, sampling_rate, low_cutoff, high_cutoff, transition_width_percent=20):
    """
    Apply bandpass and notch filters to the data.
    
    args:
        data (numpy.ndarray): input data array (samples x channels)
        sampling_rate (float): sampling rate in Hz
        low_cutoff (float): lower cutoff frequency for bandpass filter
        high_cutoff (float): upper cutoff frequency for bandpass filter
        transition_width_percent (float): transition width as percentage of cutoff frequencies
    
    returns:
        numpy.ndarray: filtered data
    """
    # Create filters
    bandpass_b, bandpass_a = create_bandpass_filter(low_cutoff, high_cutoff, sampling_rate, transition_width_percent)
    notch_b_60, notch_a_60 = create_notch_filter(60, sampling_rate)
    notch_b_120, notch_a_120 = create_notch_filter(120, sampling_rate)
    notch_b_180, notch_a_180 = create_notch_filter(180, sampling_rate)
    
    # Apply bandpass filter
    filtered_data = filtfilt(bandpass_b, bandpass_a, data, axis=0)
    
    # Apply notch filter at 60 Hz
    filtered_data = filtfilt(notch_b_60, notch_a_60, filtered_data, axis=0)
    # Apply notch filter at 120 Hz
    filtered_data = filtfilt(notch_b_120, notch_a_120, filtered_data, axis=0)
    # Apply notch filter at 180 Hz
    filtered_data = filtfilt(notch_b_180, notch_a_180, filtered_data, axis=0)
    
    return filtered_data

def plot_preprocessed_data(input_file):
    """
    Process a single EMG file and plot all 22 channels.
    
    args:
        input_file (str): path to input file
    returns: 
        None
    """
    try:
        # Read the data, skipping header rows
        print(f"Processing: {os.path.basename(input_file)}")
        
        # Read data with pandas to handle header rows
        data = pd.read_csv(input_file, header=None, skiprows=NUM_HEADER_ROWS)
        
        # Convert to numpy array
        data_array = data.values
        
        # Check if we have enough channels
        if data_array.shape[1] < TOTAL_CHANNELS:
            print(f"Warning: File has {data_array.shape[1]} channels, expected {TOTAL_CHANNELS}")
            return
        
        # Trim leading zeros
        data_array = trim_leading_zeros(data_array)
        
        # Separate processed and unprocessed channels
        processed_channels = data_array[:, :NUM_PROCESSED_CHANNELS]
        unprocessed_channels = data_array[:, NUM_PROCESSED_CHANNELS:TOTAL_CHANNELS]
        
        # Apply filters to processed channels only
        filtered_channels = apply_filters(
            processed_channels, 
            SAMPLING_RATE, 
            LOW_CUTOFF, 
            HIGH_CUTOFF, 
            TRANSITION_WIDTH_PERCENT
        )
        
        # Combine processed and unprocessed channels
        output_data = np.column_stack([filtered_channels, unprocessed_channels])
        
        # Trim leading zeros from the final data
        output_data = trim_leading_zeros(output_data)
        
        # Calculate samples to exclude from start and end
        exclude_samples = int(EXCLUDE_SECONDS * SAMPLING_RATE)
        
        # Trim the data to exclude first and last 0.4 seconds
        if output_data.shape[0] > 2 * exclude_samples:
            output_data = output_data[exclude_samples:-exclude_samples]
        else:
            print(f"Warning: Data too short to exclude {EXCLUDE_SECONDS} seconds from each end")
        
        # Create time axis
        time_axis = np.arange(output_data.shape[0]) / SAMPLING_RATE
        
        # Determine which channels to plot
        if CHANNELS_TO_PLOT is None:
            channels_to_plot = list(range(1, TOTAL_CHANNELS + 1))  # Plot all channels
        else:
            channels_to_plot = CHANNELS_TO_PLOT  # Use specified channels
        
        num_channels_to_plot = len(channels_to_plot)
        
        # Plot selected channels
        fig, axes = plt.subplots(num_channels_to_plot, 1, figsize=(12, 6 * num_channels_to_plot), sharex=True)
        
        # Handle case where only one channel is selected
        if num_channels_to_plot == 1:
            axes = [axes]
        
        for i, ch in enumerate(channels_to_plot):
            ch_idx = ch - 1  # Convert to 0-based indexing
            ax = axes[i]
            ax.plot(time_axis, output_data[:, ch_idx], linewidth=0.7, color='black')
            
            # Get min and max values for this channel
            ch_data = output_data[:, ch_idx]
            min_val = np.min(ch_data)
            max_val = np.max(ch_data)
            
            # Set y-axis label to just channel number
            ax.set_ylabel(f'{ch}', fontsize=8)
            
            # Set y-axis limits and add min/max/zero labels
            ax.set_ylim(min_val, max_val)
            
            # Create tick positions every 10 mV within the range
            tick_start = np.floor(min_val / 10) * 10
            tick_end = np.ceil(max_val / 10) * 10
            tick_positions = np.arange(tick_start, tick_end + 1, 10)
            # Ensure min, max, and 0 are included
            if min_val not in tick_positions:
                tick_positions = np.append(tick_positions, min_val)
            if max_val not in tick_positions:
                tick_positions = np.append(tick_positions, max_val)
            if min_val < 0 < max_val and 0 not in tick_positions:
                tick_positions = np.append(tick_positions, 0)
            tick_positions = np.sort(tick_positions)
            tick_labels = [f'{tp:.1f}' for tp in tick_positions]
            
            # ax.set_yticks(tick_positions)
            # ax.set_yticklabels(tick_labels, fontsize=6, rotation=0)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.margins(y=0.2)
        
        # Set x-axis label on the bottom subplot
        axes[-1].set_xlabel('Time (seconds)', fontsize=10)
        
        # Add title
        plt.suptitle(f'{os.path.basename(input_file)}', fontsize=12)
        
        # Add more space between subplots
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        
        print(f"Plot displayed for: {os.path.basename(input_file)}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def main():
    # Use the configured input file or prompt user if not set
    input_file = INPUT_FILE
    
    if not input_file or not os.path.exists(input_file):
        # Fallback to file dialog if file not found or not specified
        from tkinter import Tk, filedialog
        
        Tk().withdraw()
        input_file = filedialog.askopenfilename(
            title="Select EMG file to process",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not input_file:
            print("No file selected.")
            return
    
    print(f"Processing file: {input_file}")
    print(f"Filter parameters: Bandpass {LOW_CUTOFF}-{HIGH_CUTOFF} Hz")
    print(f"Notch filters: 60, 120, 180 Hz")
    print(f"Excluding first and last {EXCLUDE_SECONDS} seconds")
    print("-" * 80)
    
    # Process and plot the file
    plot_preprocessed_data(input_file)

if __name__ == "__main__":
    main()
