'''
This script temporally aligns EMG signals from multiple files.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ============================== CONFIGURATION ==============================
input_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\all bandpass 20_200 and notch\contraction signals\yogurt 20 ml"
output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\all bandpass 20_200 and notch\averaged"  
n_channels = 22
peak_channel = 18  
# ===========================================================================

def load_emg_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.txt')]
    data_list = []
    for file in files:
        try:
            # Read all columns and handle empty values
            data = pd.read_csv(file, sep=',', header=0, na_values=['', 'nan', 'NaN'])
            
            # Check if we have enough columns
            if data.shape[1] < n_channels:
                print(f"File {file} has fewer than {n_channels} columns. Skipping.")
                continue
                
            # Take only the first n_channels columns
            data = data.iloc[:, :n_channels]
            
            # Check for empty channels (all NaN in a column)
            empty_channels = []
            for ch in range(n_channels):
                if data.iloc[:, ch].isna().all():
                    empty_channels.append(ch)
            
            if empty_channels:
                print(f"File {os.path.basename(file)} has empty channels: {[ch+1 for ch in empty_channels]}")
            
            data_list.append((file, data, empty_channels))
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return data_list

def center_data_around_peak(data, peak_channel):
    """
    Center the data around the maximum peak in the specified channel.
    Trims data points from the longer end to make both sides equal.
    
    args:   
        data (DataFrame): DataFrame with EMG data
        peak_channel (int): Channel index (0-based) to use for finding the peak

    returns:
        DataFrame with centered data
    """
    if peak_channel >= data.shape[1]:
        print(f"Peak channel {peak_channel} is out of range. Using channel 0.")
        peak_channel = 0
    
    # Get the signal from the specified channel
    signal = data.iloc[:, peak_channel].values
    
    # Find the maximum positive value (peak)
    peak_idx = np.argmax(signal)
    
    # Calculate distances from peak to start and end
    distance_to_start = peak_idx
    distance_to_end = len(signal) - peak_idx - 1
    
    # Determine the shorter side
    min_distance = min(distance_to_start, distance_to_end)
    
    # Calculate start and end indices for centered data
    start_idx = peak_idx - min_distance
    end_idx = peak_idx + min_distance + 1
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)
    
    # Extract centered data
    centered_data = data.iloc[start_idx:end_idx, :].reset_index(drop=True)
    
    print(f"Centered data around peak at index {peak_idx} in channel {peak_channel + 1}")
    print(f"Original length: {len(signal)}, Centered length: {len(centered_data)}")
    print(f"Trimmed {distance_to_start - min_distance} points from start, {distance_to_end - min_distance} points from end")
    
    return centered_data

def segment_data(data, n_segments=10):
    """
    Segment the data into n_segments equal parts.
    
    args:
        data (DataFrame): DataFrame with EMG data
        n_segments (int): Number of segments to divide the data into
        
    returns:
        list of DataFrames: List of segments
    """
    n_samples = len(data)
    seg_size = n_samples // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_size
        end = (i+1) * seg_size if i < n_segments - 1 else n_samples
        segments.append(data.iloc[start:end, :].reset_index(drop=True))
    return segments

def main():
    print("Select the folder containing your EMG .txt files...")
    folder = input_folder
    data_list = load_emg_files(folder)
    if not data_list:
        print("No valid .txt files found in the folder.")
        return
    
    # Use the peak channel from configuration (convert from 1-based to 0-based indexing)
    peak_channel_idx = peak_channel - 1  # Convert to 0-based index
    
    # Validate peak channel
    if peak_channel_idx < 0 or peak_channel_idx >= n_channels:
        print(f"Error: Peak channel {peak_channel} is out of range (1-{n_channels}). Using channel 1.")
        peak_channel_idx = 0
    
    print(f"Using channel {peak_channel} for peak detection...")
    
    n_segments = 10
    # For each file, center data around peak and then segment
    all_segments = []  # List of [ [segment1_df, ..., segment10_df], ... ]
    centered_data_list = []  # Store centered data for plotting
    
    for file, data, empty_channels in data_list:
        # Skip if the peak channel is empty in this file
        if peak_channel_idx in empty_channels:
            print(f"Skipping {os.path.basename(file)} - peak channel {peak_channel} is empty")
            continue
            
        # Center the data around the peak
        centered_data = center_data_around_peak(data, peak_channel_idx)
        centered_data_list.append((file, centered_data, empty_channels))
        
        # Segment the centered data
        segments = segment_data(centered_data, n_segments)
        all_segments.append((segments, empty_channels))
    
    if not all_segments:
        print("No files with valid peak channel data found.")
        return
    
    # Determine which channels have data in at least one file
    channels_with_data = set()
    for _, empty_channels in all_segments:
        for ch in range(n_channels):
            if ch not in empty_channels:
                channels_with_data.add(ch)
    
    channels_with_data = sorted(list(channels_with_data))
    n_channels_with_data = len(channels_with_data)
    
    print(f"Found {n_channels_with_data} channels with data out of {n_channels} total channels")
    
    # For each channel, collect all files' data for each segment
    fig, axes = plt.subplots(n_channels_with_data, 1, figsize=(1, 2*n_channels_with_data), sharex=True)
    if n_channels_with_data == 1:
        axes = [axes]
    
    # Create legend handles and labels
    legend_handles = []
    legend_labels = []
    
    for plot_idx, ch in enumerate(channels_with_data):
        ax = axes[plot_idx]
        for file_idx, (segments, empty_channels) in enumerate(all_segments):
            file, data, _ = centered_data_list[file_idx]
            
            # Skip if this channel is empty in this file
            if ch in empty_channels:
                continue
                
            y = data.iloc[:, ch].values
            x = np.linspace(0, 100, len(y))
            line, = ax.plot(x, y, alpha=0.7, linewidth=0.7)
            
            # Store legend info only once (for first channel)
            if plot_idx == 0:
                legend_handles.append(line)
                legend_labels.append(os.path.basename(file))
        
        ax.set_ylabel(f'{ch+1}', fontsize=8)
        ax.tick_params(axis='y', labelleft=False)  # Remove y-axis tick labels
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[-1].set_xlabel('Contraction (%)', fontsize=10)
    
    # Add legend to the figure
    fig.legend(legend_handles, legend_labels, loc='center right', bbox_to_anchor=(0.9, 0.5), fontsize=8)
    
    plt.subplots_adjust(left=0.4, right=0.6)
    plt.show()

    # Ask user if they want to save the averaged data
    save_avg = input("Do you want to save the averaged data across all files? (y/n): ").strip().lower()
    if save_avg == 'y':
        if not output_folder:
            print("No output folder specified. Please set output_folder at the top of the script.")
            return
        
        # First pass: determine the global minimum length across all channels
        global_min_len = float('inf')
        channel_lengths = []
        
        for ch in range(n_channels):
            valid_files_data = []
            valid_files_lengths = []
            
            for file_idx, (_, empty_channels) in enumerate(all_segments):
                if ch in empty_channels:
                    continue
                    
                file, data, _ = centered_data_list[file_idx]
                y = data.iloc[:, ch].values
                valid_files_data.append(y)
                valid_files_lengths.append(len(y))
            
            if valid_files_data:
                channel_min_len = min(valid_files_lengths)
                global_min_len = min(global_min_len, channel_min_len)
                channel_lengths.append(channel_min_len)
            else:
                channel_lengths.append(0)
        
        if global_min_len == float('inf'):
            print("No valid data found in any channel.")
            return
            
        print(f"Global minimum length across all channels: {global_min_len}")
        
        # Second pass: collect all valid data for each channel and align them properly
        all_channels_data = []  # List to store averaged data for each channel
        
        for ch in range(n_channels):
            print(f"\nProcessing channel {ch+1}...")
            
            # Collect all valid data for this channel
            valid_files_data = []
            valid_files_lengths = []
            
            for file_idx, (_, empty_channels) in enumerate(all_segments):
                # Skip if this channel is empty in this file
                if ch in empty_channels:
                    continue
                    
                file, data, _ = centered_data_list[file_idx]
                y = data.iloc[:, ch].values
                
                valid_files_data.append(y)
                valid_files_lengths.append(len(y))
            
            if not valid_files_data:
                print(f"Channel {ch+1}: no valid data found")
                # Add zeros for empty channels to maintain column structure
                all_channels_data.append(np.zeros(global_min_len))
                continue
            
            print(f"Channel {ch+1}: found {len(valid_files_data)} files with valid data")
            print(f"File lengths: {valid_files_lengths}")
            
            # Truncate all signals to the global minimum length using percentage-based alignment
            aligned_data = []
            for i, y in enumerate(valid_files_data):
                # Calculate how many points to remove from each end to reach global_min_len
                excess_points = len(y) - global_min_len
                points_to_remove_start = excess_points // 2
                points_to_remove_end = excess_points - points_to_remove_start
                
                # Truncate the signal by removing points from both ends
                start_idx = points_to_remove_start
                end_idx = len(y) - points_to_remove_end
                y_truncated = y[start_idx:end_idx]
                
                aligned_data.append(y_truncated)
            
            # Average the aligned data
            avg_signal = np.mean(aligned_data, axis=0)
            all_channels_data.append(avg_signal)
        
        # Save all channels in one file
        os.makedirs(output_folder, exist_ok=True)
        filename = f'{os.path.basename(input_folder)} averaged.txt'
        filepath = os.path.join(output_folder, filename)
        
        # Convert list of arrays to a 2D array (samples x channels)
        all_channels_array = np.column_stack(all_channels_data)
        np.savetxt(filepath, all_channels_array, delimiter=',', fmt='%.6f')
        print(f"\nAll channel averaged data saved to: {filename}")
        print(f"File contains {all_channels_array.shape[0]} samples and {all_channels_array.shape[1]} channels")

if __name__ == "__main__":
    main()
