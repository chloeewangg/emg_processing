'''
This script temporally aligns EMG signals from multiple files.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ======= CONFIGURATION =======
input_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\all bandpass 20_200 and notch\contraction signals\apple 5 ml"
output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\all bandpass 20_200 and notch\averaged"  
n_channels = 22
# =============================

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

def segment_data(data, n_segments=10):
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
    
    n_segments = 10
    # For each file, segment data
    all_segments = []  # List of [ [segment1_df, ..., segment10_df], ... ]
    for file, data, empty_channels in data_list:
        segments = segment_data(data, n_segments)
        all_segments.append((segments, empty_channels))
    
    if not all_segments:
        print("No files with correct channel count.")
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
            file, data, _ = data_list[file_idx]
            
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
        
        # Find the sparsest file (fewest samples)
        min_len = min([data.shape[0] for _, data, _ in data_list])
        
        # Initialize output array with NaN
        avg_data = np.full((min_len, n_channels), np.nan)
        
        # For each channel, average only files that have valid data
        for ch in range(n_channels):
            valid_files_data = []
            
            for file_idx, (_, empty_channels) in enumerate(all_segments):
                # Skip if this channel is empty in this file
                if ch in empty_channels:
                    continue
                    
                file, data, _ = data_list[file_idx]
                y = data.iloc[:, ch].values
                
                # Resample to min_len
                x = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, min_len)
                y_resampled = np.interp(x_new, x, y)
                valid_files_data.append(y_resampled)
            
            # Average the valid data for this channel
            if valid_files_data:
                avg_data[:, ch] = np.mean(valid_files_data, axis=0)
                print(f"Channel {ch+1}: averaged across {len(valid_files_data)} files")
            else:
                print(f"Channel {ch+1}: no valid data found")
        
        # Save to output_folder
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f'{os.path.basename(input_folder)} averaged.txt')
        np.savetxt(out_path, avg_data, delimiter=',', fmt='%.6f')
        print(f"Averaged data saved to: {out_path}")

if __name__ == "__main__":
    main()
