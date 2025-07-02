'''
This script temporally aligns EMG signals from multiple files.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ======= CONFIGURATION =======
input_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\05_08_25\detected signals\dry swallow"  # <-- Set your input folder path here, or leave blank to select interactively
output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\05_08_25\averaged"  # <-- Set your output folder path here
n_channels = 8
# =============================

def load_emg_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.txt')]
    data_list = []
    for file in files:
        try:
            data = pd.read_csv(file, sep=',', header=0, usecols=range(8))
            data_list.append((file, data))
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
    for file, data in data_list:
        if data.shape[1] != n_channels:
            print(f"File {file} does not have 22 channels. Skipping.")
            continue
        segments = segment_data(data, n_segments)
        all_segments.append(segments)
    if not all_segments:
        print("No files with correct channel count.")
        return
    # For each channel, collect all files' data for each segment
    fig, axes = plt.subplots(n_channels, 1, figsize=(1, 2*n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    # Create legend handles and labels
    legend_handles = []
    legend_labels = []
    
    for ch in range(n_channels):
        ax = axes[ch]
        for file_idx, segments in enumerate(all_segments):
            file, data = data_list[file_idx]
            y = data.iloc[:, ch].values
            x = np.linspace(0, 100, len(y))
            line, = ax.plot(x, y, alpha=0.7, linewidth=0.7)
            
            # Store legend info only once (for first channel)
            if ch == 0:
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
        min_len = min([data.shape[0] for _, data in data_list])
        resampled_arrays = []
        for _, data in data_list:
            arr = np.zeros((min_len, n_channels))
            for ch in range(n_channels):
                y = data.iloc[:, ch].values
                x = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, min_len)
                arr[:, ch] = np.interp(x_new, x, y)
            resampled_arrays.append(arr)
        avg_data = np.mean(resampled_arrays, axis=0)
        # Save to output_folder
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f'{os.path.basename(input_folder)} averaged.txt')
        np.savetxt(out_path, avg_data, delimiter=',', fmt='%.6f')
        print(f"Averaged data saved to: {out_path}")

if __name__ == "__main__":
    main()
