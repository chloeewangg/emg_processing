'''
This script temporally aligns IMU signals from multiple files.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ============================== CONFIGURATION ==============================
input_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\extracted signals\yogurt 20 ml edited"  
output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\temporally aligned and averaged"  
align_channel = 18  
n_channels = 22
# ===========================================================================

def get_txt_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.txt')]

def load_data(file_path):
    # Assumes 1 header row, 22 columns, no time column
    return pd.read_csv(file_path, delimiter=',', header=0).values

def align_data_on_peak(data, channel):
    center_idx = np.argmax(data[:, channel])
    return data, center_idx

def pad_and_align(all_data, all_centers):
    # Find the max left and right extents needed
    max_left = max(all_centers)
    max_right = max([data.shape[0] - center - 1 for data, center in zip(all_data, all_centers)])
    total_len = max_left + max_right + 1
    aligned = []
    for data, center in zip(all_data, all_centers):
        pad_left = max_left - center
        pad_right = max_right - (data.shape[0] - center - 1)
        padded = np.pad(data, ((pad_left, pad_right), (0, 0)), mode='constant', constant_values=np.nan)
        aligned.append(padded)
    return np.array(aligned), max_left

def main():
    folder = input_folder
    if not folder or not os.path.isdir(folder):
        Tk().withdraw()
        folder = filedialog.askdirectory(title="Select input folder with .txt files")
        if not folder:
            print("No folder selected.")
            return
    files = get_txt_files(folder)
    if not files:
        print("No .txt files found in the folder.")
        return
    # Convert align_channel from 1-based to 0-based for internal use
    align_channel_idx = align_channel - 1
    all_data = []
    all_centers = []
    labels = []
    for file in files:
        data = load_data(file)
        data, center = align_data_on_peak(data, align_channel_idx)
        all_data.append(data)
        all_centers.append(center)
        labels.append(os.path.basename(file))
    aligned, center_index = pad_and_align(all_data, all_centers)
    fig, axes = plt.subplots(n_channels, 1, figsize=(8, 1.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    # Create legend handles and labels
    legend_handles = []
    legend_labels = []
    
    x = np.arange(aligned.shape[1])
    for ch in range(n_channels):
        ax = axes[ch]
        for i, arr in enumerate(aligned):
            line, = ax.plot(x, arr[:, ch], alpha=0.7, linewidth=0.7)
            
            # Store legend info only once (for first channel)
            if ch == 0:
                legend_handles.append(line)
                legend_labels.append(labels[i])
        
        ax.set_ylabel(f'{ch+1}', fontsize=8)
        ax.tick_params(axis='y', labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add a small amount of space between subplots
        ax.margins(y=0.1)
    
    # Add legend to the figure
    fig.legend(legend_handles, legend_labels, loc='center right', bbox_to_anchor=(0.9, 0.5), fontsize=8)
    
    plt.subplots_adjust(left=0.4, right=0.6)
    plt.show()

    # Compute the average across all files for each channel at every index (ignore NaNs)
    avg_data = np.nanmean(aligned, axis=0)  # shape: (n_samples, n_channels)

    # Plot the averaged data in the same format
    fig, axes = plt.subplots(n_channels, 1, figsize=(8, 1.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    x = np.arange(avg_data.shape[0])
    for ch in range(n_channels):
        ax = axes[ch]
        ax.plot(x, avg_data[:, ch], color='black', linewidth=0.7, label='Average')
        ax.set_ylabel(f'{ch+1}', fontsize=8)
        ax.tick_params(axis='y', labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(y=0.1)
    axes[-1].set_xlabel('Aligned sample index', fontsize=10)
    plt.subplots_adjust(left=0.4, right=0.6)
    plt.suptitle('Averaged Aligned Data', fontsize=14)
    plt.show()

    # Ask user if they want to save the averaged data
    save = input('Do you want to save the averaged data? (y/n): ').strip().lower()
    if save == 'y':
        # Save to output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        out_path = os.path.join(output_folder, f'{os.path.basename(input_folder)} temporal average.txt')
        np.savetxt(out_path, avg_data, delimiter=',', fmt='%.6f')
        print(f'Averaged data saved to {out_path}')

if __name__ == "__main__":
    main()
