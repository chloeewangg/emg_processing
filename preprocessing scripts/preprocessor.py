'''
This script is used to preprocess EMG data by applying a bandpass filter and trimming leading zeros.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def design_bandpass(sampling_rate, f_low, f_high, transition_frac=0.2, gpass=1, gstop=40):
    """
    Design a bandpass filter with a transition width that is 20% of each cutoff frequency.
    Returns filter coefficients (b, a).
    """
    nyquist = sampling_rate / 2
    # Passband edges (normalized)
    wp = [f_low / nyquist, f_high / nyquist]
    # Stopband edges (normalized)
    ws = [
        max((f_low - transition_frac * f_low) / nyquist, 1e-6),
        min((f_high + transition_frac * f_high) / nyquist, 1 - 1e-6)
    ]
    b, a = signal.iirdesign(wp, ws, gpass=gpass, gstop=gstop, ftype='butter')
    return b, a

def apply_filters(data, sampling_rate):
    """
    Apply bandpass (70-110 Hz) filter to the first 16 columns of the data (columns 0-15).
    Args:
        data (DataFrame): EMG data
        sampling_rate (int): Sampling rate in Hz
    Returns:
        DataFrame: Filtered EMG data
    """
    filtered_data = data.copy()
    f_low = 70
    f_high = 110
    try:
        b_bp, a_bp = design_bandpass(sampling_rate, f_low, f_high, transition_frac=0.2)
    except Exception as e:
        print(f"Warning: Could not design bandpass filter: {e}. Skipping bandpass filter.")
        b_bp, a_bp = [1], [0]
    # Apply filter to the first 16 columns
    for i in range(min(16, data.shape[1])):
        emg_data = data.iloc[:, i].values
        filtered_emg = emg_data
        # Apply bandpass filter (if valid)
        if b_bp[0] != 1 or a_bp[0] != 0:
            filtered_emg = signal.filtfilt(b_bp, a_bp, filtered_emg)
        filtered_data.iloc[:, i] = filtered_emg
    return filtered_data

def trim_leading_zeros(data):
    """
    Trim all rows before the first row where any value in any column becomes nonzero.
    Applies to all columns.
    """
    nonzero_mask = (data != 0).any(axis=1)
    if not nonzero_mask.any():
        print("Warning: All data is zero. No trimming applied.")
        return data
    first_nonzero_idx = nonzero_mask.idxmax()
    trimmed_data = data.loc[first_nonzero_idx:].reset_index(drop=True)
    return trimmed_data

def process_emg_files(input_folder, output_folder, sampling_rate=500, window_size=125, threshold=3):
    """
    Process all .txt files in the input_folder: apply bandpass filter, Hampel filter, and save to output_folder.
    Args:
        input_folder (str): Path to folder containing raw EMG .txt files
        output_folder (str): Path to folder to save processed files
        sampling_rate (int): Sampling rate in Hz (default 500)
        window_size (int): Hampel filter window size (default 7)
        threshold (float): Hampel filter threshold in MADs (default 3)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            try:
                # Load data, skip first 6 header rows
                data = pd.read_csv(input_path, sep=',', skiprows=6, header=None)
                data.columns = ['Time'] + [f'EMG_Ch{i}' for i in range(1, len(data.columns))]
                # Convert all columns to numeric
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                data = data.dropna()
                # Trim leading zeros
                data = trim_leading_zeros(data)
                # Apply bandpass filter only to first 16 columns
                filtered_data = apply_filters(data, sampling_rate)
                # Save processed data
                output_path = os.path.join(output_folder, filename)
                filtered_data.to_csv(output_path, sep=',', index=False)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Example usage: update these paths as needed
    input_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\with noise spikes"
    output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\with hampel filter"
    process_emg_files(input_folder, output_folder)