'''
This script is used to calculate the SNR of an EMG signal.
'''

import numpy as np
import pandas as pd
import os

# User input for file path and output folder
file_path = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\spikes removed\apple 5 ml 1.txt"
output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\SNR"
sampling_rate = 500

# Load data (assume one header row, no time column)
data = pd.read_csv(file_path, sep=',', header=0)

# Ask user for signal and noise time segments
signal_start = 3.7
signal_end = 4.3
noise_start = 6
noise_end = 7

# Convert times to sample indices
signal_start_idx = int(signal_start * sampling_rate)
signal_end_idx = int(signal_end * sampling_rate)
noise_start_idx = int(noise_start * sampling_rate)
noise_end_idx = int(noise_end * sampling_rate)

# Only use the first 16 channels
emg_data = data.iloc[:, :16]

snr_results = []

for ch in range(16):
    signal_segment = emg_data.iloc[signal_start_idx:signal_end_idx, ch].values
    noise_segment = emg_data.iloc[noise_start_idx:noise_end_idx, ch].values
    
    # Calculate RMS
    signal_rms = np.sqrt(np.mean(signal_segment ** 2))
    noise_rms = np.sqrt(np.mean(noise_segment ** 2))
    
    # Calculate SNR (linear and dB)
    snr_linear = (signal_rms ** 2) / (noise_rms ** 2) if noise_rms != 0 else np.nan
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else np.nan
    
    snr_results.append({'Channel': ch + 1, 'SNR_linear': snr_linear, 'SNR_dB': snr_db})

# Save results to a CSV file with the same base name as the input file
os.makedirs(output_folder, exist_ok=True)
input_base = os.path.splitext(os.path.basename(file_path))[0]
output_path = os.path.join(output_folder, f'{input_base} snr.csv')

# Calculate averages
avg_linear = np.nanmean([res['SNR_linear'] for res in snr_results])
avg_db = np.nanmean([res['SNR_dB'] for res in snr_results])

# Prepare results for DataFrame
rows = [
    {'Channel': res['Channel'], 'SNR linear': res['SNR_linear'], 'SNR dB': res['SNR_dB']}
    for res in snr_results
]
rows.append({'Channel': 'Average', 'SNR linear': avg_linear, 'SNR dB': avg_db})

results_df = pd.DataFrame(rows)
results_df.to_csv(output_path, index=False)

print(f"SNR results saved to {output_path}")
