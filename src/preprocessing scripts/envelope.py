'''
This script is used to calculate the RMS envelope of an EMG signal.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ======= CONFIGURATION =======
input_file = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\shorts removed\water 5 ml 1.txt"  
output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\envelopes"  
sampling_rate = 500  
rms_window_sec = 0.05  
show_plots = True  
# =============================

def process_and_plot_envelope(input_file, output_folder, sampling_rate, rms_window_sec, show_plots):
    """
    Loads, processes (rectify, RMS envelope), and plots the first 16 channels of an EMG signal.
    """
    # 1. Load data (comma-delimited, no header, only first 16 columns)
    try:
        data = pd.read_csv(input_file, sep=',', header=None, usecols=range(16), skiprows=1)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Rectify the data
    rectified_data = data.abs()

    # 3. Calculate RMS envelope
    window_samples = int(rms_window_sec * sampling_rate)
    if window_samples < 1:
        window_samples = 1
    
    # Use rolling window to calculate Root Mean Square for each channel
    rms_data = rectified_data.pow(2).rolling(window=window_samples, center=True).mean().pow(0.5)
    
    # Trim the first and last 0.3 seconds for plotting
    trim_samples = int(0.3 * sampling_rate)
    plot_data_rms = rms_data.iloc[trim_samples:-trim_samples].reset_index(drop=True)
    plot_data_rectified = rectified_data.iloc[trim_samples:-trim_samples].reset_index(drop=True)

    # 4. Save the processed RMS data
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create output filename
        base_filename = os.path.basename(input_file)
        name, ext = os.path.splitext(base_filename)
        output_filename = f"{name} rms envelope {rms_window_sec}s {ext}"
        output_path = os.path.join(output_folder, output_filename)

        # Save the data
        plot_data_rms.to_csv(output_path, header=False, index=False)
        print(f"Successfully saved processed data to {output_path}")

    except Exception as e:
        print(f"Error saving file: {e}")
    
    if show_plots:
        # 5. Plot all 16 channels for Rectified Data
        fig2, axes2 = plt.subplots(16, 1, figsize=(12, 16), sharex=True, sharey=True)
        if not hasattr(axes2, "__len__"): # Handle case of 1 subplot
            axes2 = [axes2]

        for i, ax in enumerate(axes2):
            if i < len(plot_data_rectified.columns):
                ax.plot(plot_data_rectified.iloc[:, i], color='orange')
                # Add channel label to the y-axis
                ax.set_ylabel(f'Ch {i+1}', rotation=0, ha='right', va='center', labelpad=10)
                
                # Remove unneeded plot elements, but keep y-axis visible
                ax.set_xticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle("Rectified EMG (16 Channels)")
        plt.show()
        
        # 6. Plot all 16 channels for RMS Envelope
        fig, axes = plt.subplots(16, 1, figsize=(12, 16), sharex=True, sharey=True)
        if not hasattr(axes, "__len__"): # Handle case of 1 subplot
            axes = [axes]

        for i, ax in enumerate(axes):
            if i < len(plot_data_rms.columns):
                ax.plot(plot_data_rms.iloc[:, i])
                # Add channel label to the y-axis
                ax.set_ylabel(f'Ch {i+1}', rotation=0, ha='right', va='center', labelpad=10)
                
                # Remove unneeded plot elements, but keep y-axis visible
                ax.set_xticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle("EMG Envelopes (16 Channels)")
        plt.show()


if __name__ == "__main__":
    process_and_plot_envelope(input_file, output_folder, sampling_rate, rms_window_sec, show_plots)
