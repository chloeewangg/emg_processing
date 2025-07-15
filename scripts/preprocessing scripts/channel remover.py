#!/usr/bin/env python3
"""
This script removes specified channels from EMG data by converting them to NaN values.
It automatically detects header rows and preserves them in the output file.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ======= CONFIGURATION =======

# Input file path (full path to the file you want to process)
INPUT_FILE_PATH = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\all bandpass 20_200 and notch\preprocessed\yogurt 5 ml 3.txt"

# Output folder path (where to save the processed file)
OUTPUT_FOLDER_PATH = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\all bandpass 20_200 and notch\preprocessed"

# Sampling rate for plotting (Hz)
SAMPLING_RATE = 500

# Plot configuration
PLOT_DATA = True  # Set to False to disable plotting
TRIM_SECONDS = 1  # Number of seconds to trim from start and end for plotting

# =============================

def plot_emg_data(data, time, title, channels_to_highlight=None):
    """
    Plot EMG data for all channels.
    
    Parameters:
    data (pd.DataFrame): EMG data (samples x channels)
    time (np.array): Time array in seconds
    title (str): Plot title
    channels_to_highlight (list): List of channel indices to highlight in red
    """
    num_channels = data.shape[1]
    
    # Create subplots
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]
    
    for ch in range(num_channels):
        ax = axes[ch]
        
        # Plot the channel data
        if channels_to_highlight and ch in channels_to_highlight:
            ax.plot(time, data.iloc[:, ch], color='red', linewidth=1, label=f'Ch {ch+1} (Removed)')
        else:
            ax.plot(time, data.iloc[:, ch], color='blue', linewidth=1, label=f'Ch {ch+1}')
        
        # Format the subplot
        ax.set_ylabel(f'{ch+1}', rotation=0, ha='right', va='center', labelpad=10)
        ax.autoscale(enable=True, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)
        
        # Only show x-axis labels on the bottom subplot
        if ch < len(axes) - 1:
            ax.set_xticks([])
        else:
            # Set x-axis ticks every second
            xticks = np.arange(np.ceil(time[0]), np.floor(time[-1]) + 1, 1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(tick)) for tick in xticks])
            ax.set_xlabel('Time (s)')
    
    plt.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.show()

def detect_header_rows(file_path):
    """
    Detect the number of header rows in the data file.
    
    Parameters:
    file_path (str): Path to the input file
    
    Returns:
    int: Number of header rows
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the first line that contains numeric data
    for i, line in enumerate(lines):
        try:
            # Try to convert the line to numeric values
            values = [float(x.strip()) for x in line.strip().split(',') if x.strip()]
            if len(values) > 0:
                return i
        except ValueError:
            continue
    
    return 0

def get_user_channels_to_remove(num_channels):
    """
    Get user input for channels to remove.
    
    Parameters:
    num_channels (int): Total number of channels
    
    Returns:
    list: List of channel indices to remove
    """
    print(f"\nData has {num_channels} channels (0-{num_channels-1})")
    print("Enter channel numbers to remove (0-based indexing), separated by spaces")
    print("Press Enter to remove no channels and save the file as-is")
    
    while True:
        try:
            channels_input = input("Channels to remove: ").strip()
            
            if not channels_input:
                print("No channels will be removed.")
                return []
            
            # Parse channel numbers
            channels_to_remove = [int(x) for x in channels_input.split()]
            
            # Validate channel numbers
            invalid_channels = [ch for ch in channels_to_remove if ch < 0 or ch >= num_channels]
            if invalid_channels:
                print(f"Invalid channel numbers: {invalid_channels}")
                print(f"Please enter channel numbers between 0 and {num_channels-1}")
                continue
            
            print(f"Channels to remove: {channels_to_remove}")
            return channels_to_remove
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None

def remove_channels(input_file, output_folder):
    """
    Remove specified channels from EMG data by setting them to NaN values.
    
    Parameters:
    input_file (str): Path to the input text file
    output_folder (str): Path to the output folder
    """
    try:
        # Generate output filename
        input_filename = os.path.basename(input_file)
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}.txt"
        output_file = os.path.join(output_folder, output_filename)
        
        # Detect header rows
        header_rows = detect_header_rows(input_file)
        print(f"Detected {header_rows} header rows")
        
        # Read the data, skipping header rows
        data = pd.read_csv(input_file, skiprows=header_rows, header=None)
        print(f"Data shape: {data.shape}")
        
        # Create time array for plotting
        n_samples = len(data)
        time = np.arange(n_samples) / SAMPLING_RATE
        
        # Plot original data if requested
        if PLOT_DATA:
            # Exclude first and last TRIM_SECONDS
            trim_samples = int(TRIM_SECONDS * SAMPLING_RATE)
            if trim_samples * 2 < n_samples:
                plot_data = data.iloc[trim_samples:-trim_samples].reset_index(drop=True)
                plot_time = time[trim_samples:-trim_samples]
                plot_emg_data(plot_data, plot_time, f"EMG Data - {os.path.basename(input_file)}")
            else:
                print(f"Warning: Data too short to trim {TRIM_SECONDS} seconds from each end")
                plot_emg_data(data, time, f"EMG Data - {os.path.basename(input_file)}")
        
        # Get user input for channels to remove
        channels_to_remove = get_user_channels_to_remove(data.shape[1])
        if channels_to_remove is None:
            return
        
        # Convert specified channels to NaN
        for channel in channels_to_remove:
            data.iloc[:, channel] = np.nan
            print(f"Channel {channel} converted to NaN")
        
        # Plot modified data if channels were removed and plotting is enabled
        if channels_to_remove and PLOT_DATA:
            # Exclude first and last TRIM_SECONDS
            trim_samples = int(TRIM_SECONDS * SAMPLING_RATE)
            if trim_samples * 2 < n_samples:
                plot_data = data.iloc[trim_samples:-trim_samples].reset_index(drop=True)
                plot_time = time[trim_samples:-trim_samples]
                plot_emg_data(plot_data, plot_time, 
                            f"Modified EMG Data (Channels {channels_to_remove} Removed) - {os.path.basename(input_file)}", 
                            channels_to_highlight=channels_to_remove)
            else:
                plot_emg_data(data, time, 
                            f"Modified EMG Data (Channels {channels_to_remove} Removed) - {os.path.basename(input_file)}", 
                            channels_to_highlight=channels_to_remove)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Write the output file with headers and modified data
        with open(output_file, 'w') as f:
            # Write the header rows
            with open(input_file, 'r') as input_f:
                for i in range(header_rows):
                    f.write(input_f.readline())
            
            # Write the modified data
            data.to_csv(f, index=False, header=False, na_rep='NaN')
        
        print(f"Data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

def validate_configuration():
    """Validate the configuration parameters."""
    errors = []
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE_PATH):
        errors.append(f"Input file does not exist: {INPUT_FILE_PATH}")
    
    # Check if output folder path is valid
    if not OUTPUT_FOLDER_PATH:
        errors.append("OUTPUT_FOLDER_PATH cannot be empty")
    
    # Check sampling rate
    if SAMPLING_RATE <= 0:
        errors.append(f"Sampling rate must be positive: {SAMPLING_RATE}")
    
    # Check trim seconds
    if TRIM_SECONDS < 0:
        errors.append(f"TRIM_SECONDS must be non-negative: {TRIM_SECONDS}")
    
    return errors

def main():
    """Main function to execute the channel removal process."""
    # Validate configuration
    config_errors = validate_configuration()
    if config_errors:
        print("Configuration errors found:")
        for error in config_errors:
            print(f"  - {error}")
        print("\nPlease fix the configuration and run again.")
        return
    
    # Generate output filename for display
    input_filename = os.path.basename(INPUT_FILE_PATH)
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}.txt"
    
    # Display configuration
    print("Configuration:")
    print(f"  Input file: {INPUT_FILE_PATH}")
    print(f"  Output folder: {OUTPUT_FOLDER_PATH}")
    print(f"  Output file: {output_filename}")
    print(f"  Sampling rate: {SAMPLING_RATE} Hz")
    print(f"  Plot data: {PLOT_DATA}")
    print(f"  Trim seconds: {TRIM_SECONDS} seconds from start and end for plotting")
    print()
    
    # Execute the channel removal
    remove_channels(INPUT_FILE_PATH, OUTPUT_FOLDER_PATH)

if __name__ == "__main__":
    main()
