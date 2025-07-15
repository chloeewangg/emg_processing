import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import glob

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Specify input and output folders
INPUT_FOLDER = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\raw txt"  # Change this to your input folder path
OUTPUT_FOLDER = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\all bandpass 20_450 and notch"  # Change this to your output folder path

# Filter parameters
SAMPLING_RATE = 500  # Hz
LOW_CUTOFF = 20  # Hz
HIGH_CUTOFF = 200  # Hz
TRANSITION_WIDTH_PERCENT = 20  # % of cutoff frequencies

# Data parameters
NUM_HEADER_ROWS = 6
NUM_PROCESSED_CHANNELS = 16  # Channels 1-16 to be processed
TOTAL_CHANNELS = 22  # Total number of channels in the data

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_bandpass_filter(low_cutoff, high_cutoff, sampling_rate, transition_width_percent=20):
    """
    Create a bandpass filter with specified transition width.
    
    Parameters:
    -----------
    low_cutoff : float
        Lower cutoff frequency in Hz
    high_cutoff : float
        Upper cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    transition_width_percent : float
        Transition width as percentage of cutoff frequencies
    
    Returns:
    --------
    b, a : tuple
        Filter coefficients
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
    
    Parameters:
    -----------
    notch_freq : float
        Frequency to notch out in Hz
    sampling_rate : float
        Sampling rate in Hz
    quality_factor : float
        Quality factor of the notch filter
    
    Returns:
    --------
    b, a : tuple
        Filter coefficients
    """
    b, a = iirnotch(notch_freq, quality_factor, sampling_rate)
    return b, a

def trim_leading_zeros(data):
    """
    Trim leading zeros from the data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    
    Returns:
    --------
    numpy.ndarray
        Data with leading zeros removed
    """
    # Find the first non-zero row
    non_zero_rows = np.any(data != 0, axis=1)
    first_non_zero = np.where(non_zero_rows)[0]
    
    if len(first_non_zero) > 0:
        start_idx = first_non_zero[0]
        return data[start_idx:]
    else:
        return data

def apply_filters(data, sampling_rate, low_cutoff, high_cutoff, notch_freq, transition_width_percent=20):
    """
    Apply bandpass and notch filters to the data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array (samples x channels)
    sampling_rate : float
        Sampling rate in Hz
    low_cutoff : float
        Lower cutoff frequency for bandpass filter
    high_cutoff : float
        Upper cutoff frequency for bandpass filter
    notch_freq : float
        Frequency to notch out (first notch)
    transition_width_percent : float
        Transition width as percentage of cutoff frequencies
    
    Returns:
    --------
    numpy.ndarray
        Filtered data
    """
    # Create filters
    bandpass_b, bandpass_a = create_bandpass_filter(low_cutoff, high_cutoff, sampling_rate, transition_width_percent)
    notch_b_60, notch_a_60 = create_notch_filter(60, sampling_rate)
    notch_b_120, notch_a_120 = create_notch_filter(120, sampling_rate)
    notch_b_180, notch_a_180 = create_notch_filter(180, sampling_rate)
    # notch_b_200, notch_a_200 = create_notch_filter(200, sampling_rate)
    
    # Apply bandpass filter
    filtered_data = filtfilt(bandpass_b, bandpass_a, data, axis=0)
    
    # Apply notch filter at 60 Hz
    filtered_data = filtfilt(notch_b_60, notch_a_60, filtered_data, axis=0)
    # Apply notch filter at 120 Hz
    filtered_data = filtfilt(notch_b_120, notch_a_120, filtered_data, axis=0)
    # Apply notch filter at 180 Hz
    filtered_data = filtfilt(notch_b_180, notch_a_180, filtered_data, axis=0)
    # Apply notch filter at 200 Hz
    # filtered_data = filtfilt(notch_b_200, notch_a_200, filtered_data, axis=0)
    
    return filtered_data

def process_emg_file(input_file, output_file):
    """
    Process a single EMG file.
    
    Parameters:
    -----------
    input_file : str
        Path to input file
    output_file : str
        Path to output file
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
            NOTCH_FREQ, 
            TRANSITION_WIDTH_PERCENT
        )
        
        # Combine processed and unprocessed channels
        output_data = np.column_stack([filtered_channels, unprocessed_channels])
        
        # Create header information
        header_lines = []
        header_lines.append(f"# EMG Data - Preprocessed")
        header_lines.append(f"# Original file: {os.path.basename(input_file)}")
        header_lines.append(f"# Sampling rate: {SAMPLING_RATE} Hz")
        header_lines.append(f"# Bandpass filter: {LOW_CUTOFF}-{HIGH_CUTOFF} Hz")
        header_lines.append(f"# Notch filter: 60 Hz, 200 Hz")
        header_lines.append(f"# Channels 1-{NUM_PROCESSED_CHANNELS}: processed, Channels {NUM_PROCESSED_CHANNELS+1}-{TOTAL_CHANNELS}: unprocessed")
        
        # Save the processed data
        np.savetxt(output_file, output_data, delimiter=',', header='\n'.join(header_lines), comments='')
        
        print(f"Saved: {os.path.basename(output_file)}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

# =============================================================================
# MAIN PROCESSING SCRIPT
# =============================================================================

def main():
    """
    Main function to process all EMG files in the input folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all .txt files in the input folder
    input_files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
    
    if not input_files:
        print(f"No .txt files found in {INPUT_FOLDER}")
        return
    
    print(f"Found {len(input_files)} files to process")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Preserving channels {NUM_PROCESSED_CHANNELS+1}-{TOTAL_CHANNELS} unprocessed")
    print("-" * 80)
    
    # Process each file
    for input_file in input_files:
        # Create output filename
        filename = os.path.basename(input_file)
        output_file = os.path.join(OUTPUT_FOLDER, filename)
        
        # Process the file
        process_emg_file(input_file, output_file)
    
    print("-" * 80)
    print("Processing complete!")

if __name__ == "__main__":
    main()
