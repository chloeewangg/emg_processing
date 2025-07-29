import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
import glob

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================7
# Input file path   
INPUT_FILE = r"C:\Users\chloe\Documents\FreeBCI_GUI\Recordings\Stream_2025_07_28_101628\1.txt"

# Filter parameters
SAMPLING_RATE = 500  # Hz
LOW_CUTOFF = 20  # Hz
HIGH_CUTOFF = 200  # Hz
TRANSITION_WIDTH_PERCENT = 20  # % of cutoff frequencies

# Data parameters
NUM_HEADER_ROWS = 6
NUM_PROCESSED_CHANNELS = 16  # Channels 1-16 to be processed
TOTAL_CHANNELS = 22  # Total number of channels in the data

# Plotting parameters
EXCLUDE_SECONDS = 1  # Seconds to exclude from start and end

# Channels to plot (1-based indexing, e.g., [1, 2, 3, 4] for channels 1, 2, 3, 4)
# Leave empty or set to None to plot all channels
CHANNELS_TO_PLOT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # Change this to a list like [1, 2, 3, 4] to plot specific channels

# Output directories (user should set these)
ORAL_PREP_OUTPUT_DIR = r"C:\Users\chloe\OneDrive\Desktop\swallow EMG\data\07_18_25\extracted signals\oral prep"
SWALLOW_OUTPUT_DIR = r"C:\Users\chloe\OneDrive\Desktop\swallow EMG\data\07_18_25\extracted signals\grape"

# Segment extraction parameter
SEGMENT_DURATION = 2  # seconds (set this value as needed)

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

def apply_filters(data, sampling_rate, low_cutoff, high_cutoff, transition_width_percent=20):
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
    
    # Apply bandpass filter
    filtered_data = filtfilt(bandpass_b, bandpass_a, data, axis=0)
    
    # Apply notch filter at 60 Hz
    filtered_data = filtfilt(notch_b_60, notch_a_60, filtered_data, axis=0)
    # Apply notch filter at 120 Hz
    filtered_data = filtfilt(notch_b_120, notch_a_120, filtered_data, axis=0)
    # Apply notch filter at 180 Hz
    filtered_data = filtfilt(notch_b_180, notch_a_180, filtered_data, axis=0)
    
    return filtered_data

def get_split_times():
    """
    Prompt the user for split times (in seconds) as a comma-separated list.
    Returns a sorted list of floats.
    """
    while True:
        split_str = input("Enter split times in seconds: ")
        try:
            split_times = [float(s.strip()) for s in split_str.split(",") if s.strip()]
            split_times = sorted(split_times)
            return split_times
        except Exception:
            print("Invalid input. Please enter numbers separated by commas.")

def get_filenames(num_segments):
    """
    Prompt the user for output filenames for each segment.
    Returns a list of filenames.
    """
    filenames = []
    for i in range(num_segments):
        fname = input(f"Enter filename for segment {i+1} (without extension): ")
        if not fname.lower().endswith('.txt'):
            fname += '.txt'
        filenames.append(fname)
    return filenames

def save_segments(output_data, split_times, filenames, oral_prep_dir):
    """
    Save data segments to files based on split times and filenames.
    Only save all but the last segment. Use commas as delimiters and preserve significant figures as in the original file.
    """
    num_samples = output_data.shape[0]
    split_samples = [int(t * SAMPLING_RATE) for t in split_times]
    split_samples = [0] + split_samples + [num_samples]
    # Only save all but the last segment
    for i in range(len(split_samples) - 2):
        seg_data = output_data[split_samples[i]:split_samples[i+1], :]
        out_dir = oral_prep_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filenames[i])
        df = pd.DataFrame(seg_data)
        df.to_csv(out_path, index=False, header=False, float_format='%.6g', sep=',')
        print(f"Saved segment {i+1} to {out_path}")

def plot_preprocessed_data(input_file, return_data=False):
    """
    Process a single EMG file and plot all 22 channels. If return_data is True, return the filtered data array.
    
    Parameters:
    -----------
    input_file : str
        Path to input file
    return_data : bool
        If True, return the filtered data array instead of plotting
    
    Returns:
    --------
    numpy.ndarray or None
        Filtered data array if return_data is True, None otherwise
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
            return None
        
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
        
        # Set x-ticks every second
        max_time = time_axis[-1] if len(time_axis) > 0 else 0
        x_ticks = np.arange(0, max_time + 1, 1)
        
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
            
            # Set x-ticks every second
            ax.set_xticks(x_ticks)
            
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
        if return_data:
            return output_data
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        if return_data:
            return None

# =============================================================================
# NEW SEGMENT EXTRACTION FUNCTIONS
# =============================================================================
def get_center_times():
    """
    Prompt the user for center times (in seconds) as a comma-separated list.
    Returns a sorted list of floats.
    """
    while True:
        center_str = input("Enter center times in seconds (comma-separated): ")
        try:
            center_times = [float(s.strip()) for s in center_str.split(",") if s.strip()]
            center_times = sorted(center_times)
            return center_times
        except Exception:
            print("Invalid input. Please enter numbers separated by commas.")

def get_filenames_for_segments(num_segments):
    """
    Prompt the user for output filenames for each segment except the last.
    Returns a list of filenames.
    """
    filenames = []
    for i in range(num_segments):
        fname = input(f"Enter filename for segment {i+1} (without extension): ")
        if not fname.lower().endswith('.txt'):
            fname += '.txt'
        filenames.append(fname)
    return filenames

def extract_segments_centered(output_data, center_times, segment_duration, sampling_rate):
    """
    Extract segments of specified duration centered at each center time.
    Returns a list of numpy arrays (segments).
    """
    num_samples = output_data.shape[0]
    total_time = num_samples / sampling_rate
    half_window = segment_duration / 2
    segments = []
    segment_indices = []
    for center in center_times:
        start_time = center - half_window
        end_time = center + half_window
        # Clamp to data bounds
        start_idx = max(0, int(np.round(start_time * sampling_rate)))
        end_idx = min(num_samples, int(np.round(end_time * sampling_rate)))
        # If segment is shorter than requested (at edges), pad with zeros
        segment = np.zeros((int(segment_duration * sampling_rate), output_data.shape[1]))
        seg_data = output_data[start_idx:end_idx, :]
        seg_len = seg_data.shape[0]
        if seg_len > 0:
            segment[:seg_len, :] = seg_data
        segments.append(segment)
        segment_indices.append((start_idx, end_idx))
    return segments, segment_indices

def plot_segment(segment_data, sampling_rate, channels_to_plot, input_file=None, center_time=None):
    """
    Plot a segment of EMG data (all or selected channels).
    """
    time_axis = np.arange(segment_data.shape[0]) / sampling_rate
    num_channels_to_plot = len(channels_to_plot)
    fig, axes = plt.subplots(num_channels_to_plot, 1, figsize=(12, 3 * num_channels_to_plot), sharex=True)
    if num_channels_to_plot == 1:
        axes = [axes]
    # Set x-ticks every second
    max_time = time_axis[-1] if len(time_axis) > 0 else 0
    x_ticks = np.arange(0, max_time + 1, 1)
    for i, ch in enumerate(channels_to_plot):
        ch_idx = ch - 1
        ax = axes[i]
        ax.plot(time_axis, segment_data[:, ch_idx], linewidth=0.7, color='black')
        ch_data = segment_data[:, ch_idx]
        min_val = np.min(ch_data)
        max_val = np.max(ch_data)
        ax.set_ylabel(f'{ch}', fontsize=8)
        ax.set_ylim(min_val, max_val)
        tick_start = np.floor(min_val / 10) * 10
        tick_end = np.ceil(max_val / 10) * 10
        tick_positions = np.arange(tick_start, tick_end + 1, 10)
        if min_val not in tick_positions:
            tick_positions = np.append(tick_positions, min_val)
        if max_val not in tick_positions:
            tick_positions = np.append(tick_positions, max_val)
        if min_val < 0 < max_val and 0 not in tick_positions:
            tick_positions = np.append(tick_positions, 0)
        tick_positions = np.sort(tick_positions)
        # ax.set_yticks(tick_positions)
        # ax.set_yticklabels([f'{tp:.1f}' for tp in tick_positions], fontsize=6, rotation=0)
        ax.set_xticks(x_ticks)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(y=0.2)
    axes[-1].set_xlabel('Time (seconds)', fontsize=10)
    title = 'Extracted Segment'
    if input_file is not None:
        title += f' from {os.path.basename(input_file)}'
    if center_time is not None:
        title += f' (centered at {center_time:.2f}s)'
    plt.suptitle(title, fontsize=12)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def save_extracted_segments(segments, filenames, input_file, oral_prep_dir, swallow_dir, center_times=None):
    """
    Plot and save all but the last segment to oral_prep_dir with user-specified filenames.
    Plot and save the last segment to swallow_dir with the same name as the input file.
    """
    # Save all but the last segment
    for i, seg in enumerate(segments[:-1]):
        # Plot the segment before saving
        ct = center_times[i] if center_times is not None and i < len(center_times) else None
        plot_segment(seg, SAMPLING_RATE, CHANNELS_TO_PLOT, input_file=input_file, center_time=ct)
        out_dir = oral_prep_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filenames[i])
        df = pd.DataFrame(seg)
        df.to_csv(out_path, index=False, header=False, float_format='%.6g', sep=',')
        print(f"Saved segment {i+1} to {out_path}")
    # Save the last segment
    last_seg = segments[-1]
    ct = center_times[-1] if center_times is not None and len(center_times) == len(segments) else None
    plot_segment(last_seg, SAMPLING_RATE, CHANNELS_TO_PLOT, input_file=input_file, center_time=ct)
    out_dir = swallow_dir
    os.makedirs(out_dir, exist_ok=True)
    input_basename = os.path.basename(input_file)
    if not input_basename.lower().endswith('.txt'):
        input_basename += '.txt'
    out_path = os.path.join(out_dir, input_basename)
    df = pd.DataFrame(last_seg)
    df.to_csv(out_path, index=False, header=False, float_format='%.6g', sep=',')
    print(f"Saved last segment to {out_path}")

# =============================================================================
# MAIN PROCESSING SCRIPT
# =============================================================================

def main():
    """
    Main function to process a single EMG file and plot the results.
    """
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
    
    # Process and plot the file, and get filtered data
    output_data = plot_preprocessed_data(input_file, return_data=True)
    if output_data is None:
        print("Error: No data to extract.")
        return
    segment_duration = SEGMENT_DURATION
    # Ask user for center times
    total_time = output_data.shape[0] / SAMPLING_RATE
    while True:
        center_times = get_center_times()
        # Validate center times: ensure segments fit within data bounds
        valid_centers = []
        for c in center_times:
            if (c - segment_duration/2) >= 0 and (c + segment_duration/2) <= total_time:
                valid_centers.append(c)
            else:
                print(f"Warning: Center time {c} would exceed data bounds and will be skipped.")
        if valid_centers:
            center_times = valid_centers
            break
        else:
            print("No valid center times entered. Please try again.")
    # Extract segments
    segments, segment_indices = extract_segments_centered(output_data, center_times, segment_duration, SAMPLING_RATE)
    if len(segments) == 0:
        print("No valid segments to save.")
        return
    # Ask user for filenames (all but the last segment)
    if len(segments) > 1:
        filenames = get_filenames_for_segments(len(segments)-1)
    else:
        filenames = []
    # Save segments
    save_extracted_segments(segments, filenames, input_file, ORAL_PREP_OUTPUT_DIR, SWALLOW_OUTPUT_DIR, center_times=center_times)

if __name__ == "__main__":
    main()
