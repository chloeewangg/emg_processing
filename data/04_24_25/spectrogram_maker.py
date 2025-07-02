import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import signal
import pandas as pd

def read_labchart_file(file_path):
    """
    Read a LabChart EMG text file.
    Returns a dictionary with channel data and metadata.
    """
    data = {}
    channel_data = {}
    current_channel = None
    sampling_rate = None
    data_start = None  # Initialize data_start
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process header information and data
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check for sampling rate
        if "Sample rate:" in line:
            parts = line.split()
            try:
                sampling_rate = float(parts[-2])  # Assuming format like "Sample rate: 1000 Hz"
            except (ValueError, IndexError):
                print("Warning: Could not parse sampling rate.")
                
        # Check for channel information
        elif line.startswith("Channel"):
            try:
                current_channel = line.split()[1]
                channel_data[current_channel] = []
            except IndexError:
                print(f"Warning: Could not parse channel information: {line}")
                
        # Check for data start
        elif line.startswith("Interval") and "msec" in line:
            data_start = i + 1
            break
    
    # Process data section
    if data_start is not None:  # Check if data_start is defined
        # First determine the number of channels and their order
        header_line = lines[data_start].strip()
        column_headers = header_line.split('\t')
        
        # Read data using pandas for better handling of tab-delimited data
        data_df = pd.read_csv(file_path, sep='\t', skiprows=data_start+1, names=column_headers)
        
        # Extract time column (typically the first column)
        time_col = column_headers[0]
        times = data_df[time_col].values
        
        # Extract each channel's data
        for col in column_headers[1:]:
            channel_name = col.strip()
            if channel_name:  # Skip empty column names
                channel_data[channel_name] = data_df[col].values
    else:
        print("Warning: Could not find data section in file.")
        times = []
    
    # Create the final data dictionary
    data['channels'] = channel_data
    data['sampling_rate'] = sampling_rate
    data['times'] = times
    
    return data

def apply_notch_filters(data, sampling_rate, notch_freqs=[60, 120, 140, 180, 240, 280, 300, 360, 420], Q=30):
    """
    Apply multiple notch filters to the data.
    """
    filtered_data = data.copy()
    
    for freq in notch_freqs:
        b, a = signal.iirnotch(freq, Q, sampling_rate)
        filtered_data = signal.filtfilt(b, a, filtered_data)
    
    return filtered_data

def apply_bandpass_filter(data, sampling_rate, low_freq, high_freq, order=4):
    """
    Apply a bandpass filter to the data.
    """
    nyquist = 0.5 * sampling_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def create_spectrogram(data, sampling_rate, nperseg=256, noverlap=128, db_range=None, freq_range=None, 
                       time_range=None, window='hann'):
    """
    Create a spectrogram from the data.
    """
    # Select time range if specified
    if time_range and len(time_range) == 2:
        start_idx = int(time_range[0] * sampling_rate)
        end_idx = int(time_range[1] * sampling_rate)
        data_segment = data[start_idx:end_idx]
    else:
        data_segment = data
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(data_segment, fs=sampling_rate, window=window, 
                                  nperseg=nperseg, noverlap=noverlap, 
                                  detrend='constant', return_onesided=True, 
                                  scaling='density', mode='psd')
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx)
    
    # Prepare output with frequency and time axes
    time_offset = 0
    if time_range:
        time_offset = time_range[0]
    
    t_adjusted = t + time_offset
    
    return f, t_adjusted, Sxx_db

def plot_spectrogram(f, t, Sxx_db, channel_name, db_range=None, freq_range=None, title=None):
    """
    Plot the spectrogram with custom settings.
    """
    plt.figure(figsize=(12, 6))
    
    # Set custom color range if specified
    if db_range and len(db_range) == 2:
        vmin, vmax = db_range
    else:
        vmin, vmax = np.percentile(Sxx_db, 5), np.percentile(Sxx_db, 95)
    
    # Create the spectrogram plot
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', 
                  norm=Normalize(vmin=vmin, vmax=vmax))
    
    # Set frequency range if specified
    if freq_range and len(freq_range) == 2:
        plt.ylim(freq_range)
    
    # Add labels and title
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Spectrogram - Channel {channel_name}')
    
    plt.tight_layout()
    
    return plt.gcf()

def main():
    # ======= CONFIGURATION SETTINGS =======
    # Set the file path here
    file_path = "C:/Users/chloe/OneDrive/Desktop/4_24_25_swallow_EMG/dry_swallow_spectrogram.txt"
    
    # Channel options - use channel names as they appear in the file, or numeric indices
    # channels = ["1", "2"]  # Specific channels to analyze by name
    channels = [2, 8]  # Specific channels to analyze by index (0-based)
    # channels = None  # Set to None to process all available channels
    
    # Use channel indices instead of names (True = use 0-based indexing, False = use names)
    use_channel_indices = True
    
    # Time range in seconds (start, end) - set to None to process all time
    # time_range = (5, 15)  # Example: 5-15 seconds
    time_range = None  # Process all available time
    
    # Frequency range for y-axis in Hz (min, max) - set to None for auto-scaling
    freq_range = (10, 500)  # Example: 10-500 Hz
    # freq_range = None  # Auto scale
    
    # dB range for colormap (min, max) - set to None for auto-scaling
    db_range = (-200, -200)  # Example: -80 to -20 dB
    # db_range = None  # Auto scale
    
    # Bandpass filter (low_freq, high_freq) - set to None to skip bandpass filtering
    # bandpass = (20, 500)  # Example: 20-500 Hz bandpass filter
    bandpass = None  # Skip bandpass filtering
    
    # Apply notch filters (60, 120, 140, 180, 240, 280, 300, 360, 420 Hz)
    apply_notch = True  # Set to False to skip notch filtering
    
    # Output file name prefix - set to None to skip saving
    output_prefix = "emg_spectrograms"  # Will save as emg_spectrograms_ch1.png, etc.
    # output_prefix = None  # Don't save files
    
    # Spectrogram parameters
    nperseg = 256  # Length of each segment for FFT
    noverlap = 128  # Overlap between segments
    
    # ======= END CONFIGURATION =======
    
    # Read the data
    print(f"Reading file: {file_path}")
    data = read_labchart_file(file_path)
    
    sampling_rate = data['sampling_rate']
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Get list of channel names
    channel_names = list(data['channels'].keys())
    print("Available channels:")
    for i, ch in enumerate(channel_names):
        print(f"  {i}: {ch}")
    
    # Determine which channels to process
    channels_to_process = []
    
    if channels is None:
        # Process all channels
        channels_to_process = channel_names
        print(f"Processing all channels: {channels_to_process}")
    else:
        # Process specified channels
        if use_channel_indices:
            # Convert indices to channel names
            for idx in channels:
                if 0 <= idx < len(channel_names):
                    channels_to_process.append(channel_names[idx])
                else:
                    print(f"Warning: Channel index {idx} is out of range. Skipping.")
        else:
            # Use channel names directly
            channels_to_process = channels
    
    # Process each channel
    for channel in channels_to_process:
        if channel not in data['channels']:
            print(f"Warning: Channel {channel} not found. Skipping.")
            continue
        
        print(f"Processing channel {channel}...")
        ch_data = data['channels'][channel]
        
        # Apply notch filters
        if apply_notch:
            print("  Applying notch filters...")
            ch_data = apply_notch_filters(ch_data, sampling_rate)
        
        # Apply bandpass filter if requested
        if bandpass and len(bandpass) == 2:
            low_freq, high_freq = bandpass
            print(f"  Applying bandpass filter: {low_freq}-{high_freq} Hz...")
            ch_data = apply_bandpass_filter(ch_data, sampling_rate, low_freq, high_freq)
        
        # Create spectrogram
        print("  Creating spectrogram...")
        f, t, Sxx_db = create_spectrogram(ch_data, sampling_rate, nperseg=nperseg, 
                                         noverlap=noverlap, time_range=time_range)
        
        # Create title with processing details
        title_parts = [f"Channel {channel} Spectrogram"]
        if apply_notch:
            title_parts.append("Notch Filtered")
        if bandpass:
            title_parts.append(f"Bandpass {bandpass[0]}-{bandpass[1]} Hz")
        if time_range:
            title_parts.append(f"Time {time_range[0]}-{time_range[1]} s")
        
        title = " | ".join(title_parts)
        
        # Plot and save
        fig = plot_spectrogram(f, t, Sxx_db, channel, db_range, freq_range, title)
        
        if output_prefix:
            output_file = f"{output_prefix}_ch{channel}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved to {output_file}")
        
        plt.show()
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
