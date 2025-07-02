'''
This script is used to calculate the median frequency of an EMG signal.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def apply_filters(data, sampling_rate, apply_notch=True, apply_bandpass=True):
    """
    Apply bandpass (70-110 Hz) and optionally notch filters (60 Hz and harmonics) to EMG data.
    
    Args:
        data (DataFrame): EMG data
        sampling_rate (int): Sampling rate in Hz
        apply_notch (bool): Whether to apply the notch filter(s)
        apply_bandpass (bool): Whether to apply the bandpass filter
    
    Returns:
        DataFrame: Filtered EMG data
    """
    filtered_data = data.copy()
    
    # Design bandpass filter (70-110 Hz)
    nyquist = sampling_rate / 2
    low_freq = 20 / nyquist
    high_freq = 450 / nyquist
    b_bp, a_bp = signal.butter(4, [low_freq, high_freq], btype='band')
    
    # Define notch frequencies (60 Hz and harmonics up to Nyquist)
    notch_frequencies = [f for f in range(60, int(nyquist), 60) if f > 0]
    Q = 30  # Quality factor for notch filters
    
    # Apply filters to each EMG channel
    for i in range(8):
        channel_name = f'EMG_Ch{i+1}'
        emg_data = data[channel_name].values
        
        filtered_emg = emg_data
        if apply_bandpass:
            filtered_emg = signal.filtfilt(b_bp, a_bp, filtered_emg)
        
        if apply_notch:
            for notch_freq_val in notch_frequencies:
                notch_freq_norm = notch_freq_val / nyquist
                # Design and apply notch filter for each frequency
                b_notch, a_notch = signal.iirnotch(notch_freq_norm, Q)
                filtered_emg = signal.filtfilt(b_notch, a_notch, filtered_emg)
        
        filtered_data[channel_name] = filtered_emg
    
    return filtered_data

def plot_emg_data(
    filepath,
    plot_time_interval=(0.4, -0.4),
    window_length_sec=1.0, # default window length in seconds
    step_size_sec=0.1,      # default step size in seconds
    channels_to_plot=None, # List of 1-indexed channels to plot (e.g., [1, 3, 5]). None means plot all.
):
    """
    Load and plot LabChart EMG data from text file.
    Calculate and plot median frequency over time using a sliding window.
    
    Args:
        filepath (str): Path to LabChart text file
        plot_time_interval (tuple): (start_time, end_time) in seconds. If end_time is negative, it is relative to the end.
        window_length_sec (float): The length of the sliding window in seconds.
        step_size_sec (float): The step size for the sliding window in seconds.
        channels_to_plot (list or None): A list of 1-indexed channel numbers to include in the plot. If None, all channels (1-8) are plotted.
    """
    # Load data, skip first 6 header rows
    data = pd.read_csv(filepath, sep='\t', skiprows=6, header=None)
    
    # Set column names: first column is time (mV), columns 1-8 are EMG channels
    data.columns = ['Time'] + [f'EMG_Ch{i}' for i in range(1, 9)]
    
    # Convert all columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Remove any rows with NaN values
    data = data.dropna()
    
    # Create time vector (sampling rate = 1000 Hz)
    sampling_rate = 1000
    window_length = int(window_length_sec * sampling_rate) # Convert window length from seconds to data points
    
    # Convert step size from seconds to data points
    step_size = int(step_size_sec * sampling_rate)
    # Ensure step_size is at least 1
    if step_size < 1:
        step_size = 1
    
    time = np.arange(len(data)) / sampling_rate
    
    # Determine indices for the time interval
    start_time, end_time = plot_time_interval
    if end_time < 0:
        end_time = time[-1] + end_time  # relative to end
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)
    if end_idx == 0:
        end_idx = len(time)
    
    # Trim data to the specified time interval
    trimmed_data = data.iloc[start_idx:end_idx].reset_index(drop=True)
    trimmed_time = time[start_idx:end_idx]
    
    # --- Median Frequency Calculation ---
    median_freqs = []
    window_times = []
    
    # Define exclusion period in data points (0.5 seconds)
    exclusion_points = int(0.5 * sampling_rate)

    # Iterate through the data with a sliding window, excluding the first and last 0.5 seconds
    # The loop starts after the first exclusion_points and ends so that the window doesn't go into the last exclusion_points
    start_loop_idx = exclusion_points
    end_loop_idx = len(trimmed_data) - exclusion_points - window_length + 1

    # Ensure the loop range is valid
    if end_loop_idx > start_loop_idx:
        for i in range(start_loop_idx, end_loop_idx, step_size):
            window_data = trimmed_data.iloc[i : i + window_length]
            window_time_center = trimmed_time[i + window_length // 2]
            
            channel_median_freqs = []
            # Determine which channels to process and plot
            if channels_to_plot is None:
                channels = range(1, 9) # Process all channels
            else:
                channels = channels_to_plot

            for channel_num in channels:
                j = channel_num - 1 # Convert 1-indexed channel number to 0-indexed list/array index
                channel_name = f'EMG_Ch{j+1}'
                emg_data_window = window_data[channel_name].values
                
                # Calculate FFT
                fft_data = np.fft.fft(emg_data_window)
                fft_magnitude = np.abs(fft_data)
                freqs = np.fft.fftfreq(len(emg_data_window), 1/sampling_rate)
                
                # Consider only positive frequencies
                positive_freqs = freqs[:len(freqs)//2]
                positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
                
                # Calculate cumulative sum of power
                cumulative_power = np.cumsum(positive_magnitude)
                total_power = cumulative_power[-1]
                
                # Find the frequency at which the cumulative power is half of the total power
                median_freq_index = np.where(cumulative_power >= total_power / 2)[0][0]
                median_freq = positive_freqs[median_freq_index]
                
                channel_median_freqs.append(median_freq)
                
            median_freqs.append(channel_median_freqs)
            window_times.append(window_time_center)
    else:
        print("Warning: Trimmed data is too short to apply the specified window size and exclusion period.")

    # Convert to numpy arrays for easier plotting
    if median_freqs:
        median_freqs = np.array(median_freqs)
        window_times = np.array(window_times)
        
        # --- Plotting Median Frequency ---
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        # Get the list of selected channels (1-indexed)
        if channels_to_plot is None:
            display_channels = range(1, 9) # Display all channels
            plot_indices = range(8) # Indices in median_freqs array (0-indexed)
        else:
            display_channels = [ch for ch in channels_to_plot if 1 <= ch <= 8]
            # Create a mapping from original channel number to index in the median_freqs array
            original_channels = [ch_num for ch_num in range(1, 9) if ch_num in channels_to_plot]
            plot_indices = [original_channels.index(ch) for ch in display_channels]

        if not display_channels:
            print("Warning: No valid channels selected for plotting.")
        else:
            # Iterate through the indices corresponding to the selected channels in the median_freqs array
            for idx, channel_num in enumerate(display_channels):
                plt.plot(window_times, median_freqs[:, idx], color=colors[channel_num - 1], label=f'Channel {channel_num}', linewidth=1)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Median Frequency (Hz)')
        # Generate plot title from filename
        try:
            filename = filepath.split('/')[-1]
            name_parts = filename.replace('.txt', '').split(' ')
            if len(name_parts) >= 3:
                activity = name_parts[0].capitalize()
                amount = name_parts[1]
                trial = name_parts[-1]
                plot_title = f'{activity} {amount} ml ({trial})'
            else:
                plot_title = f'Median Frequency Plot: {filename}'
            plt.title(plot_title)
        except Exception as e:
            print(f"Could not generate title from filename: {e}")
            plt.title('Median Frequency of EMG Channels Over Time') # Fallback title
        plt.legend()
        plt.grid(True, alpha=0.3)
        # plt.xlim(trimmed_time.iloc[0], trimmed_time.iloc[-1]) # Optional: set x-limits to the trimmed data range
        plt.tight_layout()
        plt.show()
    
# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = "C:/Users/chloe/OneDrive/Desktop/05_08_25 emg/bandpass 20_450 and notches/yogurt 20 ml 2.txt"
    
    try:
        # You can adjust the time interval, window length, and step size here
        plot_emg_data(
            file_path,
            plot_time_interval=(0, 8),  # (start_time, end_time) in seconds
            window_length_sec=0.5, # Window length in seconds
            step_size_sec=0.1,      # Step size in seconds (e.g., 0.05 seconds)
            channels_to_plot=[1,2,3,4,5,6,7,8] # Example: plot only channels 1, 2, 3, and 4
        )
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please update the file_path variable with your actual LabChart file path.")
    except Exception as e:
        print(f"Error: {str(e)}")