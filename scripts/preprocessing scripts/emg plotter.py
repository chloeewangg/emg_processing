'''
This script is used to plot the EMG data from a LabChart text file.
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
    show_fft_bandpass=True,
    show_fft_bandpass_notch=True,
    show_channel_plots=True,
    fft_in_db=False,
    apply_bandpass=True,
    apply_notch=True
):
    """
    Load and plot LabChart EMG data from text file.
    
    Args:
        filepath (str): Path to LabChart text file
        plot_time_interval (tuple): (start_time, end_time) in seconds. If end_time is negative, it is relative to the end.
        show_fft_bandpass (bool): Show FFT with bandpass filter only
        show_fft_bandpass_notch (bool): Show FFT with bandpass + notch filter
        show_channel_plots (bool): Show channel time series plots
        fft_in_db (bool): Whether to plot FFT magnitude in dB (True) or linear (False)
        apply_bandpass (bool): Whether to apply the bandpass filter
        apply_notch (bool): Whether to apply the notch filter
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
    time = np.arange(len(data)) / sampling_rate
    
    # Apply bandpass filter only (no notch)
    filtered_data_bp = apply_filters(data, sampling_rate, apply_notch=False, apply_bandpass=apply_bandpass)
    # Apply bandpass + notch filter (or just notch if bandpass is False)
    filtered_data_bp_notch = apply_filters(data, sampling_rate, apply_notch=apply_notch, apply_bandpass=apply_bandpass)
    
    # Determine indices for the time interval
    start_time, end_time = plot_time_interval
    if end_time < 0:
        end_time = time[-1] + end_time  # relative to end
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)
    if end_idx == 0:
        end_idx = len(time)
    trimmed_time = time[start_idx:end_idx]
    trimmed_filtered_data = filtered_data_bp_notch.iloc[start_idx:end_idx].reset_index(drop=True)

    # Plot channel time series if selected
    if show_channel_plots:
        fig, axes = plt.subplots(8, 1, figsize=(12, 16), sharex=True)
        for i in range(8):
            channel_name = f'EMG_Ch{i+1}'
            emg_data = trimmed_filtered_data[channel_name].values
            axes[i].plot(trimmed_time, emg_data, 'b-', linewidth=0.5)
            axes[i].set_title(f'Channel {i+1}', fontsize=9)
            axes[i].set_ylabel('mV')
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        from matplotlib.ticker import MultipleLocator
        axes[-1].xaxis.set_major_locator(MultipleLocator(0.5))
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.1))
        plt.tight_layout()
        plt.show()

    # Plot FFT before and after notch filter, based on user selection
    if show_fft_bandpass:
        print("Plotting FFT with bandpass filter only (no notch)...")
        plot_fft(filtered_data_bp, sampling_rate, title_suffix='(Bandpass Only)', in_db=fft_in_db)
    if show_fft_bandpass_notch:
        print("Plotting FFT with bandpass + notch filter...")
        plot_fft(filtered_data_bp_notch, sampling_rate, title_suffix='(Bandpass + Notch)', in_db=fft_in_db)

def plot_fft(data, sampling_rate, title_suffix='', in_db=False):
    """
    Plot FFT of all EMG channels on one plot.
    
    Args:
        data (DataFrame): EMG data
        sampling_rate (int): Sampling rate in Hz
        title_suffix (str): Suffix to add to the plot title
        in_db (bool): Whether to plot magnitude in dB (True) or linear (False)
    """
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i in range(8):
        channel_name = f'EMG_Ch{i+1}'
        emg_data = data[channel_name].values
        fft_data = np.fft.fft(emg_data)
        fft_magnitude = np.abs(fft_data)
        freqs = np.fft.fftfreq(len(emg_data), 1/sampling_rate)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
        if in_db:
            positive_magnitude = 20 * np.log10(positive_magnitude + 1e-12)
        plt.plot(positive_freqs, positive_magnitude, color=colors[i], 
                label=f'Channel {i+1}', linewidth=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)' if in_db else 'Magnitude')
    plt.title(f'FFT of EMG Channels {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = "C:/Users/chloe/OneDrive/Desktop/05_08_25 emg/bandpass 20_450 and notches/yogurt 10 ml 3.txt"
    
    try:
        # You can adjust the time interval and which plots to show here
        plot_emg_data(
            file_path,
            plot_time_interval=(1, -0.5),  # (start_time, end_time) in seconds

            show_fft_bandpass=False,         # Show FFT with bandpass filter only
            show_fft_bandpass_notch=True,   # Show FFT with bandpass + notch filter
            fft_in_db=True,                 # Set to True for dB, False for linear
            show_channel_plots=True,        # Show channel time series plots
            
            apply_bandpass=False,
            apply_notch=False  # Set to False if you do NOT want to apply the notch filter
        )
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please update the file_path variable with your actual LabChart file path.")
    except Exception as e:
        print(f"Error: {str(e)}")