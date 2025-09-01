'''
This script is used to detect contractions in an EMG signal.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ============================== CONFIGURATION ==============================
input_file = r"C:\Users\chloe\OneDrive\Desktop\swallow EMG\data\07_18_25\original\dry swallow\1.txt"
sampling_rate = 500  
rms_window_sec = 0.1  

std_threshold_start = 1  
std_threshold_end = 0.25  

num_channels = 16  
file_delimiter = ','  
has_time_column = False  

baseline_file = r"C:\Users\chloe\OneDrive\Desktop\swallow EMG\data\07_18_25\other\baseline noise\1.txt"
baseline_sampling_rate = 500  
baseline_start = 2  
baseline_end = 12     

output_folder = r"C:\Users\chloe\OneDrive\Desktop\swallow EMG\data\07_18_25\extracted signals\dry swallow"

# ===========================================================================

def find_data_start_row(filepath, delimiter, num_columns):
    """
    Returns the row index (0-based) where the first fully numeric data row starts.

    args:
        filepath (str): path to the file
        delimiter (str): delimiter used in the file
        num_columns (int): number of columns in the file
    returns:
        int: row index (0-based) where the first fully numeric data row starts
    """
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            parts = re.split(delimiter if delimiter != '\\t' else '\t', line.strip())
            if len(parts) >= num_columns:
                try:
                    [float(parts[i]) for i in range(num_columns)]
                    return idx
                except Exception:
                    continue
    return 0  # fallback if not found

def rms_contraction_detector(input_file, sampling_rate, rms_window_sec, baseline_start, baseline_end, baseline_file=None, baseline_sampling_rate=None, num_channels=16, file_delimiter=',', has_time_column=False):
    """
    Interactive contraction detection:
    1. Plots RMS for all channels (no detection).
    2. Prompts user for start and end time of the search range.
    3. Performs contraction detection only in that range.
    4. Plots with detected contraction start/end lines as before.

    args:
        input_file (str): path to the input file
        sampling_rate (float): sampling rate in Hz
        rms_window_sec (float): window size for RMS calculation
        baseline_start (float): start time of the baseline noise window
        baseline_end (float): end time of the baseline noise window
        baseline_file (str): path to the baseline noise file
        baseline_sampling_rate (float): sampling rate of the baseline noise file
        num_channels (int): number of channels in the data
        file_delimiter (str): delimiter used in the file
        has_time_column (bool): whether the file has a time column
    returns:
        None
    """
    # 1. Detect and skip header rows for input file
    total_columns = num_channels + 1 if has_time_column else num_channels
    data_start_row = find_data_start_row(input_file, file_delimiter, total_columns)
    try:
        data_raw = pd.read_csv(input_file, sep=file_delimiter, header=None, usecols=range(total_columns), skiprows=data_start_row)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    if has_time_column:
        time = data_raw.iloc[:, 0].values
        data = data_raw.iloc[:, 1:1+num_channels]
    else:
        data = data_raw.iloc[:, :num_channels]
        n_samples = len(data)
        time = np.arange(n_samples) / sampling_rate

    # 2. Rectify the data
    rectified_data = data.abs()
    n_samples = len(rectified_data)

    # 3. Get baseline noise window (indices), possibly from a separate file
    if baseline_file is not None:
        baseline_total_columns = num_channels + 1 if has_time_column else num_channels
        baseline_start_row = find_data_start_row(baseline_file, file_delimiter, baseline_total_columns)
        try:
            baseline_raw = pd.read_csv(baseline_file, sep=file_delimiter, header=None, usecols=range(baseline_total_columns), skiprows=baseline_start_row)
        except FileNotFoundError:
            print(f"Error: Baseline file not found at {baseline_file}")
            return
        except Exception as e:
            print(f"Error loading baseline file: {e}")
            return
        baseline_sr = baseline_sampling_rate if baseline_sampling_rate is not None else sampling_rate
        if has_time_column:
            baseline_time = baseline_raw.iloc[:, 0].values
            baseline_data_full = baseline_raw.iloc[:, 1:1+num_channels]
        else:
            baseline_data_full = baseline_raw.iloc[:, :num_channels]
            baseline_time = np.arange(len(baseline_data_full)) / baseline_sr
        baseline_mask = (baseline_time >= baseline_start) & (baseline_time < baseline_end)
        baseline_data = baseline_data_full[baseline_mask]
    else:
        baseline_mask = (time >= baseline_start) & (time < baseline_end)
        baseline_data = rectified_data[baseline_mask]

    # 4. Calculate RMS of baseline noise for each channel
    baseline_rms = baseline_data.pow(2).mean().pow(0.5)
    baseline_std = baseline_data.pow(2).std().pow(0.5)

    # 5. Exclude the first and last 0.4 seconds for plotting
    trim_samples = int(sampling_rate)
    plot_data = rectified_data.iloc[trim_samples:-trim_samples].reset_index(drop=True)
    plot_time = time[trim_samples:-trim_samples]
    window_samples = int(rms_window_sec * sampling_rate)
    if window_samples < 1:
        window_samples = 1
    plot_rms = plot_data.pow(2).rolling(window=window_samples, center=True).mean().pow(0.5)

    # 6. Plot RMS for all channels (no detection)
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 18), sharex=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ch, ax in enumerate(axes):
        if ch < plot_rms.shape[1]:
            ax.plot(plot_time, plot_rms.iloc[:, ch], color='orange', label='RMS')
            ax.set_ylabel(f'Ch {ch+1}', rotation=0, ha='right', va='center', labelpad=10)
            ax.autoscale(enable=True, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='y', labelsize=6)
            if ch < len(axes) - 1:
                ax.set_xticks([])
            else:
                xticks = np.arange(np.ceil(plot_time[0]), np.floor(plot_time[-1]) + 1, 1)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(int(tick)) for tick in xticks])
    plt.show()

    # 7. Prompt user for start and end time of the search range
    while True:
        try:
            user_start = float(input("Enter the START time (in seconds) of the range to search for contraction: "))
            user_end = float(input("Enter the END time (in seconds) of the range to search for contraction: "))
            if user_start < user_end and plot_time[0] <= user_start < user_end <= plot_time[-1]:
                break
            else:
                print(f"Please enter valid times between {plot_time[0]:.2f} and {plot_time[-1]:.2f}, with start < end.")
        except Exception:
            print("Invalid input. Please enter numeric values.")

    # 8. Prompt user for additional channels to exclude
    while True:
        exclude_input = input(f"Enter additional channel numbers (1-{num_channels}) to exclude from detection, separated by commas (or leave blank for none): ").strip()
        if not exclude_input:
            user_excluded_channels = []
            break
        try:
            user_excluded_channels = [int(ch.strip())-1 for ch in exclude_input.split(',') if ch.strip()]
            if all(0 <= ch < num_channels for ch in user_excluded_channels):
                break
            else:
                print(f"Please enter valid channel numbers between 1 and {num_channels}.")
        except Exception:
            print("Invalid input. Please enter numbers separated by commas.")

    # 9. Restrict search to user-specified window
    search_mask = (plot_time >= user_start) & (plot_time <= user_end)
    search_data = plot_data[search_mask].reset_index(drop=True)
    search_time = plot_time[search_mask]
    rolling_rms = search_data.pow(2).rolling(window=window_samples, center=True).mean().pow(0.5)

    # 10. Detect NaN channels and combine with user-excluded channels
    # Check for NaN channels in the search data
    nan_channels = []
    for ch in range(search_data.shape[1]):
        if search_data.iloc[:, ch].isna().any():
            nan_channels.append(ch)
    
    if nan_channels:
        print(f"Automatically detected NaN channels: {[ch+1 for ch in nan_channels]}")
    
    # Combine NaN channels and user-excluded channels
    excluded_channels = list(set(nan_channels + user_excluded_channels))
    if excluded_channels:
        print(f"Total excluded channels: {[ch+1 for ch in excluded_channels]}")

    # 11. Detect contraction start and end for each channel in the user window
    contraction_points = []
    for ch in range(rolling_rms.shape[1]):
        if ch in excluded_channels:
            contraction_points.append((None, None))
            continue
        rms_ch = rolling_rms.iloc[:, ch]
        base = baseline_rms.iloc[ch]
        std = baseline_std.iloc[ch]
        above_thresh = rms_ch > (base + std_threshold_start * std)
        below_thresh = rms_ch < (base + std_threshold_end * std)
        start_idx = None
        end_idx = None
        for i in range(len(rms_ch)):
            if above_thresh.iloc[i]:
                start_idx = i
                break
        if start_idx is not None:
            for i in range(start_idx, len(rms_ch)):
                if below_thresh.iloc[i]:
                    end_idx = i
                    break
        start_time = search_time[start_idx] if start_idx is not None else None
        end_time = search_time[end_idx] if end_idx is not None else None
        contraction_points.append((start_time, end_time))

    # Calculate number of included channels and agreement threshold
    num_included = num_channels - len(excluded_channels)
    # Adjust agreement threshold: if there are NaN channels, reduce the required agreement
    # Original: 9/16 of channels must agree, but now we adjust based on available channels
    if num_included > 0:
        # Calculate what 9/16 of total channels would be, then adjust for available channels
        original_threshold = int(np.ceil(num_channels * 9 / 16))
        # Subtract the number of NaN channels from the original threshold
        agreement_threshold = max(1, original_threshold - len(nan_channels))
    else:
        agreement_threshold = 1

    # 12. Robust multi-channel contraction start detection (exclude excluded channels)
    all_starts = [pt[0] for i, pt in enumerate(contraction_points) if pt[0] is not None and i not in excluded_channels]
    all_ends = [pt[1] for i, pt in enumerate(contraction_points) if pt[1] is not None and i not in excluded_channels]
    overall_start = None
    if all_starts:
        sorted_starts = sorted(all_starts)
        for t in sorted_starts:
            count = sum(abs(t - other) <= 0.2 for other in all_starts)
            if count >= agreement_threshold:
                overall_start = t
                break
    overall_end = None
    if all_ends:
        # Sort all end times (descending for latest cluster)
        sorted_ends = sorted(all_ends, reverse=True)
        for t in sorted_ends:
            count = sum(abs(t - other) <= 0.2 for other in all_ends)
            if count >= agreement_threshold:
                overall_end = t
                break

    # 13. Plot with detected contraction start/end lines
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 18), sharex=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ch, ax in enumerate(axes):
        if ch < plot_rms.shape[1]:
            ax.plot(plot_time, plot_rms.iloc[:, ch], color='orange', label='RMS')
            # Plot each channel's detected start/end times (temporary visualization)
            ch_start, ch_end = contraction_points[ch]
            if ch_start is not None and (plot_time[0] <= ch_start <= plot_time[-1]):
                ax.axvline(ch_start, color='#90ee90', linestyle='--', linewidth=1, label='Ch Start' if ch == 0 else None)  # light green
            if ch_end is not None and (plot_time[0] <= ch_end <= plot_time[-1]):
                ax.axvline(ch_end, color='#ffb6c1', linestyle='--', linewidth=1, label='Ch End' if ch == 0 else None)    # light pink
            # Plot overall start/end
            if overall_start is not None and (plot_time[0] <= overall_start <= plot_time[-1]):
                ax.axvline(overall_start, color='g', linestyle='--', label='Start' if ch == 0 else None)
            if overall_end is not None and (plot_time[0] <= overall_end <= plot_time[-1]):
                ax.axvline(overall_end, color='r', linestyle='--', label='End' if ch == 0 else None)
            ax.set_ylabel(f'Ch {ch+1}', rotation=0, ha='right', va='center', labelpad=10)
            ax.autoscale(enable=True, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='y', labelsize=6)
            if ch < len(axes) - 1:
                ax.set_xticks([])
            else:
                xticks = np.arange(np.ceil(plot_time[0]), np.floor(plot_time[-1]) + 1, 1)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(int(tick)) for tick in xticks])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.suptitle(f"RMS with Contraction Detection ({num_channels} Channels)")
    plt.show()

    # 14. Print only the overall contraction start and end
    if overall_start is not None and overall_end is not None:
        print(f"Overall contraction: start = {overall_start:.3f}s, end = {overall_end:.3f}s")

        # Save the segment from (start-0.2*duration) to (end+0.2*duration)
        data_start_row = find_data_start_row(input_file, file_delimiter, total_columns)
        try:
            all_data_raw = pd.read_csv(input_file, sep=file_delimiter, header=None, skiprows=data_start_row)
        except Exception as e:
            print(f"Error loading full data for saving: {e}")
            return
        if has_time_column:
            all_data = all_data_raw.iloc[:, 1:1+num_channels]
        else:
            all_data = all_data_raw.iloc[:, :num_channels]
        n_samples = len(all_data)
        duration = overall_end - overall_start
        margin = 0.2 * duration
        start_save = max(0, int((overall_start - margin) * sampling_rate))
        end_save = min(n_samples, int((overall_end + margin) * sampling_rate))

        # Processed (rectified + RMS) for channels 1-num_channels
        rectified_full = all_data.abs()
        window_samples = int(rms_window_sec * sampling_rate)
        if window_samples < 1:
            window_samples = 1
        rms_smoothed = rectified_full.pow(2).rolling(window=window_samples, center=True).mean().pow(0.5)
        # Get the segment for channels 1-num_channels
        processed_segment = rms_smoothed.iloc[start_save:end_save].reset_index(drop=True)
        # For channels num_channels+1-22, use original data
        if all_data_raw.shape[1] > (num_channels + (1 if has_time_column else 0)):
            unprocessed_segment = all_data_raw.iloc[start_save:end_save, (num_channels + (1 if has_time_column else 0)):22].reset_index(drop=True)
            combined = pd.concat([processed_segment, unprocessed_segment], axis=1)
        else:
            combined = processed_segment
        # Prepare output path
        os.makedirs(output_folder, exist_ok=True)
        base_filename = os.path.basename(input_file)
        output_path = os.path.join(output_folder, base_filename)
        combined.to_csv(output_path, index=False, header=False)
        print(f"Saved detected segment to: {output_path}")
    else:
        print("No contraction detected across all channels.")

def __main__():
    rms_contraction_detector(
        input_file,
        sampling_rate,
        rms_window_sec,
        baseline_start,
        baseline_end,
        baseline_file=baseline_file,
        baseline_sampling_rate=baseline_sampling_rate,
        num_channels=num_channels,
        file_delimiter=file_delimiter,
        has_time_column=has_time_column
    )

if __name__ == "__main__":
    __main__()
