'''
This script is used to visualize the ICA components of an EMG signal.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA # type: ignore
import os

# ============================== CONFIGURATION ==============================
file_path = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\spikes removed\apple 5 ml 1.txt"  
sampling_rate = 500  
num_channels_to_process = 16
max_points_to_plot = 5000  
# ===========================================================================

def load_emg_data(file_path):
    """
    Load data from a text file (no time column, one header row).
    Assumes comma-separated values.
    """
    try:
        # Skips the first row (header)
        data = pd.read_csv(file_path, sep=',', header=0)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def plot_data(data, time, title_prefix='', max_points_to_plot=5000):
    """
    Generic function to plot EMG data channels.

    args:
        data (pandas.DataFrame): DataFrame with EMG data
        time (numpy.ndarray): Time vector
        title_prefix (str): Prefix for the plot title
        max_points_to_plot (int): Maximum number of points to plot
    returns:
        None
    """
    if len(time) > max_points_to_plot:
        step = len(time) // max_points_to_plot
        time = time[::step]
        data = data.iloc[::step, :]
    
    num_channels = data.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 1.5 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]
    
    for i in range(num_channels):
        axes[i].plot(time, data.iloc[:, i])
        axes[i].set_ylabel(f'Ch {i+1}')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].tick_params(axis='x', bottom=False, labelbottom=False)

    axes[-1].tick_params(axis='x', bottom=True, labelbottom=True)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f'{title_prefix} EMG Data ({num_channels} Channels)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_explained_variance(data):
    """
    Performs PCA and plots the cumulative explained variance.
    """
    print("\nPerforming PCA to determine optimal number of components...")
    pca = PCA()
    pca.fit(data)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    # Add a horizontal line at 99.9% variance for reference
    plt.axhline(y=0.999, color='r', linestyle='-', label='99.9% Variance')
    plt.legend()
    
    print("Cumulative variance per number of components:")
    for i, var in enumerate(cumulative_variance):
        print(f"  {i+1} components: {var:.4f}")
        
    plt.show()
    return pca

def plot_ica_components(components, time, max_points_to_plot=5000, sampling_rate=500):
    """
    Plots the separated ICA components. Decimates data if it's too long.
    Smooths each component using RMS smoothing with a 0.2s window.

    args:
        components (numpy.ndarray): ICA components
        time (numpy.ndarray): Time vector
        max_points_to_plot (int): Maximum number of points to plot
        sampling_rate (int): Sampling rate of the EMG data
    returns:
        None
    """
    if len(time) > max_points_to_plot:
        step = len(time) // max_points_to_plot
        time = time[::step]
        components = components[:, ::step]
    
    num_components = components.shape[0]
    fig, axes = plt.subplots(num_components, 1, figsize=(15, 1.5 * num_components), sharex=True)
    if num_components == 1:
        axes = [axes]

    # RMS smoothing window size in samples
    window_size = int(0.23 * sampling_rate)
    if window_size < 1:
        window_size = 1
    def rms_smooth(x, w):
        # Pad to keep same length
        pad = w // 2
        x2 = np.pad(x**2, (pad, pad), mode='edge')
        return np.sqrt(np.convolve(x2, np.ones(w)/w, mode='valid'))

    for i in range(num_components):
        smoothed = rms_smooth(components[i, :], window_size)
        # Ensure time and smoothed have the same length
        min_len = min(len(time), len(smoothed))
        axes[i].plot(time[:min_len], smoothed[:min_len])
        axes[i].set_ylabel(f'{i+1}')

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def main():
    # Load the data
    print("Loading data...")
    data = load_emg_data(file_path)
    if data is None:
        return

    # Select first 16 channels and generate time vector
    emg_data = data.iloc[:, :num_channels_to_process]
    time = np.arange(emg_data.shape[0]) / sampling_rate
    print(f"Data loaded: {emg_data.shape[1]} channels, {len(time):.2f} seconds.")

    # Perform PCA to show explained variance and help user decide on component number
    pca = plot_explained_variance(emg_data)

    # Ask user for number of ICA components
    while True:
        try:
            n_components = int(input("\nEnter the number of ICA components to use: "))
            if n_components < 1 or n_components > num_channels_to_process:
                raise ValueError(f"Please enter a number between 1 and {num_channels_to_process}.")
            break
        except ValueError as e:
            print(f"Invalid input. {e}")

    # Perform ICA on the principal components
    print(f"\nPerforming ICA with {n_components} components on principal components...")
    pca_data = pca.transform(emg_data)[:, :n_components]
    ica = FastICA(n_components=n_components, random_state=42, whiten='unit-variance', max_iter=1000)
    sources = ica.fit_transform(pca_data)

    # Plot ICA components, excluding start and end
    print("\nPlotting ICA components...")
    trim_samples = int(0.4 * sampling_rate)
    if len(time) > 2 * trim_samples:
        plot_time_ica = time[trim_samples:-trim_samples]
        sources_for_plot = sources.T[:, trim_samples:-trim_samples]
    else:
        plot_time_ica = time
        sources_for_plot = sources.T
    plot_ica_components(sources_for_plot, plot_time_ica, max_points_to_plot=max_points_to_plot, sampling_rate=sampling_rate)

    # Compute spatial projection of each IC onto the original 16 channels
    # Use ICA unmixing matrix (ica.mixing_) and PCA projection matrix (pca.components_)
    # Channel contributions: ica.mixing_ (n_components x n_components) @ pca.components_[:n_components, :] (n_components x 16)
    ic_spatial_map = np.dot(ica.mixing_, pca.components_[:n_components, :])

    # Channel mapping for 4x4 grid (1-based channel index to (row, col))
    channel_grid = {
        13: (0, 0), 9: (0, 1), 4: (0, 2), 8: (0, 3),
        14: (1, 0), 10: (1, 1), 3: (1, 2), 7: (1, 3),
        15: (2, 0), 11: (2, 1), 2: (2, 2), 6: (2, 3),
        16: (3, 0), 12: (3, 1), 1: (3, 2), 5: (3, 3)
    }
    channel_to_grid = [None]*16
    for ch, (row, col) in channel_grid.items():
        channel_to_grid[ch-1] = (row, col)

    n_ICs = ic_spatial_map.shape[0]
    # Make each heatmap smaller (e.g., 2x2 inches per subplot)
    n_cols = int(np.ceil(np.sqrt(n_ICs)))
    n_rows = int(np.ceil(n_ICs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    if n_ICs == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape((n_rows, n_cols))
    for ic_idx in range(n_ICs):
        grid = np.zeros((4, 4))
        for ch in range(16):
            row, col = channel_to_grid[ch]
            grid[row, col] = ic_spatial_map[ic_idx, ch]
        ax = axes[ic_idx // n_cols, ic_idx % n_cols]
        im = ax.imshow(grid, cmap='bwr', aspect='equal')
        ax.set_title(f'IC {ic_idx+1}', fontsize=8)
        ax.set_xlabel('Col', fontsize=7)
        ax.set_ylabel('Row', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.tick_params(labelsize=6)
    # Hide any unused subplots
    for idx in range(n_ICs, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main() 