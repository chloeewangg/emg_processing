'''
This script is used to visualize the ICA components of an EMG signal.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from matplotlib.colors import LinearSegmentedColormap

def load_labchart_data(file_path):
    """
    Load data from a LabChart text file (already preprocessed).
    Assumes first column is time, rest are channels.
    """
    data = pd.read_csv(file_path, sep='\t', header=0)
    return data

def get_time_mask(data, start_time, end_time):
    time = data.iloc[:, 0].values
    return (time >= start_time) & (time <= end_time)

def plot_original_data(data, channels_to_analyze, channel_names=None, start_time=None, end_time=None):
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in channels_to_analyze]
    time = data.iloc[:, 0].values
    if start_time is not None and end_time is not None:
        mask = get_time_mask(data, start_time, end_time)
        time = time[mask]
    else:
        mask = slice(None)
    plt.figure(figsize=(15, 2*len(channels_to_analyze)))
    for i, (channel, name) in enumerate(zip(channels_to_analyze, channel_names)):
        plt.subplot(len(channels_to_analyze), 1, i+1)
        plt.plot(time, data.iloc[:, channel].values[mask])
        plt.title(f'Original {name}', fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.75)
    plt.show()

def plot_ica_components(components, time=None):
    plt.figure(figsize=(15, 2*components.shape[0]))
    for i in range(components.shape[0]):
        plt.subplot(components.shape[0], 1, i+1)
        if time is not None:
            plt.plot(time, components[i])
        else:
            plt.plot(components[i])
        plt.title(f'ICA Component {i+1}', fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.75)
    plt.show()

def plot_mixing_matrix_heatmaps(mixing_matrix, channel_names):
    # Channel-to-grid mapping (row, col):
    channel_grid_map = {
        2: (2, 0),  # lower left
        3: (2, 2),  # lower right
        4: (1, 0),  # left middle
        5: (1, 2),  # right middle
        6: (0, 0),  # top left
        7: (0, 2),  # top right
        8: (0, 1),  # top middle
        9: (2, 1),  # bottom middle
    }
    # Custom diverging colormap: white at 0, blue positive, red negative
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # blue, white, red
    cmap = LinearSegmentedColormap.from_list('custom_bwr', colors, N=256)
    n_components = mixing_matrix.shape[1]
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = axes.flatten()
    for comp_idx in range(n_rows * n_cols):
        ax = axes[comp_idx]
        if comp_idx < n_components:
            grid = np.full((3, 3), np.nan)
            for ch_idx in range(mixing_matrix.shape[0]):
                ch_num = int(channel_names[ch_idx].split()[-1])
                if ch_num in channel_grid_map:
                    row, col = channel_grid_map[ch_num]
                    grid[row, col] = mixing_matrix[ch_idx, comp_idx]
            grid[1, 1] = np.nan  # center blank
            vmax = np.nanmax(np.abs(grid))
            im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(f"Mixing: ICA {comp_idx+1}", fontsize=8)
            # Label each non-blank square with its value in scientific notation
            for i in range(3):
                for j in range(3):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.2e}", ha='center', va='center', color='black', fontsize=6)
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(left=0.05, right=0.85)
    if n_components > 0:
        cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Weight')
    plt.show()

def main():
    # Path to your preprocessed LabChart EMG text file
    file_path = "C:/Users/chloe/OneDrive/Desktop/05_08_25 emg/bandpass 70_110 and notch 60/dry swallow 3.txt"

    # Choose which channels to analyze (0-indexed)
    channels_to_analyze = [1, 2, 3, 4, 5, 6, 7, 8]  # Adjust as needed
    channel_names = [f"Channel {i+1}" for i in channels_to_analyze]

    # Set time interval for plotting (in seconds)
    start_time = 1.5      # Adjust as needed
    end_time = 3.5       # Adjust as needed

    # Load the data
    print("Loading data...")
    data = load_labchart_data(file_path)
    print(f"Data loaded with {data.shape[1]-1} channels and {data.shape[0]} samples")

    # Plot original data
    print("\nPlotting original data...")
    plot_original_data(data, channels_to_analyze, channel_names, start_time, end_time)

    # Prepare data for ICA (only use selected time interval)
    mask = get_time_mask(data, start_time, end_time)
    emg_data = data.iloc[:, channels_to_analyze].values[mask].T
    time = data.iloc[:, 0].values[mask]

    # Perform ICA
    print("\nPerforming ICA...")
    ica = FastICA(n_components=emg_data.shape[0], random_state=42)
    components = ica.fit_transform(emg_data.T).T
    mixing_matrix = ica.mixing_  # shape: (n_channels, n_components)

    # Display mixing matrix weights for each component and channel
    print("\nMixing matrix (weights of each component for each channel):")
    print("Rows: Channels, Columns: ICA Components")
    print(pd.DataFrame(mixing_matrix, index=channel_names, columns=[f"ICA {i+1}" for i in range(mixing_matrix.shape[1])]))

    # Plot ICA components
    print("\nPlotting ICA components...")
    plot_ica_components(components, time)

    # Plot mixing matrix as heat maps
    print("\nPlotting mixing matrix heat maps...")
    plot_mixing_matrix_heatmaps(mixing_matrix, channel_names)

    print("\nICA analysis completed!")

if __name__ == "__main__":
    main()