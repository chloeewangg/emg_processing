'''
This script is used to visualize the EMG muscle activation map of a 4x4 grid of electrodes.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import re
import os

# ============================== CONFIGURATION ==============================
FILE_PATH = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\envelopes\apple 10 ml 1 rms envelope.txt"  
SAMPLING_RATE = 500        
WINDOW_SIZE = 0.05        
MAX_AMPLITUDE = 10       
PLOT_START_TIME = 3.5     
PLOT_END_TIME = 4.9  
show_minutes = False  
normalize_by_max = True  
# ===========================================================================

class EMGMuscleActivationMap:
    def __init__(self, file_path, sampling_rate=1000, window_size=0.1, max_amplitude=0.025,
                 normalize_interval_max=False, plot_start_time=None, plot_end_time=None):
        """
        Initialize the EMG muscle activation map visualization.
        
        args:
            file_path (str): Path to the LabChart text file
            sampling_rate (float): Sampling rate of the EMG data in Hz
            window_size (float): Time window for each frame in seconds
            max_amplitude (float): Maximum EMG amplitude in mV for color scaling
            normalize_interval_max (bool): Whether to normalize by max in plot interval
            plot_start_time (float): Start time for plotting (for normalization)
            plot_end_time (float): End time for plotting (for normalization)
        returns:
            None
        """
        self.file_path = file_path
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.max_amplitude = max_amplitude
        self.normalize_interval_max = normalize_interval_max
        self.plot_start_time = plot_start_time
        self.plot_end_time = plot_end_time
        self.data = None
        self.time = None
        self.emg_channels = None
        
        # Define the electrode positions in a 4x4 grid as per the mapping:
        # 13  9  4  8
        # 14 10  3  7
        # 15 11  2  6
        # 16 12  1  5
        # Channel numbers are 1-based
        self.electrode_positions = {
            13: (0, 0), 9: (0, 1), 4: (0, 2), 8: (0, 3),
            14: (1, 0), 10: (1, 1), 3: (1, 2), 7: (1, 3),
            15: (2, 0), 11: (2, 1), 2: (2, 2), 6: (2, 3),
            16: (3, 0), 12: (3, 1), 1: (3, 2), 5: (3, 3)
        }
        
        # Create custom colormap (similar to the one in the reference image)
        colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]  # Blue -> Cyan -> Yellow -> Red
        self.cmap = LinearSegmentedColormap.from_list('emg_cmap', colors, N=256)
        
        # Create the 4x4 grid for visualization
        self.grid = np.zeros((4, 4))
        
        # Store all grid frames
        self.all_grids = []
        self.all_times = []
        
        # Load the data
        self.load_data()
    
    def load_data(self):
        """
        Load EMG data from a text file with no header row and 16 columns of data.
        The data is expected to be comma-separated. A time column is generated 
        based on the sampling rate.
        """
        print(f"Loading data from {self.file_path}...")
        try:
            # Load data from a comma-separated file with no header.
            # This expects 16 columns of rectified and smoothed EMG data.
            self.emg_channels = np.loadtxt(self.file_path, delimiter=',')

            # Generate the time vector since it's not in the file
            num_samples = self.emg_channels.shape[0]
            self.time = np.arange(num_samples) / self.sampling_rate

            # The data is assumed to be already processed (rectified and smoothed),
            # so taking the absolute value is no longer necessary.

            print(f"Loaded {len(self.time)} data points with {self.emg_channels.shape[1]} channels")
            self.data = self.emg_channels # For compatibility with other parts of the class

            # Calculate total number of frames
            samples_per_window = int(self.window_size * self.sampling_rate)
            self.total_frames = len(self.time) // samples_per_window
            print(f"Total frames: {self.total_frames} (at {self.window_size}s window)")

            # Pre-compute all grid frames
            print("Pre-computing all grid frames...")
            for frame_idx in range(self.total_frames):
                time_idx = min(frame_idx * samples_per_window, len(self.time) - 1)
                grid = self.compute_grid(time_idx)
                self.all_grids.append(grid.copy())
                self.all_times.append(self.time[time_idx])

            print(f"Pre-computed {len(self.all_grids)} frames")

        except Exception as e:
            print(f"Error loading data: {e}")
    
    def compute_grid(self, time_idx):
        """
        Compute the grid with EMG values for a specific time index.
        
        args:
            time_idx (int): Time index in the data array
        returns:
            grid (numpy.ndarray): Grid with EMG values
        """
        if self.data is None or time_idx >= len(self.time):
            return np.zeros((4, 4))
        
        # Reset grid to zeros
        grid = np.zeros((4, 4))
        
        # Fill grid with EMG values
        # The data file may only have 8 channels, so we need to map them to the correct channel numbers
        # We'll assume the data columns are channels 1-8 (i.e., emg_channels[:, 0] is channel 1, etc.)
        # If you have more channels, adjust accordingly
        for channel in range(self.emg_channels.shape[1]):
            channel_num = channel + 1  # Data columns are 0-based, channel numbers are 1-based
            if channel_num in self.electrode_positions:
                row, col = self.electrode_positions[channel_num]
                grid[row, col] = self.emg_channels[time_idx, channel]
        
        return grid
    
    def create_visualization(self, plot_start_time=None, plot_end_time=None, show_minutes=False, normalize_by_max=False):
        """
        Create the interactive visualization with a slider for time navigation.
        Only plots frames within the specified time window if provided.
        If show_minutes is True, time labels are shown as mm:ss.s; otherwise, as seconds.
        If normalize_by_max is True, data is normalized by max in interval and color scale is set to [0, 1].

        args:
            plot_start_time (float): Start time for plotting
            plot_end_time (float): End time for plotting
            show_minutes (bool): Whether to show time labels as mm:ss.s
            normalize_by_max (bool): Whether to normalize by max in plot interval
        returns:
            None
        """
        if self.data is None:
            print("Error: No data loaded.")
            return
        # Normalize and recompute grids if requested
        if normalize_by_max:
            self.normalize_channels_by_max()
            # Recompute all_grids and all_times using normalized data
            self.all_grids = []
            self.all_times = []
            samples_per_window = int(self.window_size * self.sampling_rate)
            self.total_frames = len(self.time) // samples_per_window
            for frame_idx in range(self.total_frames):
                time_idx = min(frame_idx * samples_per_window, len(self.time) - 1)
                grid = self.compute_grid(time_idx)
                self.all_grids.append(grid.copy())
                self.all_times.append(self.time[time_idx])
            self.max_amplitude = 1  # Override max_amplitude for normalized data
        if not self.all_grids:
            print("Error: No frames computed.")
            return
        
        # Filter frames by time window if specified
        if plot_start_time is not None and plot_end_time is not None:
            indices = [i for i, t in enumerate(self.all_times) if plot_start_time <= t <= plot_end_time]
            if not indices:
                print(f"No frames found in the time window {plot_start_time}-{plot_end_time}s.")
                return
            grids_to_plot = [self.all_grids[i] for i in indices]
            times_to_plot = [self.all_times[i] for i in indices]
        else:
            grids_to_plot = self.all_grids
            times_to_plot = self.all_times
        
        # Set the colormap's bad color to white
        cmap_with_white = self.cmap.copy()
        cmap_with_white.set_bad(color='white')
        
        # Set up the figure with subplots
        num_frames = len(grids_to_plot)
        num_cols = int(np.ceil(np.sqrt(num_frames)))  # Square-ish grid
        num_rows = int(np.ceil(num_frames / num_cols))
        
        fig_width = num_cols * 2.2  # 2.2 inches per subplot (smaller)
        fig_height = num_rows * 2.2
        self.fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        axes = axes.flatten()  # Flatten to make indexing easier
        
        # Add file name as a small title
        file_name = os.path.basename(self.file_path)
        # Remove .txt extension
        name_without_extension = file_name.rsplit('.', 1)[0]
        
        # Try to find a number at the end of the string
        match = re.search(r'\s(\d+)$', name_without_extension)
        
        if match:
            # Extract the number and the part before it
            number_str = match.group(1)
            base_name = name_without_extension[:match.start()]
            # Capitalize the first letter of the base name
            formatted_name = base_name.capitalize()
            # Construct the new title
            plot_title = f'{formatted_name} ({number_str})'
        else:
            # If no number is found, just capitalize the whole name without extension
            plot_title = name_without_extension.capitalize()

        self.fig.suptitle(plot_title, fontsize=10, y=0.98) # Adjusted y position
        
        # Plot each frame
        for idx, (grid, time) in enumerate(zip(grids_to_plot, times_to_plot)):
            ax = axes[idx]
            # Set color scale depending on normalization
            vmax = 1 if normalize_by_max else self.max_amplitude
            img = ax.imshow(grid, cmap=cmap_with_white, vmin=0, vmax=vmax, interpolation='gaussian')
            ax.set_xticks([])
            ax.set_yticks([])
            if show_minutes:
                minutes = int(time // 60)
                seconds = time % 60
                ax.set_title(f"{minutes:02d}:{seconds:04.1f}", fontsize=8)
            else:
                ax.set_title(f"{time:.3f}s", fontsize=8)
        
        # Hide empty subplots
        for idx in range(num_frames, len(axes)):
            axes[idx].set_visible(False)
        
        # Add a single colorbar for all plots, placed to the right and not overlapping
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        plt.subplots_adjust(wspace=0.4, hspace=0.6, right=0.88)  # Add space between plots and for colorbar
        divider = make_axes_locatable(axes[0])
        cax = self.fig.add_axes([0.92, 0.30, 0.015, 0.4])  # [left, bottom, width, height] in figure fraction (smaller)
        self.fig.colorbar(img, cax=cax, label='EMG Amplitude (mV)')
        plt.show()
    
    def normalize_channels_by_max(self):
        """
        Normalize each EMG channel by its maximum value within the interval
        [self.plot_start_time, self.plot_end_time].
        Updates self.emg_channels in-place so that each channel's maximum in this interval is 1 (unless the channel is all zeros).
        If the time range is not set or results in no data, prints a warning and does not modify the data.
        """
        if self.emg_channels is None or self.time is None:
            print("No EMG data loaded.")
            return
        if self.plot_start_time is None or self.plot_end_time is None:
            print("plot_start_time and plot_end_time must be set to normalize by interval max.")
            return
        t_start = self.plot_start_time
        t_end = self.plot_end_time
        mask = (self.time >= t_start) & (self.time <= t_end)
        if not np.any(mask):
            print(f"No data in the interval {t_start} to {t_end} seconds. No normalization applied.")
            return
        interval_data = self.emg_channels[mask]
        max_vals = np.max(interval_data, axis=0, keepdims=True)
        # Print max value for each channel
        for idx, val in enumerate(max_vals.flatten(), 1):
            print(f"Channel {idx} max in interval: {val:.6f}")
        max_vals[max_vals == 0] = 1  # Prevent division by zero
        self.emg_channels = self.emg_channels / max_vals
        print(f"Channels normalized by their maximum value in the interval {t_start:.2f} to {t_end:.2f} seconds.")

def main():
    emg_map = EMGMuscleActivationMap(
        file_path=FILE_PATH,
        sampling_rate=SAMPLING_RATE,
        window_size=WINDOW_SIZE,
        max_amplitude=MAX_AMPLITUDE,
        plot_start_time=PLOT_START_TIME,
        plot_end_time=PLOT_END_TIME
    )
    emg_map.create_visualization(plot_start_time=PLOT_START_TIME, plot_end_time=PLOT_END_TIME, show_minutes=show_minutes, normalize_by_max=normalize_by_max)

if __name__ == '__main__':
    main()