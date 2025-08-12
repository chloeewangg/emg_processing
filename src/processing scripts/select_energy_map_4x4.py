'''
This script is used to visualize the EMG muscle activation map for selected channels of a 4x4 grid of electrodes.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import re
import os

# ======= MAIN CONFIGURATION (EDIT THESE VALUES) =======
FILE_PATH = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\temporally aligned and averaged\imu aligned\yogurt 20 ml edited temporal average.txt"  
WINDOW_SIZE = 30         # Number of data points per frame
PLOT_START_FRAME = 0     # Start frame index for plotting
PLOT_END_FRAME = None    # End frame index for plotting (None for all)
normalize_by_max = True  # Set to True to normalize by max in plot interval
MAX_AMPLITUDE = 10        # Maximum EMG amplitude in mV for color scaling
AUTO_SET_MAX_AMPLITUDE = True  # Set to True to auto-set max amplitude from data
# List of channels to display (1-16). Unselected channels will appear white.
# Example: SELECTED_CHANNELS = [1, 2, 3, 4] to show only channels 1-4
SELECTED_CHANNELS = [1, 2, 3, 4]  # Test with only first 4 channels
# =====================================================

class EMGMuscleActivationMap:
    def __init__(self, file_path, window_size=25, max_amplitude=10,
                 normalize_interval_max=False, plot_start_frame=None, plot_end_frame=None, normalize_by_max=False, auto_set_max_amplitude=True, selected_channels=None):
        """
        Initialize the EMG muscle activation map visualization.
        Args:
            file_path (str): Path to the LabChart text file
            window_size (int): Number of data points per frame
            max_amplitude (float): Maximum EMG amplitude in mV for color scaling
            normalize_interval_max (bool): Whether to normalize by max in plot interval
            plot_start_frame (int): Start frame index for plotting (for normalization)
            plot_end_frame (int): End frame index for plotting (for normalization)
            normalize_by_max (bool): Whether to normalize by max in plot interval
            auto_set_max_amplitude (bool): Whether to auto-set max amplitude from data
            selected_channels (list): List of channel numbers to display (1-16)
        """
        self.file_path = file_path
        self.window_size = window_size
        self.max_amplitude = max_amplitude
        self.normalize_interval_max = normalize_interval_max
        self.plot_start_frame = plot_start_frame
        self.plot_end_frame = plot_end_frame
        self.normalize_by_max = normalize_by_max
        self.auto_set_max_amplitude = auto_set_max_amplitude
        self.selected_channels = selected_channels if selected_channels is not None else list(range(1, 17))
        self.data = None
        self.emg_channels = None
        
        print(f"Selected channels: {self.selected_channels}")
        print(f"Unselected channels will appear white")
        
        # Define the electrode positions in a 4x4 grid as per the mapping:
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
        self.all_indices = []
        
        # Load the data
        self.load_data()
        # Auto-set max_amplitude if not normalizing and option is enabled
        if not self.normalize_by_max and self.auto_set_max_amplitude:
            if self.emg_channels is not None:
                max_val = np.max(self.emg_channels[:, :16])
                self.max_amplitude = 0.6 * max_val
                print(f"Auto-set max_amplitude to {self.max_amplitude:.6f} (0.6 * peak of first 16 channels)")
    
    def load_data(self):
        """
        Load EMG data from a text file with no header row and 16 columns of data.
        The data is expected to be comma-separated.
        """
        print(f"Loading data from {self.file_path}...")
        try:
            self.emg_channels = np.loadtxt(self.file_path, delimiter=',')
            num_samples = self.emg_channels.shape[0]
            print(f"Loaded {num_samples} data points with {self.emg_channels.shape[1]} channels")
            self.data = self.emg_channels
            samples_per_window = self.window_size
            self.total_frames = num_samples // samples_per_window
            print(f"Total frames: {self.total_frames} (at {self.window_size} data points per frame)")
            print("Pre-computing all grid frames...")
            for frame_idx in range(self.total_frames):
                data_idx = min(frame_idx * samples_per_window, num_samples - 1)
                grid = self.compute_grid(data_idx)
                self.all_grids.append(grid.copy())
                self.all_indices.append(data_idx)
            print(f"Pre-computed {len(self.all_grids)} frames")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def compute_grid(self, data_idx):
        """
        Compute the grid with EMG values for a specific data index.
        Only selected channels will show data; unselected channels will be white.
        Args:
            data_idx (int): Data index in the data array
        """
        if self.data is None or data_idx >= self.data.shape[0]:
            return np.zeros((4, 4))
        
        # Initialize grid with NaN (will appear white)
        grid = np.full((4, 4), np.nan)
        
        # Fill in selected channels with their values
        for channel in range(self.emg_channels.shape[1]):
            channel_num = channel + 1
            if channel_num in self.electrode_positions:
                row, col = self.electrode_positions[channel_num]
                if channel_num in self.selected_channels:
                    grid[row, col] = self.emg_channels[data_idx, channel]
        
        # Apply Gaussian smoothing only to non-NaN values (colored channels)
        # First, create a copy with zeros instead of NaN for smoothing
        grid_for_smoothing = grid.copy()
        grid_for_smoothing[np.isnan(grid_for_smoothing)] = 0
        
        # Apply Gaussian smoothing
        sigma = 0.7  # Adjust for smoothing strength
        smoothed_grid = gaussian_filter(grid_for_smoothing, sigma=sigma)
        
        # Restore NaN values for unselected channels (will appear white)
        smoothed_grid[np.isnan(grid)] = np.nan
        
        # Create masked array
        masked_grid = np.ma.masked_invalid(smoothed_grid)
        return masked_grid
    
    def create_visualization(self, plot_start_frame=None, plot_end_frame=None, normalize_by_max=False):
        """
        Create the visualization with a grid of frames. Plots frames within the specified frame window if provided.
        If normalize_by_max is True, data is normalized by max in plot interval and color scale is set to [0, 1].
        """
        if self.data is None:
            print("Error: No data loaded.")
            return
        if normalize_by_max:
            self.normalize_channels_by_max()
            self.all_grids = []
            self.all_indices = []
            samples_per_window = self.window_size
            num_samples = self.data.shape[0]
            self.total_frames = num_samples // samples_per_window
            for frame_idx in range(self.total_frames):
                data_idx = min(frame_idx * samples_per_window, num_samples - 1)
                grid = self.compute_grid(data_idx)
                self.all_grids.append(grid.copy())
                self.all_indices.append(data_idx)
            self.max_amplitude = 1
        if not self.all_grids:
            print("Error: No frames computed.")
            return
        # Filter frames by frame window if specified
        if plot_start_frame is not None or plot_end_frame is not None:
            start = plot_start_frame if plot_start_frame is not None else 0
            end = plot_end_frame if plot_end_frame is not None else len(self.all_grids)
            grids_to_plot = self.all_grids[start:end]
            indices_to_plot = self.all_indices[start:end]
        else:
            grids_to_plot = self.all_grids
            indices_to_plot = self.all_indices
        cmap_with_white = self.cmap.copy()
        cmap_with_white.set_bad(color='white')
        num_frames = len(grids_to_plot)
        num_cols = int(np.ceil(np.sqrt(num_frames)))
        num_rows = int(np.ceil(num_frames / num_cols))
        fig_width = num_cols * 2.2
        fig_height = num_rows * 2.2
        self.fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        axes = axes.flatten()
        file_name = os.path.basename(self.file_path)
        name_without_extension = file_name.rsplit('.', 1)[0]
        match = re.search(r'\s(\d+)$', name_without_extension)
        if match:
            number_str = match.group(1)
            base_name = name_without_extension[:match.start()]
            formatted_name = base_name.capitalize()
            plot_title = f'{formatted_name} ({number_str})'
        else:
            plot_title = name_without_extension.capitalize()
        self.fig.suptitle(plot_title, fontsize=10, y=0.98)
        for idx, (grid, data_idx) in enumerate(zip(grids_to_plot, indices_to_plot)):
            ax = axes[idx]
            vmax = 1 if normalize_by_max else self.max_amplitude
            
            # Debug: Print grid info for first frame
            if idx == 0:
                print(f"First grid shape: {grid.shape}")
                print(f"First grid type: {type(grid)}")
                if hasattr(grid, 'mask'):
                    print(f"Masked values: {np.sum(grid.mask)} out of {grid.size}")
                    print(f"Non-masked values: {np.sum(~grid.mask)} out of {grid.size}")
                print(f"Selected channels: {self.selected_channels}")
            
            # Use masked array properly
            img = ax.imshow(grid, cmap=cmap_with_white, vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Index {data_idx}", fontsize=8)
        for idx in range(num_frames, len(axes)):
            axes[idx].set_visible(False)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        plt.subplots_adjust(wspace=0.4, hspace=0.6, right=0.88)
        divider = make_axes_locatable(axes[0])
        cax = self.fig.add_axes([0.92, 0.30, 0.015, 0.4])
        self.fig.colorbar(img, cax=cax, label='EMG Amplitude (mV)')
        plt.show()
    
    def normalize_channels_by_max(self):
        """
        Normalize each EMG channel by its maximum value within the selected frame interval.
        Updates self.emg_channels in-place so that each channel's maximum in this interval is 1 (unless the channel is all zeros).
        If the frame range is not set or results in no data, prints a warning and does not modify the data.
        """
        if self.emg_channels is None:
            print("No EMG data loaded.")
            return
        start = self.plot_start_frame if self.plot_start_frame is not None else 0
        end = self.plot_end_frame if self.plot_end_frame is not None else self.emg_channels.shape[0]
        if start >= end or end > self.emg_channels.shape[0]:
            print(f"Invalid frame interval {start} to {end}. No normalization applied.")
            return
        interval_data = self.emg_channels[start:end]
        max_vals = np.max(interval_data, axis=0, keepdims=True)
        for idx, val in enumerate(max_vals.flatten(), 1):
            print(f"Channel {idx} max in interval: {val:.6f}")
        max_vals[max_vals == 0] = 1
        self.emg_channels = self.emg_channels / max_vals
        print(f"Channels normalized by their maximum value in the interval frames {start} to {end}.")

def main():
    emg_map = EMGMuscleActivationMap(
        file_path=FILE_PATH,
        window_size=WINDOW_SIZE,
        max_amplitude=MAX_AMPLITUDE,
        plot_start_frame=PLOT_START_FRAME,
        plot_end_frame=PLOT_END_FRAME,
        normalize_by_max=normalize_by_max,
        auto_set_max_amplitude=AUTO_SET_MAX_AMPLITUDE,
        selected_channels=SELECTED_CHANNELS
    )
    emg_map.create_visualization(plot_start_frame=PLOT_START_FRAME, plot_end_frame=PLOT_END_FRAME, normalize_by_max=normalize_by_max)

if __name__ == '__main__':
    main()