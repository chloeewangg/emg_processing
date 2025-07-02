'''
This script is used to visualize the EMG muscle activation map for multiple files.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
import os

# ======= MAIN CONFIGURATION (EDIT THESE VALUES) =======
FILE_PATHS = [
    r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\temporally aligned and averaged\yogurt 20 ml edited temporal average.txt",
    r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\temporally aligned and averaged\apple 20 ml averaged.txt", 
    r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\temporally aligned and averaged\water 20 ml averaged.txt"
]
WINDOW_SIZE = 30        # Number of data points per frame (same across all plots)
# =====================================================

class EMGMuscleActivationMap:
    def __init__(self, file_path, window_size=25):
        """
        Initialize the EMG muscle activation map visualization.
        Args:
            file_path (str): Path to the LabChart text file
            window_size (int): Number of data points per frame
        """
        self.file_path = file_path
        self.window_size = window_size
        self.data = None
        self.emg_channels = None
        
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
        Args:
            data_idx (int): Data index in the data array
        """
        if self.data is None or data_idx >= self.data.shape[0]:
            return np.zeros((4, 4))
        grid = np.zeros((4, 4))
        for channel in range(self.emg_channels.shape[1]):
            channel_num = channel + 1  # Data columns are 0-based, channel numbers are 1-based
            if channel_num in self.electrode_positions:
                row, col = self.electrode_positions[channel_num]
                grid[row, col] = self.emg_channels[data_idx, channel]
        return grid
    
    def normalize_channels_by_max(self):
        """
        Normalize each EMG channel by its maximum value across the entire dataset.
        Updates self.emg_channels in-place so that each channel's maximum is 1 (unless the channel is all zeros).
        """
        if self.emg_channels is None:
            print("No EMG data loaded.")
            return
        max_vals = np.max(self.emg_channels, axis=0, keepdims=True)
        for idx, val in enumerate(max_vals.flatten(), 1):
            print(f"Channel {idx} max: {val:.6f}")
        max_vals[max_vals == 0] = 1
        self.emg_channels = self.emg_channels / max_vals
        print(f"Channels normalized by their maximum value.")

class SwallowComparisonHeatMap:
    def __init__(self, file_paths, window_size=25):
        """
        Initialize the swallow comparison heat map visualization for multiple files.
        Args:
            file_paths (list): List of paths to the LabChart text files
            window_size (int): Number of data points per frame (same across all plots)
        """
        self.file_paths = file_paths
        self.window_size = window_size
        self.emg_maps = []
        
        # First, load all data to calculate global maximums
        all_data = []
        for file_path in file_paths:
            emg_map = EMGMuscleActivationMap(file_path, window_size)
            if emg_map.data is not None:
                all_data.append(emg_map.emg_channels)
                self.emg_maps.append(emg_map)
            else:
                print(f"Warning: Could not load data from {file_path}")
        
        if not all_data:
            print("Error: No data loaded from any files.")
            return
        
        # Calculate global maximums across all files for each channel
        global_max_vals = np.maximum.reduce([np.max(data, axis=0) for data in all_data])
        print("Global maximum values for each channel:")
        for idx, val in enumerate(global_max_vals, 1):
            print(f"Channel {idx} global max: {val:.6f}")
        
        # Normalize each file using the global maximums
        for emg_map in self.emg_maps:
            # Normalize using global maximums
            for channel in range(emg_map.emg_channels.shape[1]):
                if global_max_vals[channel] > 0:
                    emg_map.emg_channels[:, channel] = emg_map.emg_channels[:, channel] / global_max_vals[channel]
            
            # Recompute grids after normalization
            emg_map.all_grids = []
            emg_map.all_indices = []
            samples_per_window = emg_map.window_size
            num_samples = emg_map.data.shape[0]
            emg_map.total_frames = num_samples // samples_per_window
            for frame_idx in range(emg_map.total_frames):
                data_idx = min(frame_idx * samples_per_window, num_samples - 1)
                grid = emg_map.compute_grid(data_idx)
                emg_map.all_grids.append(grid.copy())
                emg_map.all_indices.append(data_idx)
    
    def create_visualization(self):
        """
        Create separate heat map visualizations for each file.
        """
        if not self.emg_maps:
            print("Error: No data loaded.")
            return
        
        # Create custom colormap
        colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]  # Blue -> Cyan -> Yellow -> Red
        cmap = LinearSegmentedColormap.from_list('emg_cmap', colors, N=256)
        cmap_with_white = cmap.copy()
        cmap_with_white.set_bad(color='white')
        
        # Plot each file separately
        for emg_map in self.emg_maps:
            # Get file name for title
            file_name = os.path.basename(emg_map.file_path)
            name_without_extension = file_name.rsplit('.', 1)[0]
            match = re.search(r'\s(\d+)$', name_without_extension)
            if match:
                number_str = match.group(1)
                base_name = name_without_extension[:match.start()]
                formatted_name = base_name.capitalize()
                plot_title = f'{formatted_name} ({number_str})'
            else:
                plot_title = name_without_extension.capitalize()
            
            # Get all grids for this file
            grids_to_plot = emg_map.all_grids
            indices_to_plot = emg_map.all_indices
            
            # Calculate grid dimensions
            num_frames = len(grids_to_plot)
            num_cols = int(np.ceil(np.sqrt(num_frames)))
            num_rows = int(np.ceil(num_frames / num_cols))
            
            # Create subplot grid for this file
            sub_fig, sub_axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
            if num_frames == 1:
                sub_axes = [sub_axes]
            elif num_rows == 1:
                sub_axes = sub_axes.reshape(1, -1)
            elif num_cols == 1:
                sub_axes = sub_axes.reshape(-1, 1)
            else:
                sub_axes = sub_axes.flatten()
            
            # Plot each frame
            for idx, (grid, data_idx) in enumerate(zip(grids_to_plot, indices_to_plot)):
                if idx < len(sub_axes):
                    sub_ax = sub_axes[idx]
                    img = sub_ax.imshow(grid, cmap=cmap_with_white, vmin=0, vmax=1, interpolation='gaussian')
                    sub_ax.set_xticks([])
                    sub_ax.set_yticks([])
                    sub_ax.set_title(f"Frame {idx+1}", fontsize=8)
            
            # Hide unused subplots
            for idx in range(num_frames, len(sub_axes)):
                sub_axes[idx].set_visible(False)
            
            # Set title for this file
            sub_fig.suptitle(plot_title, fontsize=14, y=0.95)
            
            # Add colorbar
            plt.subplots_adjust(wspace=0.3, hspace=0.4, right=0.85)
            cax = sub_fig.add_axes([0.87, 0.15, 0.02, 0.7])
            sub_fig.colorbar(img, cax=cax, label='Normalized EMG')
            
            # Display this file's plot
            plt.show()

def main():
    comparison_map = SwallowComparisonHeatMap(
        file_paths=FILE_PATHS,
        window_size=WINDOW_SIZE
    )
    comparison_map.create_visualization()

if __name__ == '__main__':
    main()