'''
This script is used to visualize the EMG muscle activation map of a 3x3 grid of electrodes.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import re
import os

# ============================== CONFIGURATION ==============================
FILE_PATH = "C:/Users/chloe/OneDrive/Desktop/05_08_25 emg/bandpass_and_notch/yogurt 20 ml 2.txt"  
SAMPLING_RATE = 1000        
WINDOW_SIZE = 0.01          
MAX_AMPLITUDE = 1       
PLOT_START_TIME = 6     
PLOT_END_TIME = 7  
# ===========================================================================

class EMGMuscleActivationMap:
    def __init__(self, file_path, sampling_rate=1000, window_size=0.1, max_amplitude=0.025,
                 normalize_interval_max=False, plot_start_time=None, plot_end_time=None):
        """
        Initialize the EMG muscle activation map visualization.
        
        Args:
            file_path (str): Path to the LabChart text file
            sampling_rate (float): Sampling rate of the EMG data in Hz
            window_size (float): Time window for each frame in seconds
            max_amplitude (float): Maximum EMG amplitude in mV for color scaling
            normalize_interval_max (bool): Whether to normalize by max in plot interval
            plot_start_time (float): Start time for plotting (for normalization)
            plot_end_time (float): End time for plotting (for normalization)
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
        
        # Channel mapping:
        # 6 8 7
        # 4   5  
        # 2 9 3
        self.electrode_positions = {
            2: (2, 0),  # Ch 2: lower left (row, col)
            3: (2, 2),  # Ch 3: lower right
            4: (1, 0),  # Ch 4: left middle
            5: (1, 2),  # Ch 5: right middle
            6: (0, 0),  # Ch 6: top left
            7: (0, 2),  # Ch 7: top right
            8: (0, 1),  # Ch 8: top middle
            9: (2, 1)   # Ch 9: bottom middle
        }
        
        # Create custom colormap (similar to the one in the reference image)
        colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]  # Blue -> Cyan -> Yellow -> Red
        self.cmap = LinearSegmentedColormap.from_list('emg_cmap', colors, N=256)
        
        # Create the 3x3 grid for visualization
        self.grid = np.zeros((3, 3))
        
        # Store all grid frames
        self.all_grids = []
        self.all_times = []
        
        # Load the data
        self.load_data()
    
    def load_data(self):
        """
        Load EMG data from a LabChart text file.
        """
        print(f"Loading data from {self.file_path}...")
        
        try:
            # Find data section (skip header)
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
            
            data_start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('0') or '\t' in line:
                    data_start_idx = i
                    break
            
            # Process data lines
            data_lines = lines[data_start_idx:]
            data_array = []
            
            for line in data_lines:
                try:
                    values = line.strip().split('\t')
                    if len(values) >= 9:  # Time + 8 channels
                        data_array.append([float(val) for val in values[:9]])
                except ValueError:
                    continue  # Skip lines that cannot be converted to float
            
            if not data_array:
                print("Error: No valid data found in the file")
                return
                
            # Convert to numpy array
            data_array = np.array(data_array)
            
            # Extract time and EMG channels
            self.time = data_array[:, 0]
            self.emg_channels = data_array[:, 1:9]  # 8 EMG channels
            
            # Preprocess data (take absolute values)
            self.emg_channels = np.abs(self.emg_channels)
            
            # Apply normalization by max in plot interval if selected
            if self.normalize_interval_max and self.plot_start_time is not None and self.plot_end_time is not None:
                mask = (self.time >= self.plot_start_time) & (self.time <= self.plot_end_time)
                interval_max = np.max(self.emg_channels[mask], axis=0, keepdims=True)
                interval_max[interval_max == 0] = 1  # Prevent division by zero
                self.emg_channels = self.emg_channels / interval_max
                self.max_amplitude = 1.0
            
            print(f"Loaded {len(self.time)} data points with {self.emg_channels.shape[1]} channels")
            self.data = data_array
            
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
        
        Args:
            time_idx (int): Time index in the data array
        """
        if self.data is None or time_idx >= len(self.time):
            return np.zeros((3, 3))
        
        # Reset grid to zeros
        grid = np.zeros((3, 3))
        
        # Fill grid with EMG values
        for channel in range(8):
            # Channel numbers in the data are 0-based, but in the mapping they're 1-based
            # and we start with channel 2, so we add 2 to the channel index
            channel_num = channel + 2
            if channel_num in self.electrode_positions:
                row, col = self.electrode_positions[channel_num]
                grid[row, col] = self.emg_channels[time_idx, channel]
        
        return grid
    
    def create_visualization(self, plot_start_time=None, plot_end_time=None):
        """
        Create the interactive visualization with a slider for time navigation.
        Only plots frames within the specified time window if provided.
        """
        if self.data is None or not self.all_grids:
            print("Error: No data loaded or no frames computed.")
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
        
        # Set the middle square to np.nan for all grids
        for grid in grids_to_plot:
            grid[1, 1] = np.nan
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
            img = ax.imshow(grid, cmap=cmap_with_white, vmin=0, vmax=self.max_amplitude)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Time: {time:.3f}s", fontsize=8)
        
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
    
    def plot_channel_mse(self, plot_start_time=None, plot_end_time=None):
        """
        Plot normalized mean squared error between paired channels (2-3, 4-5, 6-7) over the entire time interval of the data, excluding the first and last 0.2 seconds. Each in its own subplot.
        Each MSE value is divided by the mean of the two channel values at that point.
        All subplots share the same y-axis scale for direct comparison.
        """
        # Use the full time and channel data
        times = self.time
        emg = self.emg_channels

        # Exclude the first and last 0.2 seconds
        mask = (times >= (times[0] + 0.2)) & (times <= (times[-1] - 0.2))
        times = times[mask]
        emg = emg[mask]

        epsilon = 1e-8
        # Channel indices: channel 2 = 0, 3 = 1, 4 = 2, 5 = 3, 6 = 4, 7 = 5
        mean_2_3 = (emg[:, 0] + emg[:, 1]) / 2
        mse_2_3 = ((emg[:, 0] - emg[:, 1]) ** 2) / (mean_2_3 + epsilon)

        mean_4_5 = (emg[:, 2] + emg[:, 3]) / 2
        mse_4_5 = ((emg[:, 2] - emg[:, 3]) ** 2) / (mean_4_5 + epsilon)

        mean_6_7 = (emg[:, 4] + emg[:, 5]) / 2
        mse_6_7 = ((emg[:, 4] - emg[:, 5]) ** 2) / (mean_6_7 + epsilon)

        # Find global min and max for y-axis
        all_mse = np.concatenate([mse_2_3, mse_4_5, mse_6_7])
        y_min = np.min(all_mse)
        y_max = np.max(all_mse)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(times, mse_2_3, label='Channels 2-3', color='tab:blue', alpha=0.8)
        axes[0].set_ylabel('Normalized MSE')
        axes[0].set_title('Channels 2-3')
        axes[0].set_ylim(y_min, y_max)
        axes[0].grid(True)

        axes[1].plot(times, mse_4_5, label='Channels 4-5', color='tab:orange', alpha=0.8)
        axes[1].set_ylabel('Normalized MSE')
        axes[1].set_title('Channels 4-5')
        axes[1].set_ylim(y_min, y_max)
        axes[1].grid(True)

        axes[2].plot(times, mse_6_7, label='Channels 6-7', color='tab:green', alpha=0.8)
        axes[2].set_ylabel('Normalized MSE')
        axes[2].set_title('Channels 6-7')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylim(y_min, y_max)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the EMG Muscle Activation Map."""
    # Set normalization option here
    normalize_interval_max = True  # Set to True to normalize by max in plot interval
    emg_map = EMGMuscleActivationMap(
        file_path=FILE_PATH,
        sampling_rate=SAMPLING_RATE,
        window_size=WINDOW_SIZE,
        max_amplitude=MAX_AMPLITUDE,
        normalize_interval_max=normalize_interval_max,
        plot_start_time=PLOT_START_TIME,
        plot_end_time=PLOT_END_TIME
    )
    emg_map.create_visualization(plot_start_time=PLOT_START_TIME, plot_end_time=PLOT_END_TIME)
    # emg_map.plot_channel_mse(plot_start_time=PLOT_START_TIME, plot_end_time=PLOT_END_TIME)

if __name__ == '__main__':
    main()