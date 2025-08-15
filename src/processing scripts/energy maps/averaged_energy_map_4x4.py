'''
This script is used to visualize the EMG muscle activation map of a 4x4 grid of electrodes.

It is designed specifically for a file made of multiple averaged EMG signals.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
import os
from scipy.ndimage import gaussian_filter

# ============================== CONFIGURATION ==============================
INPUT_FOLDER = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\data\06_18_25\all bandpass 20_200 and notch\averaged"  
OUTPUT_FOLDER = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\results\06_18_25\averaged plots 20_200 bandpass\not normalized"  
WINDOW_SIZE = 20  
PLOT_START_FRAME = 0  
PLOT_END_FRAME = None   

normalize_by_max = False  
percent_max = 1

AUTO_SET_MAX_AMPLITUDE = True  
SMOOTH = True
SMOOTH_SIGMA = 10
INTERP_FACTOR = 20
# ===========================================================================

class EMGMuscleActivationMap:
    def __init__(self, file_path, window_size=25, max_amplitude=10,
                 normalize_interval_max=False, plot_start_frame=None, plot_end_frame=None, normalize_by_max=False, auto_set_max_amplitude=True):
        """
        Initialize the EMG muscle activation map visualization.

        args:
            file_path (str): Path to the LabChart text file
            window_size (int): Number of data points per frame
            max_amplitude (float): Maximum EMG amplitude in mV for color scaling
            normalize_interval_max (bool): Whether to normalize by max in plot interval
            plot_start_frame (int): Start frame index for plotting (for normalization)
            plot_end_frame (int): End frame index for plotting (for normalization)
            normalize_by_max (bool): Whether to normalize by max in plot interval
            auto_set_max_amplitude (bool): Whether to auto-set max amplitude from data

        returns:
            None
        """
        self.file_path = file_path
        self.window_size = window_size
        self.max_amplitude = max_amplitude
        self.normalize_interval_max = normalize_interval_max
        self.plot_start_frame = plot_start_frame
        self.plot_end_frame = plot_end_frame
        self.normalize_by_max = normalize_by_max
        self.auto_set_max_amplitude = auto_set_max_amplitude
        self.data = None
        self.emg_channels = None
        self.zero_channels = set()  # Track channels that are all zeros
        
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
                self.max_amplitude = percent_max * max_val
    
    def detect_zero_channels(self):
        """
        Detect channels that are all zeros across all data points.
        """
        if self.emg_channels is None:
            return
        
        for channel in range(self.emg_channels.shape[1]):
            channel_num = channel + 1  # Data columns are 0-based, channel numbers are 1-based
            if channel_num in self.electrode_positions:
                if np.all(self.emg_channels[:, channel] == 0):
                    self.zero_channels.add(channel_num)
                    print(f"Channel {channel_num} is all zeros - will be masked")
    
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
            
            # Detect zero channels
            self.detect_zero_channels()
            
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

        args:
            data_idx (int): Data index in the data array
        returns:
            grid (numpy.ndarray): Grid with EMG values
        """
        if self.data is None or data_idx >= self.data.shape[0]:
            return np.zeros((4, 4))
        grid = np.zeros((4, 4))
        for channel in range(self.emg_channels.shape[1]):
            channel_num = channel + 1  # Data columns are 0-based, channel numbers are 1-based
            if channel_num in self.electrode_positions:
                row, col = self.electrode_positions[channel_num]
                # Set to NaN if channel is all zeros, otherwise use the actual value
                if channel_num in self.zero_channels:
                    grid[row, col] = np.nan
                else:
                    grid[row, col] = self.emg_channels[data_idx, channel]
        return grid
    
    def create_visualization(self, plot_start_frame=None, plot_end_frame=None, normalize_by_max=False, smooth=False, smooth_sigma=2, interp_factor=10, save_path=None):
        """
        Create the visualization with a grid of frames. Plots frames within the specified frame window if provided.
        If normalize_by_max is True, data is normalized by max in plot interval and color scale is set to [0, 1].
        If smooth is True, applies Gaussian smoothing to each grid (after upsampling), ignoring NaNs.
        If save_path is provided, saves the plot instead of showing it.

        args:
            plot_start_frame (int): Start frame index for plotting
            plot_end_frame (int): End frame index for plotting
            normalize_by_max (bool): Whether to normalize by max in plot interval
            smooth (bool): Whether to smooth the grid
            smooth_sigma (float): Sigma for Gaussian smoothing
            interp_factor (int): Interpolation factor for upsampling
            save_path (str): Path to save the plot

        returns:
            None
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
            
            # Smoothing and upsampling
            if smooth:
                # Upsample
                grid_upsampled = np.kron(grid, np.ones((interp_factor, interp_factor)))
                # Mask for valid values
                mask = ~np.isnan(grid_upsampled)
                grid_filled = np.where(mask, grid_upsampled, 0)
                # Gaussian filter
                smoothed = gaussian_filter(grid_filled, sigma=smooth_sigma)
                smoothed_mask = gaussian_filter(mask.astype(float), sigma=smooth_sigma)
                with np.errstate(invalid='ignore', divide='ignore'):
                    result = smoothed / smoothed_mask
                    result[smoothed_mask == 0] = np.nan
                # Restore NaN values for zero channels
                for channel_num in self.zero_channels:
                    if channel_num in self.electrode_positions:
                        row, col = self.electrode_positions[channel_num]
                        # Convert to upsampled coordinates
                        up_row_start = row * interp_factor
                        up_row_end = (row + 1) * interp_factor
                        up_col_start = col * interp_factor
                        up_col_end = (col + 1) * interp_factor
                        result[up_row_start:up_row_end, up_col_start:up_col_end] = np.nan
                
                cmap_with_white = self.cmap.copy()
                cmap_with_white.set_bad(color='white')
                img = ax.imshow(result, cmap=cmap_with_white, vmin=0, vmax=vmax, interpolation='none')
            else:
                cmap_with_white = self.cmap.copy()
                cmap_with_white.set_bad(color='white')
                img = ax.imshow(grid, cmap=cmap_with_white, vmin=0, vmax=vmax, interpolation='none')
            
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
        
        if save_path:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
            plt.close()  # Close the figure to free memory
        else:
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

def process_all_files(input_folder, output_folder, **kwargs):
    """
    Process all .txt files in the input folder and save energy maps to the output folder.
    
    args:
        input_folder (str): Path to folder containing .txt files
        output_folder (str): Path to folder where plots will be saved
        **kwargs: Additional arguments to pass to create_visualization
        
    returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all .txt files in the input folder
    txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.txt')]
    
    if not txt_files:
        print(f"No .txt files found in {input_folder}")
        return
    
    print(f"Found {len(txt_files)} .txt files to process")
    
    for i, filename in enumerate(txt_files, 1):
        file_path = os.path.join(input_folder, filename)
        print(f"\nProcessing file {i}/{len(txt_files)}: {filename}")
        
        try:
            # Create EMG map object
            emg_map = EMGMuscleActivationMap(
                file_path=file_path,
                window_size=kwargs.get('window_size', WINDOW_SIZE),
                plot_start_frame=kwargs.get('plot_start_frame', PLOT_START_FRAME),
                plot_end_frame=kwargs.get('plot_end_frame', PLOT_END_FRAME),
                normalize_by_max=kwargs.get('normalize_by_max', normalize_by_max),
                auto_set_max_amplitude=kwargs.get('auto_set_max_amplitude', AUTO_SET_MAX_AMPLITUDE)
            )
            
            # Generate output filename
            name_without_extension = filename.rsplit('.', 1)[0]
            output_filename = f"{name_without_extension} energy map.png"
            save_path = os.path.join(output_folder, output_filename)
            
            # Create and save visualization
            emg_map.create_visualization(
                plot_start_frame=kwargs.get('plot_start_frame', PLOT_START_FRAME),
                plot_end_frame=kwargs.get('plot_end_frame', PLOT_END_FRAME),
                normalize_by_max=kwargs.get('normalize_by_max', normalize_by_max),
                smooth=kwargs.get('smooth', SMOOTH),
                smooth_sigma=kwargs.get('smooth_sigma', SMOOTH_SIGMA),
                interp_factor=kwargs.get('interp_factor', INTERP_FACTOR),
                save_path=save_path
            )
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\nProcessing complete! Plots saved to: {output_folder}")

def main():
    # Process all files in the input folder
    process_all_files(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        window_size=WINDOW_SIZE,
        plot_start_frame=PLOT_START_FRAME,
        plot_end_frame=PLOT_END_FRAME,
        normalize_by_max=normalize_by_max,
        auto_set_max_amplitude=AUTO_SET_MAX_AMPLITUDE,
        smooth=SMOOTH,
        smooth_sigma=SMOOTH_SIGMA,
        interp_factor=INTERP_FACTOR
    )

if __name__ == '__main__':
    main()