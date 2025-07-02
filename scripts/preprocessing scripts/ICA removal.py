'''
This script is used to remove ICA components from an EMG signal.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA # type: ignore
import os

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


def plot_ica_components(components, time, max_points_to_plot=10000):
    """
    Plots the separated ICA components. Decimates data if it's too long.
    """
    if len(time) > max_points_to_plot:
        step = len(time) // max_points_to_plot
        time = time[::step]
        components = components[:, ::step]
        
    num_components = components.shape[0]
    fig, axes = plt.subplots(num_components, 1, figsize=(15, 1.5 * num_components), sharex=True)
    if num_components == 1:
        axes = [axes]

    for i in range(num_components):
        axes[i].plot(time, components[i, :])
        axes[i].set_ylabel(f'{i+1}')

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def main():
    # ======= CONFIGURATION =======
    file_path = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\shorts removed\yogurt 20 ml 2.txt"  # <-- Set your input file path
    output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\other"  # <-- Set output folder, or leave blank to skip
    sampling_rate = 500  # Hz
    num_channels_to_process = 16
    max_points_to_plot = 5000  # Plot a max of 5000 points to keep it fast
    # =============================

    # Load the data
    print("Loading data...")
    data = load_emg_data(file_path)
    if data is None:
        return

    # Select first 16 channels and generate time vector
    emg_data = data.iloc[:, :num_channels_to_process]
    time = np.arange(emg_data.shape[0]) / sampling_rate
    print(f"Data loaded: {emg_data.shape[1]} channels, {len(time):.2f} seconds.")

    # Plot original data (with same formatting as reconstructed data)
    print("\nPlotting original data...")
    trim_samples = int(0.3 * sampling_rate)
    plot_time = time[trim_samples:-trim_samples]
    plot_data_df = emg_data.iloc[trim_samples:-trim_samples]
    plot_data(plot_data_df, plot_time, title_prefix='Original', max_points_to_plot=max_points_to_plot)

    # Perform PCA to show explained variance and help user decide on component number
    plot_explained_variance(emg_data)

    # Ask user for number of ICA components
    while True:
        try:
            n_components = int(input("\nEnter the number of ICA components to use: "))
            if n_components < 1 or n_components > num_channels_to_process:
                raise ValueError(f"Please enter a number between 1 and {num_channels_to_process}.")
            break
        except ValueError as e:
            print(f"Invalid input. {e}")

    # Perform ICA
    print("\nPerforming ICA...")
    ica = FastICA(n_components=n_components, random_state=42, whiten='unit-variance', max_iter=1000)
    # Fit ICA on (samples, features) and get sources
    sources = ica.fit_transform(emg_data)  # Shape: (n_samples, n_components)

    # Plot ICA components, excluding start and end
    print("\nPlotting ICA components...")
    # Exclude the first and last 0.4 seconds for plotting to avoid boundary artifacts
    trim_samples = int(0.4 * sampling_rate)

    if len(time) > 2 * trim_samples:
        plot_time_ica = time[trim_samples:-trim_samples]
        sources_for_plot = sources.T[:, trim_samples:-trim_samples]
    else:
        plot_time_ica = time
        sources_for_plot = sources.T
    
    # Pass transposed sources to plotting function which expects (n_components, n_samples)
    plot_ica_components(sources_for_plot, plot_time_ica, max_points_to_plot=max_points_to_plot)

    # Get user input to remove components
    while True:
        try:
            prompt = "\nEnter the component numbers to remove (e.g., '1, 3, 5') or press Enter to keep all: "
            user_input = input(prompt)
            if not user_input.strip():
                components_to_remove = []
                break
            # Convert 1-based user input to 0-based indices
            components_to_remove = [int(i.strip()) - 1 for i in user_input.replace(',', ' ').split()]
            # Validate indices
            if any(i < 0 or i >= num_channels_to_process for i in components_to_remove):
                raise ValueError("Component number out of range.")
            break
        except ValueError as e:
            print(f"Invalid input. Please enter numbers between 1 and {num_channels_to_process}. Details: {e}")

    # Reconstruct the signal without the selected components
    if components_to_remove:
        print(f"\nRemoving components: {[i + 1 for i in components_to_remove]}...")
        # Create a copy of the sources to modify
        modified_sources = sources.copy()
        # Zero out the columns for the components to remove
        modified_sources[:, components_to_remove] = 0
        # Inverse transform to get the cleaned data
        reconstructed_emg = ica.inverse_transform(modified_sources)
        reconstructed_df = pd.DataFrame(reconstructed_emg, columns=emg_data.columns)
    else:
        print("\nNo components removed.")
        reconstructed_df = emg_data # No change

    # Plot reconstructed data
    print("\nPlotting reconstructed data...")
    # Exclude first and last 0.3 seconds for plotting
    trim_samples = int(0.3 * sampling_rate)
    plot_time = time[trim_samples:-trim_samples]
    plot_data_df = reconstructed_df.iloc[trim_samples:-trim_samples]
    plot_data(plot_data_df, plot_time, title_prefix='Reconstructed', max_points_to_plot=max_points_to_plot)

    # Save the reconstructed data if a path is provided
    if output_folder:
        # Construct the full output path including the original filename
        output_filename = os.path.basename(file_path)
        full_output_path = os.path.join(output_folder, output_filename)
        
        print(f"\nSaving reconstructed data to {full_output_path}...")
        try:
            # Ensure the output directory exists
            os.makedirs(output_folder, exist_ok=True)
            # Concatenate the last 6 unprocessed channels to the reconstructed data
            unprocessed_channels = data.iloc[:, -6:]
            combined_df = pd.concat([reconstructed_df, unprocessed_channels], axis=1)
            # Save the full (untrimmed) reconstructed + unprocessed data
            combined_df.to_csv(full_output_path, sep=',', index=False)
            print("Save complete.")
        except Exception as e:
            print(f"Error saving file: {e}")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main() 