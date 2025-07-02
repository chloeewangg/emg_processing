#!/usr/bin/env python3
"""
This script allows users to cut data from a file based on specified time ranges.
It can remove data from the beginning to a given time, or from a given time to the end.

Usage:
    Edit the configuration section below, then run: python "data remover.py"

Features:
- Remove data from start to a specified time
- Remove data from a specified time to the end
- Preserve file header information
- Support for 22-channel EMG data with 500 Hz sampling rate
- Configuration-based operation (no user prompts)
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np

# ======= CONFIGURATION =======

# Input file path (full path to the file you want to process)
INPUT_FILE_PATH = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\extracted signals\yogurt 20 ml\yogurt 20 ml 2.txt"

# Output directory path (where to save the processed file)
OUTPUT_DIRECTORY = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\temporally aligned and averaged\cut data"

# Time parameters (in seconds):
# cut_start_time: Time to start keeping data (0 = from beginning)
# cut_end_time: Time to stop keeping data ('END' = to the end, or specify a time in seconds)
CUT_START_TIME = 0.0  # Start from beginning
CUT_END_TIME = 1.16  # Go to the end

# Sampling rate (Hz) - used to convert time to sample indices
SAMPLING_RATE = 500

# Overwrite existing files without asking (True/False)
AUTO_OVERWRITE = False

# ============================

def get_user_input(prompt, input_type=str, allow_empty=False):
    """Get user input with validation."""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and not allow_empty:
                print("Input cannot be empty. Please try again.")
                continue
            
            if input_type == float:
                return float(user_input) if user_input else None
            elif input_type == int:
                return int(user_input) if user_input else None
            else:
                return user_input
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)

def read_data_file(file_path):
    """Read data file and separate header from data."""
    header_lines = []
    data_lines = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # First line is header
        if len(lines) > 0:
            header_lines = [lines[0]]
            data_lines = lines[1:]  # All lines after header
        
        return header_lines, data_lines
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def time_to_sample_index(time_seconds, sampling_rate):
    """Convert time in seconds to sample index."""
    return int(time_seconds * sampling_rate)

def find_sample_indices(data_lines, start_time, end_time, sampling_rate):
    """Find start and end sample indices for the specified time range."""
    total_samples = len(data_lines)
    
    # Convert times to sample indices
    if start_time == 0:
        start_idx = 0
    else:
        start_idx = time_to_sample_index(start_time, sampling_rate)
    
    if end_time == 'END':
        end_idx = total_samples
    else:
        end_idx = time_to_sample_index(end_time, sampling_rate)
    
    # Ensure indices are within bounds
    start_idx = max(0, min(start_idx, total_samples))
    end_idx = max(start_idx, min(end_idx, total_samples))
    
    return start_idx, end_idx

def write_data_file(output_path, header_lines, data_lines, start_idx, end_idx):
    """Write the processed data to a new file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w') as file:
            # Write header
            for line in header_lines:
                file.write(line)
            
            # Write data
            for i in range(start_idx, end_idx):
                if i < len(data_lines):
                    file.write(data_lines[i])
        
        return True
        
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def validate_configuration():
    """Validate the configuration parameters."""
    errors = []
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE_PATH):
        errors.append(f"Input file does not exist: {INPUT_FILE_PATH}")
    
    # Check time parameters
    if CUT_START_TIME < 0:
        errors.append(f"Cut start time must be non-negative: {CUT_START_TIME}")
    
    if CUT_END_TIME != 'END':
        try:
            end_time = float(CUT_END_TIME)
            if end_time < 0:
                errors.append(f"Cut end time must be non-negative: {CUT_END_TIME}")
            if CUT_START_TIME >= end_time:
                errors.append(f"Cut start time ({CUT_START_TIME}) must be less than cut end time ({CUT_END_TIME})")
        except ValueError:
            errors.append(f"Cut end time must be 'END' or a valid number: {CUT_END_TIME}")
    
    # Check sampling rate
    if SAMPLING_RATE <= 0:
        errors.append(f"Sampling rate must be positive: {SAMPLING_RATE}")
    
    return errors

def main():
    """Main function to run the data remover."""
    print("=" * 60)
    print("DATA REMOVER SCRIPT")
    print("=" * 60)
    print("Configuration-based data cutting tool for 22-channel EMG data.")
    print()
    
    # Validate configuration
    config_errors = validate_configuration()
    if config_errors:
        print("Configuration errors found:")
        for error in config_errors:
            print(f"  - {error}")
        print("\nPlease fix the configuration and run again.")
        return
    
    # Display configuration
    print("Configuration:")
    print(f"  Input file: {INPUT_FILE_PATH}")
    print(f"  Output directory: {OUTPUT_DIRECTORY}")
    print(f"  Sampling rate: {SAMPLING_RATE} Hz")
    print(f"  Cut start time: {CUT_START_TIME}s")
    print(f"  Cut end time: {CUT_END_TIME}")
    print()
    
    # Read the data file
    print(f"Reading file: {INPUT_FILE_PATH}")
    header_lines, data_lines = read_data_file(INPUT_FILE_PATH)
    
    if data_lines is None:
        print("Failed to read the data file. Exiting.")
        return
    
    print(f"File loaded successfully!")
    print(f"Total data points: {len(data_lines)}")
    if data_lines:
        duration = len(data_lines) / SAMPLING_RATE
        print(f"Duration: {duration:.3f}s")
    print()
    
    # Find sample indices for cutting
    start_idx, end_idx = find_sample_indices(data_lines, CUT_START_TIME, CUT_END_TIME, SAMPLING_RATE)
    
    # Calculate what will be kept
    total_points = len(data_lines)
    kept_points = end_idx - start_idx
    removed_points = total_points - kept_points
    
    # Determine operation description
    if CUT_START_TIME == 0 and CUT_END_TIME == 'END':
        operation_desc = "no cutting (keeping all data)"
    elif CUT_START_TIME == 0:
        operation_desc = f"removed from {CUT_END_TIME}s to end"
    elif CUT_END_TIME == 'END':
        operation_desc = f"removed from start to {CUT_START_TIME}s"
    else:
        operation_desc = f"removed from start to {CUT_START_TIME}s and from {CUT_END_TIME}s to end"
    
    print(f"Operation summary:")
    print(f"- {operation_desc}")
    print(f"- Total data points: {total_points}")
    print(f"- Points to keep: {kept_points}")
    print(f"- Points to remove: {removed_points}")
    
    if kept_points <= 0:
        print("Warning: No data will remain after this operation!")
        if not AUTO_OVERWRITE:
            confirm = get_user_input("Continue anyway? (y/n): ").lower()
            if confirm != 'y':
                print("Operation cancelled.")
                return
    
    # Generate output filename
    input_filename = os.path.basename(INPUT_FILE_PATH)
    name, ext = os.path.splitext(input_filename)
    
    output_filename = f"{name} cut.txt"
    
    output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
    
    # Check if output file already exists
    if os.path.exists(output_path) and not AUTO_OVERWRITE:
        overwrite = get_user_input(f"File '{output_path}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Operation cancelled.")
            return
    
    # Process and save the file
    print(f"\nProcessing data...")
    success = write_data_file(output_path, header_lines, data_lines, start_idx, end_idx)
    
    if success:
        print(f"Success! Processed file saved to: {output_path}")
        
        # Show time range of kept data
        if kept_points > 0:
            kept_start_time = start_idx / SAMPLING_RATE
            kept_end_time = end_idx / SAMPLING_RATE
            print(f"Kept data time range: {kept_start_time:.3f}s to {kept_end_time:.3f}s")
    else:
        print("Failed to save the processed file.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your configuration and try again.")
