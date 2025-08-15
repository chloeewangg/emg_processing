'''
This script is used to rectify an EMG signal.
'''

import pandas as pd
import os

# ======= CONFIGURATION =======
input_file = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\shorts removed\apple 10 ml 3.txt"  
output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\other"  
# =============================

def rectify_and_save(input_file, output_folder):
    """
    Loads and rectifies (absolute value) the first 16 channels of an EMG signal, then saves the result.
    """
    # 1. Load data (comma-delimited, no header, only first 16 columns)
    try:
        data = pd.read_csv(input_file, sep=',', header=None, usecols=range(16), skiprows=1)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Rectify the data
    rectified_data = data.abs()

    # 3. Save the rectified data
    try:
        os.makedirs(output_folder, exist_ok=True)
        base_filename = os.path.basename(input_file)
        name, ext = os.path.splitext(base_filename)
        output_filename = f"{name} rectified{ext}"
        output_path = os.path.join(output_folder, output_filename)
        rectified_data.to_csv(output_path, header=False, index=False)
        print(f"Successfully saved rectified data to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    rectify_and_save(input_file, output_folder)
