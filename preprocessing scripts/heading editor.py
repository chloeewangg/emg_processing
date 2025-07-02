import pandas as pd
import os

# ======= CONFIGURATION =======
input_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\with noise spikes"  # <-- Set your input folder path
output_folder = r"C:\Users\chloe\OneDrive\Desktop\LEMG research\06_18_25 processed text\with noise spikes fixed headers"  # <-- Set your output folder
# =============================

def replace_header_with_colnums(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.txt') or filename.lower().endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            df = pd.read_csv(input_path, sep=',', header=0)
            n_cols = df.shape[1]
            df.columns = [f'Ch {i+1}' for i in range(n_cols)]
            output_path = os.path.join(output_folder, filename)
            df.to_csv(output_path, sep=',', index=False)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    replace_header_with_colnums(input_folder, output_folder)
