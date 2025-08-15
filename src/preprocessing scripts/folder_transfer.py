import os
import shutil

# ---------------- CONFIGURATION ----------------
INPUT_DIR = r"C:\Users\chloe\Documents\FreeBCI_GUI\Recordings\wenjian"  
OUTPUT_DIR = r"C:\Users\chloe\OneDrive\Desktop\swallow EMG\data\participants\wenjian"  
# ----------------------------------------------

def main():
    # Check input directory exists
    if not os.path.isdir(INPUT_DIR):
        print(f"Input directory does not exist: {INPUT_DIR}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Loop through class folders
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue  # skip files

        # Prepare output class folder
        output_class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        # Loop through sample folders
        for sample_name in os.listdir(class_path):
            sample_path = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_path):
                continue  # skip files

            # Find the .txt file in the sample folder
            txt_files = [f for f in os.listdir(sample_path) if f.lower().endswith('.txt')]
            if not txt_files:
                print(f"No .txt file found in {sample_path}")
                continue
            txt_file = txt_files[0]
            src_txt_path = os.path.join(sample_path, txt_file)

            # Destination path: output/class/sample.txt
            dst_txt_path = os.path.join(output_class_path, f"{sample_name}.txt")

            # Read, process, and write the file (remove header rows and leading zero rows)
            with open(src_txt_path, 'r') as infile, open(dst_txt_path, 'w') as outfile:
                data_rows = []
                for line in infile:
                    tokens = [t.strip() for t in line.strip().split(',')]
                    # Skip header rows (any row with non-numeric data)
                    if not tokens or any(not (token.lstrip('-').replace('.', '', 1).isdigit()) for token in tokens):
                        continue
                    data_rows.append(tokens)
                # Remove leading rows where all columns are zero
                first_nonzero_idx = 0
                for i, row in enumerate(data_rows):
                    # Convert all tokens to float for zero check
                    try:
                        nums = [float(token) for token in row]
                    except ValueError:
                        continue  # skip malformed rows
                    if any(val != 0 for val in nums):
                        first_nonzero_idx = i
                        break
                # Write from the first nonzero row onward
                for row in data_rows[first_nonzero_idx:]:
                    outfile.write(','.join(row) + '\n')
            print(f"Processed and copied {src_txt_path} -> {dst_txt_path}")

if __name__ == "__main__":
    main() 