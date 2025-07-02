'''
This script is used to process text files by copying them to a new folder.
'''

import os
import shutil

def process_folders(source_path, destination_path):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    
    # Iterate through each folder in the source path
    for folder_name in os.listdir(source_path):
        folder_path = os.path.join(source_path, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
            
        # Look for text files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                # Source file path
                source_file = os.path.join(folder_path, file_name)
                
                # New file name will be the folder name with .txt extension
                new_file_name = folder_name + '.txt'
                destination_file = os.path.join(destination_path, new_file_name)
                
                # Copy and rename the file
                shutil.copy2(source_file, destination_file)
                print(f"Copied {file_name} from {folder_name} to {new_file_name}")
                
                # Break after finding first text file
                break

if __name__ == "__main__":
    # Specify your source and destination paths here
    source_path = r"C:\Users\chloe\Documents\FreeBCI_GUI\Recordings"
    destination_path = r"C:\Users\chloe\OneDrive\Desktop\EMG stuff\06_18_25 processed text"
    
    process_folders(source_path, destination_path)
