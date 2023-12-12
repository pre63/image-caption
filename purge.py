import os
import shutil
import pandas as pd


data = pd.read_csv('final.csv')



# Ensure the destination directory exists
destination_folder = 'images2/'
os.makedirs(destination_folder, exist_ok=True)

# Copy each file from the DataFrame to the destination folder
for file_name in data['file_name']:
    source_path = os.path.join("images/", os.path.basename(file_name))
    destination_path = os.path.join(destination_folder, os.path.basename(file_name))

    # Copy the file
    shutil.copy(source_path, destination_path)

print("Files copied successfully.")

