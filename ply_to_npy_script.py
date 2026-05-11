import os
import numpy as np
from plyfile import PlyData

def convert_ply_to_npy(source_folder, dest_folder):
    # Create the destination 'raw' folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created destination folder: {dest_folder}")

    # List all .ply files
    ply_files = [f for f in os.listdir(source_folder) if f.endswith('.ply')]

    if not ply_files:
        print(f"No .ply files found in {source_folder}")
        return

    for filename in ply_files:
        ply_path = os.path.join(source_folder, filename)
        npy_path = os.path.join(dest_folder, filename.replace('.ply', '.npy'))

        try:
            # Load the binary PLY data
            plydata = PlyData.read(ply_path)
            v_data = plydata['vertex'].data
            
            # Stack x, y, z, and class into a single Nx4 array
            # Note: Using .astype(np.float64) to keep the file size manageable 
            # while preserving coordinate precision.
            points_with_labels = np.column_stack((
                v_data['x'], 
                v_data['y'], 
                v_data['z'], 
                v_data['scalar_Classification']
            )).astype(np.float64)

            # Save to the 'raw' folder
            np.save(npy_path, points_with_labels)
            print(f"Converted {filename} -> {npy_path} | Shape: {points_with_labels.shape}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    SOURCE = 'segmented_tiles'
    DESTINATION = 'raw'
    convert_ply_to_npy(SOURCE, DESTINATION)

