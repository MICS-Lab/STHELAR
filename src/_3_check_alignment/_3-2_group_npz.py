"""Group all npz in one unique file"""


import os
from scipy.sparse import load_npz, vstack, save_npz
from tqdm import tqdm
import argparse
import gc
import re
import numpy as np



def consolidate_chunks(folder_path, chunk_prefix, output_file):
    """
    Consolidate multiple sparse chunk files into a single sparse .npz file.
    
    Args:
        folder_path (str): Path to the folder containing chunk files.
        chunk_prefix (str): Prefix of the chunk files (e.g., 'masks_chunk', 'masks_cells_chunk').
        output_file (str): Path to save the consolidated .npz file.
    """
    print(f"\n-> Consolidating chunks with prefix '{chunk_prefix}' in {folder_path}...")

    # Collect all chunk files matching the prefix
    chunk_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith(chunk_prefix) and f.endswith(".npz")]
    )

    if not chunk_files:
        print(f"No chunks found with prefix '{chunk_prefix}'. Skipping.")
        return

    # Extract chunk indices and sort numerically
    def extract_index(file_name):
        match = re.search(rf"{chunk_prefix}_(\d+)\.npz$", file_name)
        return int(match.group(1)) if match else float('inf')

    chunk_files = sorted(chunk_files, key=lambda x: extract_index(os.path.basename(x)))

    # Load and combine all sparse chunks
    sparse_matrices = []
    for chunk_file in tqdm(chunk_files, desc=f"Loading {chunk_prefix}", unit="chunk"):
        sparse_chunk = load_npz(chunk_file)
        sparse_matrices.append(sparse_chunk)
        del sparse_chunk  # Release memory
        gc.collect()

    # Combine into a single sparse matrix
    print(f"-> Combining {len(sparse_matrices)} chunks...")
    final_sparse_matrix = vstack(sparse_matrices)

    # Save the combined sparse matrix
    print(f"-> Saving...")
    save_npz(output_file, final_sparse_matrix)

    # # Ensure the output file is saved
    # if os.path.exists(output_file):
    #     # If the file is saved successfully, delete all chunk files
    #     print(f"-> Deleting chunk files after successful save...")
    #     for chunk_file in chunk_files:
    #         os.remove(chunk_file)
    #         print(f"  - Deleted {chunk_file}")
    # else:
    #     print(f"-> Warning: Output file '{output_file}' was not created. Chunk files retained.")

    # Cleanup
    del sparse_matrices, final_sparse_matrix
    gc.collect()

    print(f"Done.")



def consolidate_npy_chunks(folder_path, file_prefix, output_file):
    """
    Consolidate multiple .npy chunk files into a single .npy file.
    
    Args:
        folder_path (str): Path to the folder containing chunk files.
        file_prefix (str): Prefix of the chunk files (e.g., 'images_chunk').
        output_file (str): Path to save the consolidated .npy file.
    """
    print(f"\n-> Consolidating .npy chunks with prefix '{file_prefix}' in {folder_path}...")

    # Collect all .npy chunk files matching the prefix
    chunk_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith(file_prefix) and f.endswith(".npy")],
        key=lambda x: int(re.search(rf"{file_prefix}_(\d+)\.npy$", os.path.basename(x)).group(1))
    )

    if not chunk_files:
        print(f"No .npy chunks found with prefix '{file_prefix}'. Skipping.")
        return

    # Load and combine all chunks
    arrays = []
    for chunk_file in tqdm(chunk_files, desc=f"Loading {file_prefix}", unit="chunk"):
        arrays.append(np.load(chunk_file))

    # Concatenate and save the final array
    final_array = np.concatenate(arrays, axis=0)
    np.save(output_file, final_array)

    # # Ensure the output file is saved before deleting chunks
    # if os.path.exists(output_file):
    #     print(f"-> Deleting chunk files after successful save...")
    #     for chunk_file in chunk_files:
    #         os.remove(chunk_file)
    #         print(f"  - Deleted {chunk_file}")
    # else:
    #     print(f"-> Warning: Output file '{output_file}' was not created. Chunk files retained.")
    
    # Cleanup
    del arrays, final_array
    gc.collect()

    print(f"Done.")




def consolidate_nested_chunks(folder_path, chunk_prefix, output_file):
    """
    Consolidate nested sparse chunk files into a single sparse .npz file.
    
    Args:
        folder_path (str): Path to the folder containing chunk files.
        chunk_prefix (str): Prefix of the chunk files (e.g., 'masks_chunk').
        output_file (str): Path to save the consolidated .npz file.
    """
    print(f"\n-> Consolidating nested chunks with prefix '{chunk_prefix}' in {folder_path}...")

    # Collect all nested chunk files matching the prefix
    nested_chunk_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith(chunk_prefix) and f.endswith(".npz")]
    )

    if not nested_chunk_files:
        print(f"No nested chunks found with prefix '{chunk_prefix}'. Skipping.")
        return

    # Group by "i" and then by "j"
    def extract_indices(file_name):
        match = re.search(rf"{chunk_prefix}_(\d+)_chunk_(\d+)\.npz$", file_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return float('inf'), float('inf')

    nested_chunk_files = sorted(nested_chunk_files, key=lambda x: extract_indices(os.path.basename(x)))

    # Combine all nested sparse chunks
    sparse_matrices = []
    for nested_chunk_file in tqdm(nested_chunk_files, desc=f"Loading {chunk_prefix}", unit="nested_chunk"):
        sparse_chunk = load_npz(nested_chunk_file)
        sparse_matrices.append(sparse_chunk)

    # Combine into a single sparse matrix
    print(f"-> Combining {len(sparse_matrices)} nested chunks...")
    final_sparse_matrix = vstack(sparse_matrices)

    # Save the combined sparse matrix
    print(f"-> Saving...")
    save_npz(output_file, final_sparse_matrix)

    # # Ensure the output file is saved before deleting chunks
    # if os.path.exists(output_file):
    #     print(f"-> Deleting nested chunk files after successful save...")
    #     for nested_chunk_file in nested_chunk_files:
    #         os.remove(nested_chunk_file)
    #         print(f"  - Deleted {nested_chunk_file}")
    # else:
    #     print(f"-> Warning: Output file '{output_file}' was not created. Nested chunk files retained.")

    # Cleanup
    del sparse_matrices, final_sparse_matrix
    gc.collect()

    print(f"Done.")




def process_slide_folders(slide_ids, folder_name):
    """
    Process all slide folders to consolidate sparse mask chunks into single .npz files.
    
    Args:
        slide_ids (list): List of slide IDs to process.
        folder_name (str): Path to the parent folder containing slide subfolders.
    """
    for slide_id in slide_ids:
        print(f"\n===== PROCESSING SLIDE: {slide_id} =====")
        slide_folder = os.path.join(folder_name, slide_id)
        
        if not os.path.exists(slide_folder):
            print(f"Slide folder '{slide_folder}' does not exist. Skipping.")
            continue

        # Check for images.npy or chunked images
        images_file = os.path.join(slide_folder, "images.npy")
        chunked_images = sorted(
            [os.path.join(slide_folder, f) for f in os.listdir(slide_folder) if f.startswith("images_chunk") and f.endswith(".npy")]
        )

        if os.path.exists(images_file):
            print(f"[INFO] Single 'images.npy' file detected for slide {slide_id}.")
            consolidate_chunks(slide_folder, "masks_chunk", os.path.join(slide_folder, "masks.npz"))
            consolidate_chunks(slide_folder, "masks_cells_chunk", os.path.join(slide_folder, "masks_cells.npz"))
        elif chunked_images:
            print(f"[INFO] Chunked 'images_chunk' files detected for slide {slide_id}.")
            consolidate_npy_chunks(slide_folder, "images_chunk", os.path.join(slide_folder, "images.npy"))
            consolidate_npy_chunks(slide_folder, "types_chunk", os.path.join(slide_folder, "types.npy"))
            consolidate_npy_chunks(slide_folder, "patch_ids_chunk", os.path.join(slide_folder, "patch_ids.npy"))
            consolidate_nested_chunks(slide_folder, "masks_chunk", os.path.join(slide_folder, "masks.npz"))
            consolidate_nested_chunks(slide_folder, "masks_cells_chunk", os.path.join(slide_folder, "masks_cells.npz"))

    print("\nAll slides processed successfully.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate sparse mask chunks for multiple slides.")

    # Input arguments
    parser.add_argument("--slide_ids", type=str, nargs="+", required=True, help="List of slide IDs to process.")
    parser.add_argument("--folder_name", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/patches_xenium", help="Parent folder containing slide subfolders.")

    args = parser.parse_args()

    # Run the consolidation process
    process_slide_folders(args.slide_ids, args.folder_name)
