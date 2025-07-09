"""
Convert PanNuke dataset masks to sparse .npz files.
"""

import argparse
import os
from tqdm import tqdm
import gc
from scipy.sparse import csr_matrix, vstack
from scipy.sparse import save_npz
import numpy as np



def save_sparse_3d_array_hw(file_path, data, mask_shape, chunk_size=3000):
    """
    Save a list of 3D masks (height, width, channels) as multiple sparse .npz files incrementally.

    Args:
        file_path (str): Path to save the .npz file.
        data (list or ndarray): List or array of 3D masks (height, width, channels).
        mask_shape (tuple): Original shape of the mask (height, width, channels).
        chunk_size (int): Number of masks to process per chunk.
    """

    height, width, channels = mask_shape

    for i in tqdm(range(0, len(data), chunk_size), unit="chunk"):
        chunk = data[i:i + chunk_size]  # Get current chunk

        # Flatten each 3D mask into 2D and convert to sparse
        sparse_chunk = []
        for mask in chunk:
            mask_flat = mask.reshape(height * width, channels)  # Flatten spatial dimensions
            sparse_chunk.append(csr_matrix(mask_flat))  # Convert to sparse
        
        sparse_matrix = vstack(sparse_chunk)  # Combine the chunk into a sparse matrix
        chunk_file_path = f"{file_path}_chunk_{i // chunk_size}.npz"
        save_npz(chunk_file_path, sparse_matrix)
        
        del chunk, sparse_chunk, sparse_matrix
        gc.collect()
    



def main(args):
    
    print(f"\n ===== PROCESSING PanNuke Dataset / Fold {args.fold} =====")

    # Load sdata
    print("\n* Loading data...")
    data_dir = os.path.join(args.pannuke_folder, args.fold)
    list_masks = np.load(os.path.join(data_dir, "masks.npy"))
    print("Masks shape:", list_masks.shape)
    
    # Mask shape
    he_patch_width = list_masks[0].shape[0]
    nb_cell_types = 5
    mask_shape = (he_patch_width, he_patch_width, nb_cell_types+1)
    print("Mask shape for saving:", mask_shape)
    
    # Save results
    print("* Saving results...")
    save_sparse_3d_array_hw(os.path.join(data_dir, f"masks"), list_masks, mask_shape)
    
    print("Done.")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get cell type dataset using ST data")

    parser.add_argument("--pannuke_folder", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/ds_slides_cat/pannuke", help="Path to pannuke folder")
    parser.add_argument("--fold", type=str, default="fold2", help="Fold folder")
    
    args = parser.parse_args()
    main(args)

