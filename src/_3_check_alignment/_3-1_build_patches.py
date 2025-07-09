"""
Generating patches with corresponding masks in the same format as the Pannuke dataset but to check align (fake cell types)
!!! WARNING: A solution is also implemented in the CellViT_for_STHELAR github repository to check alignment directly from final patches with 
right cell type category number to avoid too many computation and memory (this step can be skipped in that case). !!!
"""

import argparse
import os
from tqdm import tqdm
import numpy as np
import random
from joblib import Parallel, delayed
import pandas as pd
import gc
from scipy.sparse import csr_matrix, vstack
from scipy.sparse import save_npz

import spatialdata as sd
from skimage.draw import polygon
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.validation import explain_validity

import matplotlib.pyplot as plt

from sopa.segmentation import Patches2D
from sopa._sdata import get_spatial_image, to_intrinsic



def int_cell_id(cell_id_str: str) -> int:
    """
    Converts a Xenium Explorer alphabetical cell_id back to an integer.
    E.g., int_cell_id('aaaachba-1') = 10000
    """
    cell_id_str = cell_id_str[:-2]  # Remove the '-1' suffix
    cell_id = 0
    for char in cell_id_str:
        cell_id = cell_id * 16 + (ord(char) - 97)  # Base-16 conversion (a-p -> 0-15)
    # Shift by 1 to avoid having 0 as a cell ID
    return cell_id + 1



def str_cell_id(cell_id: int) -> str:
    """Transforms an integer cell ID into an Xenium Explorer alphabetical cell id
    E.g., str_cell_id(10000) = 'aaaachba-1'"""
    cell_id -= 1 # Shift by 1 to avoid having 0 as a cell ID because of background
    coefs = []
    for _ in range(8):
        cell_id, coef = divmod(cell_id, 16)
        coefs.append(coef)
    return "".join([chr(97 + coef) for coef in coefs][::-1]) + "-1"



def prepare_sdata(sdata, patch_width, patch_overlap):

    he = get_spatial_image(sdata, "he")  # Gets a DataArray from a SpatialData object (if the image has multiple scale, the `scale0` is returned)
    geo_df_nuclei = to_intrinsic(sdata, sdata.shapes['nucleus_boundaries'], he)
    geo_df_cells = to_intrinsic(sdata, sdata.shapes['cell_boundaries'], he)

    if "he_patches" in sdata.shapes:
        print("[INFO] 'he_patches' already exists in sdata. Skipping writing.")
    else:
        patches = Patches2D(sdata, 'he', patch_width=patch_width, patch_overlap=patch_overlap)
        patches.write()
        sdata.shapes['he_patches'] = sdata.shapes.pop('sopa_patches')
        # Add patch_id to the patch dataframe (needed for check align)
        sdata.shapes['he_patches']['patch_id'] = sdata.shapes['he_patches'].index
        # Replace on disk
        sdata.delete_element_from_disk("sopa_patches")
        sdata.write_element("he_patches")

    patch_df = sdata.shapes['he_patches']

    return he, geo_df_nuclei, geo_df_cells, patch_df



def patchs_cells_intersections(patch_df, geo_df_nuclei):

    # Returns the indices of the patches and corresponding cells that intersect
    patch_idx, cell_idx = geo_df_nuclei.geometry.sindex.query(patch_df.geometry, predicate=None)

    # Group the cell data by patch index
    patch_idx_2_cells_idx = pd.Series(geo_df_nuclei.index[cell_idx]).groupby(patch_idx).apply(list)
    patch_cell_pairs_idx = [(patch_idx, patch_idx_2_cells_idx[patch_idx]) for patch_idx in patch_idx_2_cells_idx.index]

    # Return the patch index and their corresponding cell indices in the same order
    return patch_cell_pairs_idx



def rasterize(mask, cell_patch_intersection, xy_min, nb_cell_types, count, cell_id):

    # Extract the coordinates of the intersection
    if isinstance(cell_patch_intersection, Polygon):
        intersection_coords = np.array(cell_patch_intersection.exterior.coords)
    elif isinstance(cell_patch_intersection, MultiPolygon):
        intersection_coords = []
        for poly in cell_patch_intersection.geoms:
            intersection_coords.extend(np.array(poly.exterior.coords))
        intersection_coords = np.array(intersection_coords)
    else:
        #print(type(cell_patch_intersection))
        #print(cell_patch_intersection)
        raise ValueError("Unsupported geometry type")

    # Convert intersection coordinates to pixel coordinates
    intersection_coords[:, 0] -= xy_min[0]
    intersection_coords[:, 1] -= xy_min[1]

    # Fill the polygon defined by the intersection with ones in the mask
    rr, cc = polygon(intersection_coords[:, 1], intersection_coords[:, 0], mask.shape[0:2])
    mask[rr, cc, 0] = count
    mask[rr, cc, nb_cell_types] = int_cell_id(cell_id)  # use cell_id for H&E features after

    return mask



def rasterize_cells(mask, cell_patch_intersection, xy_min, cell_id):

    # Extract the coordinates of the intersection
    if isinstance(cell_patch_intersection, Polygon):
        intersection_coords = np.array(cell_patch_intersection.exterior.coords)
    elif isinstance(cell_patch_intersection, MultiPolygon):
        intersection_coords = []
        for poly in cell_patch_intersection.geoms:
            intersection_coords.extend(np.array(poly.exterior.coords))
        intersection_coords = np.array(intersection_coords)
    else:
        #print(type(cell_patch_intersection))
        #print(cell_patch_intersection)
        raise ValueError("Unsupported geometry type")

    # Convert intersection coordinates to pixel coordinates
    intersection_coords[:, 0] -= xy_min[0]
    intersection_coords[:, 1] -= xy_min[1]

    # Fill the polygon defined by the intersection with ones in the mask
    rr, cc = polygon(intersection_coords[:, 1], intersection_coords[:, 0], mask.shape[0:2])
    mask[rr, cc] = int_cell_id(cell_id)  # use cell_id for H&E features after

    return mask



def correct_geometry(geometry):
    """
    Corrects the input geometry if invalid due to self-intersection
    """
    if not geometry.is_valid:
        explanation = explain_validity(geometry)
        #print(f"Invalid geometry: {explanation}")
        if "Self-intersection" in explanation:
            # Attempt to fix self-intersection by buffering with zero distance
            corrected_geometry = geometry.buffer(0)
            if corrected_geometry.is_valid:
                return corrected_geometry
    return geometry



def handling_patch(p2c_idx, patch_df, geo_df_nuclei, geo_df_cells, he_data, nb_cell_types, he_patch_width):

    patch_idx = p2c_idx[0]
    
    patch_id = patch_df['patch_id'].iloc[patch_idx]
    patch_bounds = patch_df.bounds.iloc[patch_idx]

    list_patches_n = []
    list_masks_nuclei_n = []
    list_masks_cells_n = []
    list_patch_ids_n = []

    ymin, ymax, xmin, xmax = int(patch_bounds['miny']), int(patch_bounds['maxy']), int(patch_bounds['minx']), int(patch_bounds['maxx'])
    patch = box(xmin, ymin, xmax, ymax)
    
    # Create an empty mask for the current patch
    mask_nuclei = np.zeros((ymax-ymin, xmax-xmin, nb_cell_types+1), dtype=np.int32)
    mask_cells = np.zeros((ymax-ymin, xmax-xmin), dtype=np.int32)
    skip = 0
    
    # Iterate over cells in the patch
    count = 0

    for cell_idx in p2c_idx[1]:

        nuclei = geo_df_nuclei.loc[cell_idx].geometry
        cell = geo_df_cells.loc[cell_idx].geometry
        
        if not isinstance(nuclei, (Polygon, MultiPolygon)) or not isinstance(cell, (Polygon, MultiPolygon)):
            print(f"Unexpected type for cell: {type(nuclei)}, {type(cell)}, value: {nuclei}, {cell}")
            continue

        nuclei = correct_geometry(nuclei)
        cell = correct_geometry(cell)

        if not nuclei.is_valid or not cell.is_valid:
            # Handle invalid geometry
            print(f"Invalid geometry for {cell_idx}")
            continue
    
        if patch.intersects(nuclei) and patch.intersects(cell):
            nuclei_patch_intersection = nuclei.intersection(patch)
            cell_patch_intersection = cell.intersection(patch)

            if not nuclei_patch_intersection.is_empty and not cell_patch_intersection.is_empty:
                count += 1
                try:
                    mask_nuclei = rasterize(mask_nuclei, nuclei_patch_intersection, (xmin, ymin), nb_cell_types, count, cell_idx)  # Update the mask to include the intersection area of the current cell
                    mask_cells = rasterize_cells(mask_cells, cell_patch_intersection, (xmin, ymin), cell_idx)  # Update the mask to include the intersection area of the current cell
                except ValueError as e:
                    #print("Error rasterizing cell:", e)
                    skip = 1
                    continue      
    
    if not np.all(mask_nuclei == 0) and not np.all(mask_cells): #and skip == 0:
        img = np.transpose(he_data[:, ymin:ymax, xmin:xmax], (1, 2, 0))
        if img.shape == (he_patch_width, he_patch_width, 3) and mask_nuclei.shape == (he_patch_width, he_patch_width, nb_cell_types+1):
            # Append the patch and mask to the lists
            list_patches_n.append(img)
            list_masks_nuclei_n.append(mask_nuclei)
            list_masks_cells_n.append(mask_cells)
            list_patch_ids_n.append(patch_id)
        else:
            print("Patch and mask shapes do not match the expected shape -- skipping")
    
    return list_patches_n, list_masks_nuclei_n, list_masks_cells_n, list_patch_ids_n



def generate_final_dataset(he_data, geo_df_nuclei, geo_df_cells, nb_cell_types, patch_df, num_processes, he_patch_width, patch_cell_pairs_idx):

    list_masks_nuclei = []
    list_masks_cells = []
    list_patches = []
    list_patch_ids = []
    skipped_count = 0
    
    def joblib_handling_patch(p2c_idx):
        return handling_patch(p2c_idx, patch_df, geo_df_nuclei, geo_df_cells, he_data, nb_cell_types, he_patch_width)

    # Multiprocessing
    print("Processing patches...")
    results = Parallel(n_jobs=num_processes, backend='threading')(delayed(joblib_handling_patch)(p2c_idx) for p2c_idx in tqdm(patch_cell_pairs_idx))

    for patches, masks_nuclei, masks_cells, patch_ids in results:
        if not patches and not masks_nuclei and not masks_cells:
            skipped_count += 1  # Count the skipped patches
        else:
            list_patches.extend(patches)
            list_masks_nuclei.extend(masks_nuclei)
            list_masks_cells.extend(masks_cells)
            list_patch_ids.extend(patch_ids)

    print(f"Number of skipped patches: {skipped_count} over {skipped_count+len(list_patches)}")
    return list_patches, list_masks_nuclei, list_masks_cells, list_patch_ids



def checking_results(list_patches, list_masks_nuclei, list_masks_cells, list_patch_ids, check_dir, N_checks):

    N_checks = min(N_checks, len(list_patches))
    random_patch_idx = random.sample(range(len(list_patches)), N_checks)
    
    for num in tqdm(random_patch_idx):

        plt.figure()
        
        # Plot H&E patch
        plt.subplot(1,3,1)
        plt.imshow(list_patches[num])
        
        # Plot H&E patch with mask_nuclei
        plt.subplot(1,3,2)
        plt.imshow(list_patches[num])
        mask_nuclei = list_masks_nuclei[num][:,:,0]
        plt.imshow(mask_nuclei, cmap='jet', alpha=np.where(mask_nuclei > 0, 0.5, 0))  # alpha=0 for background (0 values)

        # Plot H&E patch with mask_cells
        plt.subplot(1,3,3)
        plt.imshow(list_patches[num])
        mask_cells = list_masks_cells[num]
        plt.imshow(mask_cells, cmap='jet', alpha=np.where(mask_cells > 0, 0.5, 0))
        
        plt.savefig(os.path.join(check_dir, f"patch{list_patch_ids[num]}.png"), bbox_inches='tight')
        plt.close()



def save_large_array(file_path, data, chunk_size=1000):
    """
    Save a large array incrementally using open_memmap to create a proper .npy file.

    Args:
        file_path (str): Path to the output .npy file.
        data (list or np.ndarray): The array of data to save.
        chunk_size (int): Number of slices to save per chunk (along the first axis).
    """
    print(f"-> Saving array incrementally to {file_path}...")

    # Ensure the data is a NumPy array
    data = np.array(data) if not isinstance(data, np.ndarray) else data
    shape = data.shape  # Full shape of the array
    dtype = data.dtype  # Data type of the array

    # Create a writable memory-mapped .npy file
    memmap_array = np.lib.format.open_memmap(
        file_path, mode='w+', dtype=dtype, shape=shape
    )

    # Write data in chunks along the first axis
    for i in tqdm(range(0, shape[0], chunk_size), unit="chunk"):
        chunk = data[i:i + chunk_size]  # Extract the current chunk
        memmap_array[i:i + chunk.shape[0]] = chunk  # Write the chunk to the file

    # Clean up by deleting references
    del memmap_array
    del data




def save_sparse_3d_array_hw(file_path, data, mask_shape, chunk_size=1000):
    """
    Save a list of 3D masks (height, width, channels) as multiple sparse .npz files incrementally.

    Args:
        file_path (str): Path to save the .npz file.
        data (list or ndarray): List or array of 3D masks (height, width, channels).
        mask_shape (tuple): Original shape of the mask (height, width, channels).
        chunk_size (int): Number of masks to process per chunk.
    """
    print(f"-> Saving sparse 3D masks to {file_path} incrementally...")

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



def save_sparse_2d_masks(file_path, data, chunk_size=1000):
    """
    Save a list of 2D masks (height * width) as a sparse .npz file incrementally.

    Args:
        file_path (str): Path to save the .npz file.
        data (list or ndarray): List or array of 2D masks (height, width).
        chunk_size (int): Number of masks to process per chunk.
    """
    print(f"-> Saving sparse 2D masks to {file_path} incrementally...")


    for i in tqdm(range(0, len(data), chunk_size), unit="chunk"):
        chunk = data[i:i + chunk_size]  # Get current chunk

        # Flatten each 2D mask into a 1D array and convert to sparse format
        sparse_chunk = [csr_matrix(mask.flatten()) for mask in chunk]

        # Combine the chunk into a single sparse matrix
        sparse_matrix = vstack(sparse_chunk)
        chunk_file_path = f"{file_path}_chunk_{i // chunk_size}.npz"
        save_npz(chunk_file_path, sparse_matrix)

        # Free memory
        del chunk, sparse_chunk, sparse_matrix
        gc.collect()




def main(args):

    print(f"\n ===== PROCESSING SLIDE {args.slide_id} =====")
    
    # Load ST data
    print("\n## Loading data ##")
    sdata_path = os.path.join(args.sdata_folder, f"sdata_{args.slide_id}.zarr")
    sdata = sd.read_zarr(sdata_path)
    print(sdata)

    # Get cell indexes and generate HE patches
    print("\n## Get final dataset ##")
    he, geo_df_nuclei, geo_df_cells, patch_df = prepare_sdata(sdata, args.he_patch_width, args.he_patch_overlap)
    nb_cell_types = 5  # same number of categories than the CellVit initially pre-trained by the authors (excluding background)

    # Correct geometries
    geo_df_nuclei['geometry'] = geo_df_nuclei['geometry'].apply(correct_geometry)
    geo_df_cells['geometry'] = geo_df_cells['geometry'].apply(correct_geometry)

    # Get the indices of the patches and corresponding cells that intersect
    patch_cell_pairs_idx = patchs_cells_intersections(patch_df, geo_df_nuclei)
    print(f"Processing {len(patch_cell_pairs_idx)} patches over a total of {len(patch_df)} patches.")

    # Preload H&E data into memory
    print("Preloading H&E data into memory...")
    he_data = he.values 

    if len(patch_cell_pairs_idx) <= 50000:

        # Get final dataset in the Pannuke format
        list_patches, list_masks_nuclei, list_masks_cells, list_patch_ids = generate_final_dataset(he_data, geo_df_nuclei, geo_df_cells, nb_cell_types, patch_df, args.num_processes, args.he_patch_width, patch_cell_pairs_idx)
        list_types = ["Breast" for t in range(len(list_patches))]  # Fake tissue types

        # Visualize slide results
        output_dir_slide = os.path.join(args.output_dir, args.slide_id)
        os.makedirs(output_dir_slide, exist_ok=True)
        print("\n## Get checking ##")
        if args.checking=='yes':
            check_dir = os.path.join(output_dir_slide, "checking")
            os.makedirs(check_dir, exist_ok=True)
            checking_results(list_patches, list_masks_nuclei, list_masks_cells, list_patch_ids, check_dir, args.N_checks)
        
        # Clean memory removing all the non-necessary data
        del sdata, he, geo_df_nuclei, geo_df_cells, patch_df
        gc.collect()

        mask_shape = (args.he_patch_width, args.he_patch_width, nb_cell_types+1)  # +1 for the cell_id

        # Save
        print("\n## Saving results ##")
        save_large_array(os.path.join(output_dir_slide, f"images.npy"), list_patches)
        save_sparse_3d_array_hw(os.path.join(output_dir_slide, f"masks"), list_masks_nuclei, mask_shape)
        save_sparse_2d_masks(os.path.join(output_dir_slide, f"masks_cells"), list_masks_cells)
        save_large_array(os.path.join(output_dir_slide, f"types.npy"), list_types)
        save_large_array(os.path.join(output_dir_slide, f"patch_ids.npy"), list_patch_ids)
        
        print("\n## Done ##")
    
    else:
        print("[INFO] The number of patches is too large to process in one go. We will process the patches in chunks.")

        # Split the patch-cell pairs into chunks
        chunk_size = 40000
        patch_cell_pairs_chunks = [patch_cell_pairs_idx[i:i + chunk_size] for i in range(0, len(patch_cell_pairs_idx), chunk_size)]

        for i, patch_cell_pairs_chunk in enumerate(patch_cell_pairs_chunks):

            print(f"\n----- Processing chunk {i + 1}/{len(patch_cell_pairs_chunks)} for slide {args.slide_id} -----")
            list_patches, list_masks_nuclei, list_masks_cells, list_patch_ids = generate_final_dataset(he_data, geo_df_nuclei, geo_df_cells, nb_cell_types, patch_df, args.num_processes, args.he_patch_width, patch_cell_pairs_chunk)
            list_types = ["Breast" for t in range(len(list_patches))]  # Fake tissue types

            # Visualize slide results
            output_dir_slide = os.path.join(args.output_dir, args.slide_id)
            os.makedirs(output_dir_slide, exist_ok=True)
            print("\n## Get checking ##")
            if args.checking=='yes':
                check_dir = os.path.join(output_dir_slide, f"checking{i}")
                os.makedirs(check_dir, exist_ok=True)
                checking_results(list_patches, list_masks_nuclei, list_masks_cells, list_patch_ids, check_dir, args.N_checks)

            mask_shape = (args.he_patch_width, args.he_patch_width, nb_cell_types+1)  # +1 for the cell_id

            # Save
            print("\n## Saving results ##")
            save_large_array(os.path.join(output_dir_slide, f"images_chunk_{i}.npy"), list_patches)
            save_sparse_3d_array_hw(os.path.join(output_dir_slide, f"masks_chunk_{i}"), list_masks_nuclei, mask_shape)
            save_sparse_2d_masks(os.path.join(output_dir_slide, f"masks_cells_chunk_{i}"), list_masks_cells)
            save_large_array(os.path.join(output_dir_slide, f"types_chunk_{i}.npy"), list_types)
            save_large_array(os.path.join(output_dir_slide, f"patch_ids_chunk_{i}.npy"), list_patch_ids)

            # Clean memory removing all the non-necessary data
            del list_patches, list_masks_nuclei, list_masks_cells, list_patch_ids, list_types
            gc.collect()

            print("\n## Done ##")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get patches for checking alignment")
    
    # Data paths
    parser.add_argument("--slide_id", type=str, default='heart_s0', help="Slide ID")
    parser.add_argument("--sdata_folder", type=str, default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/sdata_final", help="Path to sdata folder")

    # Dataset building
    parser.add_argument("--he_patch_width", type=int, default=256, help="Patch width for H&E images")
    parser.add_argument("--he_patch_overlap", type=int, default=64, help="Patch overlap for H&E images")   
    parser.add_argument("--num_processes", type=int, default=10, help="Number of processes for multiprocessing")
    
    # Save results
    parser.add_argument("--output_dir", type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/patches_xenium', help="Output directory")
    parser.add_argument("--checking", type=str, default='yes', help="Save also result visualization, choose 'yes' or 'no'")
    parser.add_argument("--N_checks", type=int, default=50, help="Number of patches to check")
    
    args = parser.parse_args()
    main(args)

