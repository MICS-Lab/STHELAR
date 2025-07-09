"""
Generating patches with corresponding masks in the same format as the Pannuke dataset

!!!! 
WARNING: 
If you do not need to have mask in the Pannuke format, you can use the script shortcut_build_dataset.py instead of this one.
The shortcut_build_dataset.py script will generate the patches and masks directly in the format used by CellViT, which is much faster and easier to use. 
Also annexe files needed for the CellViT training will also already be generated. And everything will be already zipped.
Thus using the shortcut saves a lot of time (all preprocessing in CellViT part will be already done).
!!!!
"""

import argparse
import os
from tqdm import tqdm
import numpy as np
import random
import json
from joblib import Parallel, delayed
import pandas as pd
import gc
from scipy.sparse import csr_matrix, vstack
from scipy.sparse import save_npz
import zipfile

import spatialdata as sd
from skimage.draw import polygon
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.validation import explain_validity
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sopa._sdata import get_spatial_image, to_intrinsic



def get_zip_file_names(zip_path):

    already_patch_ids = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        for file in z.namelist():
            name, _ = os.path.splitext(os.path.basename(file))
            # Skip hidden or invalid files
            if name.startswith('._') or not name.isdigit():
                continue
            name = int(name)  # Convert name to int to get the same type as patch_id in patch_df
            already_patch_ids.append(name)

    return already_patch_ids



def prepare_sdata(sdata, label2cat, cat2idx, he_patches_selection, annots_table, annots_column, already_patch_ids):

    he = get_spatial_image(sdata, "he")  # Gets a DataArray from a SpatialData object (if the image has multiple scale, the `scale0` is returned)
    geo_df = to_intrinsic(sdata, sdata.shapes['nucleus_boundaries'], he)

    nb_cell_types = len(cat2idx)

    patch_df = sdata.shapes['he_patches'].copy()
    print(f"Number of patches before selection is {len(patch_df)}.")
    patch_df = patch_df[(patch_df[he_patches_selection[0]]>=he_patches_selection[1]) & (patch_df[he_patches_selection[0]]<=he_patches_selection[2])]
    print(f"Number of patches selected is {len(patch_df)} using {he_patches_selection} for he_patches_selection.")

    if already_patch_ids: # filter patch_df, keeping only the patches that are present in already_patch_ids
        patch_df = patch_df[patch_df['patch_id'].isin(already_patch_ids)]
        print(f"Number of patches selected is {len(patch_df)} using already_patch_ids.")

    adata_obs = sdata.tables[annots_table].obs.copy()
    adata_obs['cat'] = adata_obs[annots_column].map(label2cat)
    adata_obs['cat_idx'] = adata_obs['cat'].map(cat2idx)

    return he, geo_df, nb_cell_types, patch_df, adata_obs



def patchs_cells_intersections(patch_df, geo_df):

    # Returns the indices of the patches and corresponding cells that intersect
    patch_idx, cell_idx = geo_df.geometry.sindex.query(patch_df.geometry, predicate=None)

    # Group the cell data by patch index
    patch_idx_2_cells_idx = pd.Series(geo_df.index[cell_idx]).groupby(patch_idx).apply(list)
    patch_cell_pairs_idx = [(patch_idx, patch_idx_2_cells_idx[patch_idx]) for patch_idx in patch_idx_2_cells_idx.index]

    # Return the patch index and their corresponding cell indices in the same order
    return patch_cell_pairs_idx



def rasterize(mask, cell_patch_intersection, xy_min, cat_idx, nb_cell_types, count):

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
    mask[rr, cc, cat_idx] = count
    mask[rr, cc, nb_cell_types] = cat_idx+1

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



def handling_patch(p2c_idx, patch_df, geo_df, he_data, cell2cat_idx, nb_cell_types, skip_unknown, unknown_idx):

    patch_idx = p2c_idx[0]

    patch_id = patch_df['patch_id'].iloc[patch_idx]
    patch_bounds = patch_df.bounds.iloc[patch_idx]

    list_patches_n = []
    list_masks_n = []
    list_patch_ids_n = []

    ymin, ymax, xmin, xmax = int(patch_bounds['miny']), int(patch_bounds['maxy']), int(patch_bounds['minx']), int(patch_bounds['maxx'])
    he_patch_width = ymax - ymin
    patch = box(xmin, ymin, xmax, ymax)
    
    # Create an empty mask for the current patch
    mask = np.zeros((ymax-ymin, xmax-xmin, nb_cell_types+1), dtype=np.int32)
    skip = 0
    has_unknown_cell = False
    
    # Iterate over cells in the patch
    count = 0

    for cell_idx in p2c_idx[1]:

        cell = geo_df.loc[cell_idx].geometry
        
        if not isinstance(cell, (Polygon, MultiPolygon)):
            print(f"Unexpected type for cell: {type(cell)}, value: {cell}")
            continue

        cell = correct_geometry(cell)

        if not cell.is_valid:
            # Handle invalid geometry
            print(f"Invalid geometry for {cell_idx}")
            continue
    
        if patch.intersects(cell):
            cell_patch_intersection = cell.intersection(patch)

            if not cell_patch_intersection.is_empty:
                #cat_idx = adata_obs[adata_obs.cell_id==cell_idx]['cat_idx'].values[0]  # not efficient, far too long ! using cell2cat_idx instead
                cat_idx = cell2cat_idx.get(cell_idx, None)
                # Check for unknown cell type
                if skip_unknown == 'yes' and cat_idx == unknown_idx:
                    has_unknown_cell = True
                    break
                count += 1
                try:
                    mask = rasterize(mask, cell_patch_intersection, (xmin, ymin), cat_idx, nb_cell_types, count)  # Update the mask to include the intersection area of the current cell
                except ValueError as e:
                    print("Error rasterizing cell:", e)
                    skip = 1
                    continue      

    # Skip adding patch if there is an unknown cell
    if has_unknown_cell:
        return [], []
    
    if not np.all(mask == 0): #and skip == 0:    # not np.all(np.delete(mask, [unknown_idx, nb_cell_types], axis=-1) == 0)
        img = np.transpose(he_data[:, ymin:ymax, xmin:xmax], (1, 2, 0))
        if img.shape == (he_patch_width, he_patch_width, 3) and mask.shape == (he_patch_width, he_patch_width, nb_cell_types+1):
            # Append the patch and mask to the lists
            list_patches_n.append(img)
            list_masks_n.append(mask)
            list_patch_ids_n.append(patch_id)
        else:
            print("Patch and mask shapes do not match the expected shape -- skipping")
    else:
        print("Mask with only unknown cells -- skipping")
    
    return list_patches_n, list_masks_n,  list_patch_ids_n



def generate_final_dataset(he_data, geo_df, nb_cell_types, patch_df, cell2cat_idx, num_processes, skip_unknown, unknown_idx, patch_cell_pairs_idx):

    list_masks = []
    list_patches = []
    list_patch_ids = []
    skipped_count = 0

    def joblib_handling_patch(p2c_idx):
        return handling_patch(p2c_idx, patch_df, geo_df, he_data, cell2cat_idx, nb_cell_types, skip_unknown, unknown_idx)

    # Multiprocessing
    print("Processing patches...")
    results = Parallel(n_jobs=num_processes, backend='threading')(delayed(joblib_handling_patch)(p2c_idx) for p2c_idx in tqdm(patch_cell_pairs_idx))

    for patches, masks, patch_ids in results:
        if not patches and not masks:
            skipped_count += 1  # Count the skipped patches
        else:
            list_patches.extend(patches)
            list_masks.extend(masks)
            list_patch_ids.extend(patch_ids)

    print(f"Number of skipped patches: {skipped_count} over {skipped_count+len(list_patches)}")
    return list_patches, list_masks, list_patch_ids



def checking_results(list_patches, list_masks, list_patch_ids, check_dir, cat2color, cat2idx, N_checks):

    N_checks = min(N_checks, len(list_patches))
    random_patch_idx = random.sample(range(len(list_patches)), N_checks)
    
    idx2color = {idx+1: cat2color[cat]+[0.5*255] for cat, idx in cat2idx.items()}
    idx2color[0] = [0, 0, 0, 0]  # Background color
    
    for num in tqdm(random_patch_idx):

        # Create legend handles and labels
        legend_handles = []
        legend_labels = []
        for cat, color in cat2color.items():
            legend_handles.append(Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=[c/255 for c in color], markersize=10))
            legend_labels.append(cat)

        plt.figure()
        
        # Plot H&E patch
        plt.subplot(1,2,1)
        plt.imshow(list_patches[num])
        
        # Plot H&E patch with mask
        plt.subplot(1,2,2)
        
        plt.imshow(list_patches[num])
        
        mask = list_masks[num][:,:,:-1]
        mask_with_bg = np.concatenate((np.zeros((mask.shape[0], mask.shape[1], 1), dtype=mask.dtype), mask), axis=-1)
        mask_idx = np.argmax(mask_with_bg, axis=-1)
        mask_plotting = np.array([[idx2color[idx] for idx in row] for row in mask_idx]) / 255    # / 255 to Normalize colors
        plt.imshow(mask_plotting)
        plt.legend(handles=legend_handles, labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
        
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
    



def main(args):
    
    print(f"\n ===== PROCESSING SLIDE {args.slide_id} =====")

    # Get names of already computed images
    if args.already_images_path:
        print("\n## Loading already computed images ##")
        already_patch_ids = get_zip_file_names(args.already_images_path)
        print(f"Number of already computed patches: {len(already_patch_ids)}")

    # Load ST data
    print("\n## Loading data ##")
    sdata = sd.read_zarr(args.sdata_path)
    for table_name in ['features_cellvit', 'features_phikonv2', 'features_vit_google', 'table_cells', 'table_combined', 'table_nuclei', 'table_scvi']:
        if table_name != args.annots_table:
            del sdata.tables[table_name]
            gc.collect()
    print(sdata)

    # Load dictionaries
    print("\n## Loading annotations dictionaries ##")
    with open(args.label2cat_path, 'r') as f:
        label2cat = json.load(f)
    with open(args.cat2idx_path, 'r') as f:
        cat2idx = json.load(f)
    with open(args.cat2color_path, 'r') as f:
        cat2color = json.load(f)
    print("cat2idx dictionary:", cat2idx)

    # Get cell indexes and generate HE patches
    print("\n## Get final dataset ##")
    he, geo_df, nb_cell_types, patch_df, adata_obs = prepare_sdata(sdata, label2cat, cat2idx, args.he_patches_selection, args.annots_table, args.annots_column, already_patch_ids)
    unknown_idx = None
    if args.skip_unknown == 'yes':
        unknown_idx = cat2idx[args.unknown_label]
        print(f"Unknown cat index: {unknown_idx}")

    # Correct geometries
    geo_df['geometry'] = geo_df['geometry'].apply(correct_geometry)

    # Get the indices of the patches and corresponding cells that intersect
    patch_cell_pairs_idx = patchs_cells_intersections(patch_df, geo_df)

    print(f"Processing {len(patch_cell_pairs_idx)} patches over a total of {len(patch_df)} patches.")

    # Preload H&E data into memory
    print("Preloading H&E data into memory...")
    he_data = he.values

    # Get cell2cat_idx dictionary
    cell2cat_idx = dict(zip(adata_obs['cell_id'], adata_obs['cat_idx']))

    if len(patch_cell_pairs_idx) <= 50000:
    
        # Get final dataset in the Pannuke format and save
        list_patches, list_masks, list_patch_ids = generate_final_dataset(he_data, geo_df, nb_cell_types, patch_df, cell2cat_idx, args.num_processes, args.skip_unknown, unknown_idx, patch_cell_pairs_idx)
        list_types = [args.tissue_type for t in range(len(list_patches))]

        # Visualize slide results
        print("\n## Get checking ##")
        os.makedirs(args.output_dir, exist_ok=True)
        if args.checking=='yes':
            check_dir = os.path.join(args.output_dir, "checking")
            os.makedirs(check_dir, exist_ok=True)
            checking_results(list_patches, list_masks, list_patch_ids, check_dir, cat2color, cat2idx, args.N_checks)
        
        # Save patch metrics in csv
        patch_metrics = patch_df[patch_df['patch_id'].isin(list_patch_ids)]
        patch_metrics = patch_metrics[['patch_id', 'Dice', 'Jaccard', 'bPQ']].set_index('patch_id').reindex(list_patch_ids).reset_index()
        patch_metrics.to_csv(os.path.join(args.output_dir, f"patch_metrics.csv"), index=False)

        # Clean memory removing all the non-necessary data
        del sdata, he, geo_df, patch_df, adata_obs, he_data
        gc.collect()

        he_patch_width = list_patches[0].shape[0]
        print(f"HE patch width: {he_patch_width}")
        mask_shape = (he_patch_width, he_patch_width, nb_cell_types+1)
        
        # Save results
        print("\nSaving results...")
        if not args.already_images_path: # Save images only if args.already_images_path is None
            save_large_array(os.path.join(args.output_dir, f"images.npy"), list_patches)
        save_sparse_3d_array_hw(os.path.join(args.output_dir, f"masks"), list_masks, mask_shape)
        save_large_array(os.path.join(args.output_dir, f"patch_ids.npy"), list_patch_ids)
        save_large_array(os.path.join(args.output_dir, f"types.npy"), list_types)
        
        print("\n## Done ! ##")

    else:
        print("[INFO] The number of patches is too large to process in one go. We will process the patches in chunks.")

        # Split the patch-cell pairs into chunks
        chunk_size = 40000
        patch_cell_pairs_chunks = [patch_cell_pairs_idx[i:i + chunk_size] for i in range(0, len(patch_cell_pairs_idx), chunk_size)]

        for i, patch_cell_pairs_chunk in enumerate(patch_cell_pairs_chunks):

            print(f"\n----- Processing chunk {i + 1}/{len(patch_cell_pairs_chunks)} for slide {args.slide_id} -----")
            list_patches, list_masks, list_patch_ids = generate_final_dataset(he_data, geo_df, nb_cell_types, patch_df, cell2cat_idx, args.num_processes, args.skip_unknown, unknown_idx, patch_cell_pairs_chunk)
            list_types = [args.tissue_type for t in range(len(list_patches))]

            # Visualize slide results
            print("\n## Get checking ##")
            os.makedirs(args.output_dir, exist_ok=True)
            if args.checking=='yes':
                check_dir = os.path.join(args.output_dir, "checking")
                os.makedirs(check_dir, exist_ok=True)
                checking_results(list_patches, list_masks, list_patch_ids, check_dir, cat2color, cat2idx, args.N_checks)
            
            # Save patch metrics in csv
            patch_metrics = patch_df[patch_df['patch_id'].isin(list_patch_ids)]
            patch_metrics = patch_metrics[['patch_id', 'Dice', 'Jaccard', 'bPQ']].set_index('patch_id').reindex(list_patch_ids).reset_index()
            patch_metrics.to_csv(os.path.join(args.output_dir, f"patch_metrics_chunk_{i}.csv"), index=False)

            he_patch_width = list_patches[0].shape[0]
            print(f"HE patch width: {he_patch_width}")
            mask_shape = (he_patch_width, he_patch_width, nb_cell_types+1)

            # Save results
            print("\nSaving results...")
            if not args.already_images_path: # Save images only if args.already_images_path is None
                save_large_array(os.path.join(args.output_dir, f"images_chunk_{i}.npy"), list_patches)
            save_sparse_3d_array_hw(os.path.join(args.output_dir, f"masks_chunk_{i}"), list_masks, mask_shape)
            save_large_array(os.path.join(args.output_dir, f"patch_ids_chunk_{i}.npy"), list_patch_ids)
            save_large_array(os.path.join(args.output_dir, f"types_chunk_{i}.npy"), list_types)

            # Clean memory removing all the non-necessary data
            del list_patches, list_masks, list_patch_ids, list_types
            gc.collect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get cell type dataset using ST data")
    
    # Data paths
    parser.add_argument("--sdata_path", type=str, default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/sdata_final/sdata_heart_s0.zarr", help="Path to sdata")
    parser.add_argument("--label2cat_path", type=str, default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/annots/annot_dicts_ct_1/label2cat.json", help="Path to the json file with dictionary of final labels to training categories")
    parser.add_argument("--cat2idx_path", type=str, default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/annots/annot_dicts_ct_1/cat2idx.json", help="Path to the json file with dictionary of training categories to index")
    parser.add_argument("--cat2color_path", type=str, default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/annots/annot_dicts_ct_1/cat2color.json", help="Path to the json file with dictionary of training categories to colors")
    parser.add_argument("--tissue_type", type=str, default='Heart', help="Tissue type with a capital letter")

    # Dataset building
    parser.add_argument("--he_patches_selection", type=tuple, default=('Jaccard', 0.0, 1.0), help="Metric (Dice, Jaccard, bPQ) and interval for filtering he_patches")
    parser.add_argument("--annots_table", type=str, default="table_cells", help="Table to use for annotations (Choose table_cells or table_nuclei or table_combined)")
    parser.add_argument("--annots_column", type=str, default="final_label", help="Column in the table to use for annotations (Choose final_label or final_label_combined)")
    parser.add_argument("--num_processes", type=int, default=10, help="Number of processes for multiprocessing")
    parser.add_argument("--skip_unknown", type=str, default='no', help="Skip patch with at least one unknown cell type, choose 'no' for no skipping and 'yes' otherwise")
    parser.add_argument("--unknown_label", type=str, default='Unknown', help="Label for unknown cell type")
    
    parser.add_argument("--already_images_path", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium/heart_s0/images.zip", 
                        help="Path to the zip file containing the images from check align => images will not be computed again and also only the patch_ids in the zip file will be considered / otherwise set to None")
    
    # Save results
    parser.add_argument("--slide_id", type=str, default='heart_s0', help="Slide ID")
    parser.add_argument("--output_dir", type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/ds_slides_cat/ct_1/heart_s0', help="Output directory")
    parser.add_argument("--checking", type=str, default='yes', help="Save also result visualization, choose 'yes' or 'no'")
    parser.add_argument("--N_checks", type=int, default=50, help="Number of patches to check")
    
    args = parser.parse_args()
    main(args)

