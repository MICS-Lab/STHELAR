"""
Generating images.zip, labels.zip, cell_count.csv and types.csv directly from one or many SpatialData slides.
=> We directly have the dataset in the 'CellViT' format, avoiding to compute first the dataset in the 'PanNuke' format.
We can also choose the resolution we want to use for the H&E images (20x or 40x).
"""

import argparse, json, io, os, gc, zipfile
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import spatialdata as sd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.validation import explain_validity
from skimage.draw import polygon
from PIL import Image
from scipy.sparse import csr_matrix, save_npz
from joblib import Parallel, delayed
from datatree import DataTree

from sopa._sdata import get_spatial_image, to_intrinsic
from sopa.segmentation import Patches2D


def promote_20x_to_default(sdata, key: str = "he") -> None:
    """
    Re-wire sdata in-memory so that _return_element(sdata, key, …, as_spatial_image=True) in SOPA returns the 20 × (level-1) image instead of 40 ×.
    """
    img_tree: DataTree = sdata.images[key]              # original multiscale tree

    # Build a wrapper whose only child is 'scale0' -> original 'scale1'
    wrapper = DataTree(name=key)
    wrapper["scale0"] = img_tree["scale1"]      # just a pointer, no data copy
    wrapper.attrs.update(img_tree.attrs)        # keep transforms & metadata

    # Overwrite the entry so future look-ups hit our wrapper.
    sdata.images[key] = wrapper



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



def patchs_cells_intersections(patch_df, geo_df):

    # Returns the indices of the patches and corresponding cells that intersect
    patch_idx, cell_idx = geo_df.geometry.sindex.query(patch_df.geometry, predicate=None)

    # Group the cell data by patch index
    patch_idx_2_cells_idx = pd.Series(geo_df.index[cell_idx]).groupby(patch_idx).apply(list)
    patch_cell_pairs_idx = [(patch_idx, patch_idx_2_cells_idx[patch_idx]) for patch_idx in patch_idx_2_cells_idx.index]

    # Return the patch index and their corresponding cell indices in the same order
    return patch_cell_pairs_idx




def rasterize(mask, cell_patch_intersection, xy_min, cat_idx, count, nb_cat, cell_id):

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

    if rr.size == 0:                       # <-- nothing inside the patch
        return False
    
    mask[rr, cc, cat_idx] = count
    mask[rr, cc, nb_cat] = int_cell_id(cell_id)
    return True




def remap_label(pred):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    """
    pred_id = list(np.unique(pred))
    
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred




def checking_results(img, mask, patch_id, cat2color, idx2color, check_dir):

    # Create legend handles and labels
    legend_handles = []
    legend_labels = []
    for cat, color in cat2color.items():
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=[c/255 for c in color], markersize=10))
        legend_labels.append(cat)

    plt.figure()
    
    # Plot H&E patch
    plt.subplot(1,2,1)
    plt.imshow(img)
    
    # Plot H&E patch with mask
    plt.subplot(1,2,2)

    plt.imshow(img)
    # Build an RGBA overlay where background stays transparent
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)  # H×W×4
    for idx, rgba in idx2color.items():
        if idx == 0:                       # keep background fully transparent
            continue
        overlay[mask == idx] = rgba        # Fill the overlay with colors from idx2color

    plt.imshow(overlay, interpolation="none")
    plt.legend(handles=legend_handles, labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.savefig(os.path.join(check_dir, f"patch_{patch_id}.png"), bbox_inches='tight')
    plt.close()



def process_patch(pair: Tuple[int, List[int]], patch_df, geo_df, he_data,
                  cell2cat: Dict[int, int], nb_cat: int, 
                  slide_id: str, tissue: str, cat_names: List[str]):

    p_idx, cell_idxs = pair
    p_id = patch_df['patch_id'].iloc[p_idx]
    patch_bounds = patch_df.bounds.iloc[p_idx]
    ymin, ymax, xmin, xmax = int(patch_bounds['miny']), int(patch_bounds['maxy']), int(patch_bounds['minx']), int(patch_bounds['maxx'])
    bnd = [xmin, ymin, xmax, ymax]  # patch bounding box
    patch_w = ymax - ymin

    patch_box = box(xmin, ymin, xmax, ymax)
    mask = np.zeros((patch_w, patch_w, nb_cat+1), np.int32)

    inst = 1
    exact_counts = {name: 0 for name in cat_names}

    for c_idx in cell_idxs:
        cell = geo_df.loc[c_idx].geometry
        if not isinstance(cell, (Polygon, MultiPolygon)):
            print(f"Unexpected type for cell: {type(cell)}, value: {cell}")
            continue
        cell = correct_geometry(cell)
        if not cell.is_valid:
            print(f"Invalid geometry for {c_idx}")
            continue
        if cell.is_empty or not patch_box.intersects(cell):
            continue
        cat_idx = cell2cat.get(c_idx)
        inter = cell.intersection(patch_box)
        try:
            success = rasterize(mask, inter, (xmin, ymin), cat_idx, inst, nb_cat, c_idx)
            if not success:
                continue
            exact_counts[cat_names[cat_idx]] += 1
            inst += 1
        except ValueError as e:
            print("Error rasterizing cell:", e)
            continue

    if not mask.any():
        return None

    # Build the image patch
    img = np.transpose(he_data[:, ymin:ymax, xmin:xmax], (1, 2, 0)).astype(np.uint8)
    if img.shape != (patch_w, patch_w, 3) or mask.shape != (patch_w, patch_w, nb_cat+1):  # corrupt geometry
        print(f"Patch {p_id} has corrupt geometry, skipping.")
        return None
    img = Image.fromarray(img.astype(np.uint8))

    # Build the instance map
    inst_map = np.zeros((patch_w, patch_w), np.int32)
    offset = 0
    for c in range(nb_cat):
        layer_res = remap_label(mask[:, :, c])
        inst_map = np.where(layer_res != 0, layer_res + offset, inst_map)
        offset += np.max(layer_res)
    inst_map = remap_label(inst_map)

    # Build the type map
    type_map = np.zeros((256, 256)).astype(np.int32)
    for c in range(nb_cat):
        layer_res = ((c + 1) * np.clip(mask[:, :, c], 0, 1)).astype(np.int32)
        type_map = np.where(layer_res != 0, layer_res, type_map)

    return p_id, img, inst_map, type_map, mask[:, :, -1], tissue, exact_counts, bnd




def process_slide(slide_id, tissue, cfg, label2cat, cat2idx, cat2color, idx2color, cat_names, nb_cat):
    
    # Loading sdata, keeping only annotation table we need
    sdata = sd.read_zarr(cfg.sdata_dir / f"sdata_{slide_id}.zarr")
    for t in list(sdata.tables):
        if t != cfg.annots_table:
            del sdata.tables[t]
    gc.collect()
    logging.info(sdata)
    print(sdata)

    # Promote 20x to default if needed
    logging.info(f"\nUsing H&E resolution: {cfg.he_resolution}x")
    print(f"\nUsing H&E resolution: {cfg.he_resolution}x")
    resolution_40 = get_spatial_image(sdata, "he").shape
    logging.info(f"40x resolution: {resolution_40}")
    print(f"40x resolution: {resolution_40}")
    if cfg.he_resolution == 20:
        promote_20x_to_default(sdata, key="he")
        resolution_20 = get_spatial_image(sdata, "he").shape
        logging.info(f"20x resolution: {resolution_20}")
        print(f"20x resolution: {resolution_20}")
        logging.info(f"Check dividing 40x by 2 / Diff are : {resolution_40[1] // 2 - resolution_20[1]} and {resolution_40[2] // 2 - resolution_20[2]}")
        print(f"Check dividing 40x by 2 / Diff are : {resolution_40[1] // 2 - resolution_20[1]} and {resolution_40[2] // 2 - resolution_20[2]}")
    elif cfg.he_resolution != 40:
        raise ValueError("Unsupported H&E resolution. Choose 20 or 40.")
    logging.info(f"\nNew sdata: {sdata}")
    print(f"\nNew sdata: {sdata}")

    # Get HE and segmentation data
    he = get_spatial_image(sdata, "he")  # Gets a DataArray from a SpatialData object (if the image has multiple scale, the `scale0` is returned)
    geo_df = to_intrinsic(sdata, sdata.shapes['nucleus_boundaries'], he)
    geo_df['geometry'] = geo_df['geometry'].apply(correct_geometry)

    # Check names of the columns in geo_df
    logging.info(f"Columns in geo_df: {geo_df.columns}")
    print(f"Columns in geo_df: {geo_df.columns}")

    # Build patches
    patches = Patches2D(sdata, 'he', patch_width=cfg.he_patch_width, patch_overlap=cfg.he_patch_overlap)
    patches.write()
    patch_df = sdata.shapes['sopa_patches'].copy()
    sdata.delete_element_from_disk("sopa_patches")
    patch_df['patch_id'] = slide_id + '_' + patch_df.index.astype(str)

    # Get cell2cat idx dictionary
    ad_obs = sdata.tables[cfg.annots_table].obs.copy()
    ad_obs['cat_idx'] = ad_obs[cfg.annots_column].map(label2cat).map(cat2idx)
    cell2cat = dict(zip(ad_obs.cell_id, ad_obs.cat_idx))  

    # Already-present files (avoid duplicates on re-run)
    existing_imgs   = set(cfg.images_zip.namelist())
    existing_labels = set(cfg.labels_zip.namelist())
    existing_masks  = set(cfg.cell_ids_zip.namelist()) 

    # Build spatial index mapping (patch -> cells)
    pairs = patchs_cells_intersections(patch_df, geo_df)
    logging.info(f"Processing {len(pairs)} patches over a total of {len(patch_df)} patches.")
    print(f"Processing {len(pairs)} patches over a total of {len(patch_df)} patches.")

    # Select a random subset of pairs for checking if enabled
    if cfg.checking == 'yes':
        check_dir = cfg.output / 'checking'
        check_dir.mkdir(parents=True, exist_ok=True)
        patch_to_check = random.sample(range(len(pairs)), min(cfg.N_checks, len(pairs)))
        patch_ids_to_check = [pairs[i][0] for i in patch_to_check]
        patch_ids_to_check = [f"{slide_id}_{p_id}" for p_id in patch_ids_to_check]

    # Preload H&E data into memory
    logging.info("Preloading H&E data into memory")
    print("Preloading H&E data into memory")
    he_data = he.values
    
    # Generate patches
    logging.info("\nGenerating patches...")
    print("\nGenerating patches...")
    worker = delayed(process_patch)
    results = Parallel(cfg.num_workers, prefer='threads')(
        worker(pair, patch_df, geo_df, he_data, cell2cat, nb_cat, slide_id, tissue, cat_names)
        for pair in tqdm(pairs))

    logging.info("\nSaving patches to zip files and preparing DataFrames...")
    print("\nSaving patches to zip files and preparing DataFrames...")
    rows_cc, rows_tp, patch_ids = [], [], []
    for res in filter(None, results):
        p_id, img, inst_map, type_map, cell_id_mask, tissue_type, counts, bnd = res
        
        # Checking only cfg.N_checks patches if cfg.checking is enabled
        if cfg.checking == 'yes':
            if p_id in patch_ids_to_check:
                checking_results(img, type_map, p_id, cat2color, idx2color, check_dir)
        
        # Names
        fname = f"{p_id}.png"
        npz_name = fname.replace('.png', '.npz')

        # images.zip
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        if fname not in existing_imgs:
            cfg.images_zip.writestr(fname, buf.getvalue())
            existing_imgs.add(fname)

        # labels.zip
        inst_sp = csr_matrix(inst_map)
        type_sp = csr_matrix(type_map)
        buf = io.BytesIO()
        np.savez(buf,
                inst_map_data=inst_sp.data, inst_map_indices=inst_sp.indices,
                inst_map_indptr=inst_sp.indptr, inst_map_shape=inst_sp.shape,
                type_map_data=type_sp.data, type_map_indices=type_sp.indices,
                type_map_indptr=type_sp.indptr, type_map_shape=type_sp.shape)
        if npz_name not in existing_labels:
            cfg.labels_zip.writestr(npz_name, buf.getvalue())
            existing_labels.add(npz_name)

        # masks_cell_ids_nuclei.zip
        cell_id_sp = csr_matrix(cell_id_mask)
        buf = io.BytesIO()
        save_npz(buf, cell_id_sp)
        if npz_name not in existing_masks:
            cfg.cell_ids_zip.writestr(npz_name, buf.getvalue())
            existing_masks.add(npz_name)

        rows_cc.append({'Image': fname, **counts})
        rows_tp.append({'img': fname, 'type': tissue_type})
        patch_ids.append({'slide_id': slide_id, 'patch_id': fname, 'xmin': bnd[0], 'ymin': bnd[1], 'xmax': bnd[2], 'ymax': bnd[3]})

    return pd.DataFrame(rows_cc), pd.DataFrame(rows_tp), pd.DataFrame(patch_ids)





def main(cfg):
    
    if len(cfg.slides) != len(cfg.slide_tissues):
        raise ValueError("Number of slides must match number of slide tissues.")
    
    # Init logger
    cfg.output.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output / 'get_patches.log'
    print(f"Logging to {log_path}")
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Loading annots dictionaries
    with open(cfg.dict_dir / 'label2cat.json') as f: label2cat = json.load(f)
    with open(cfg.dict_dir / 'cat2idx.json') as f: cat2idx = json.load(f)
    with open(cfg.dict_dir / 'cat2color.json') as f: cat2color = json.load(f)
    cat_names = [k for k, _ in sorted(cat2idx.items(), key=lambda kv: kv[1])]
    nb_cat = len(cat2idx)
    idx2color = {idx+1: cat2color[cat]+[0.5*255] for cat, idx in cat2idx.items()}
    idx2color[0] = [0, 0, 0, 0]  # Background color
    logging.info(f"\nNumber of categories: {nb_cat}")
    print(f"\nNumber of categories: {nb_cat}")
    logging.info(f"Categories: {cat_names}")
    print(f"Categories: {cat_names}")

    # Zip files
    zip_mode = 'a' if (cfg.output.parent / 'images.zip').exists() else 'w'   # append if files already exist
    cfg.images_zip = zipfile.ZipFile(cfg.output.parent / 'images.zip', zip_mode, zipfile.ZIP_DEFLATED)
    zip_mode = 'a' if (cfg.output / 'labels.zip').exists() else 'w'
    cfg.labels_zip = zipfile.ZipFile(cfg.output / 'labels.zip', zip_mode, zipfile.ZIP_DEFLATED)
    zip_mode = 'a' if (cfg.output.parent / 'masks_cell_ids_nuclei.zip').exists() else 'w'
    cfg.cell_ids_zip = zipfile.ZipFile(cfg.output.parent / 'masks_cell_ids_nuclei.zip', zip_mode, zipfile.ZIP_DEFLATED)

    # Process each slide
    all_cc, all_tp, all_pi = [], [], []
    logging.info("Processing slides...")
    print("Processing slides...")
    for i, (slide_id, tissue) in enumerate(zip(cfg.slides, cfg.slide_tissues)):
        logging.info(f"\n\n\n****** Processing slide {i+1}/{len(cfg.slides)}: {slide_id} ({tissue}) ******")
        print(f"\n\n\n****** Processing slide {i+1}/{len(cfg.slides)}: {slide_id} ({tissue}) ******")
        cc, tp, pi = process_slide(slide_id, tissue, cfg, label2cat, cat2idx, cat2color, idx2color, cat_names, nb_cat)
        all_cc.append(cc)
        all_tp.append(tp)
        all_pi.append(pi)
        gc.collect()

    cfg.images_zip.close()
    cfg.labels_zip.close()
    cfg.cell_ids_zip.close()

    # Save cell counts, types, and patch_ids to CSV
    logging.info("\nSaving cell counts, types, and patch IDs to CSV files...")
    print("\nSaving cell counts, types, and patch IDs to CSV files...")

    cc_path = cfg.output / 'cell_count.csv'
    tp_path = cfg.output.parent / 'types.csv'
    pi_path = cfg.output.parent / 'patch_metrics.csv'

    new_cc = pd.concat(all_cc)
    new_tp = pd.concat(all_tp)
    new_pi = pd.concat(all_pi)

    if cc_path.exists():
        new_cc = pd.concat([pd.read_csv(cc_path), new_cc]).drop_duplicates(subset='Image')
    if tp_path.exists():
        new_tp = pd.concat([pd.read_csv(tp_path), new_tp]).drop_duplicates(subset='img')
    if pi_path.exists():
        new_pi = pd.concat([pd.read_csv(pi_path), new_pi]).drop_duplicates(subset='patch_id')

    new_cc.to_csv(cc_path, index=False)
    new_tp.to_csv(tp_path, index=False)
    new_pi.to_csv(pi_path, index=False)

    logging.info("Done.")
    print("Done.")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="HE2CellType dataset builder")
    
    parser.add_argument('--slides', nargs='+', required=True, help='slide IDs')
    parser.add_argument('--slide_tissues', nargs='+', required=True, help='tissue type per slide (SAME ORDER AS SLIDES and WITH FIRST LETTER CAPITALIZED, e.g. "Kidney", "Liver")')
    parser.add_argument('--sdata_dir', type=Path, default=Path('/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/sdata_final/'), help='Directory with final sdata zarr files')
    parser.add_argument('--dict_dir', type=Path, default=Path('/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/annots/annot_dicts_ct_1'), help='Directory with annot dictionaries for given ct dataset')
    parser.add_argument('--output', type=Path, default=Path('/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat_20x/ct_1'), help='Output directory for images.zip, labels.zip, cell_count.csv, types.csv and patches.csv')
    parser.add_argument('--annots_table', type=str, default='table_cells', help='Name of the annotations table in sdata / Choose table_cells or table_nuclei or table_combined')
    parser.add_argument('--annots_column', type=str, default='final_label', help='Name of the column in the annotations table with cell types / Choose final_label or final_label_combined')
    parser.add_argument('--he_resolution', type=int, default=20, help='H&E image resolution between 20x and 40x / Choose 20 or 40')
    parser.add_argument("--he_patch_width", type=int, default=256, help="Patch width for H&E images")
    parser.add_argument("--he_patch_overlap", type=int, default=64, help="Patch overlap for H&E images") 
    parser.add_argument('--num_workers', type=int, default=10, help='Number of parallel workers for patch processing')
    parser.add_argument("--checking", type=str, default='yes', help="Plot few images per slide for checking, choose 'yes' or 'no'")
    parser.add_argument("--N_checks", type=int, default=50, help="Number of patches to check per slide, if checking is enabled")
    
    cfg = parser.parse_args()

    main(cfg)