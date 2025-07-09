"""
Adding cells boundaries and cells table (default from Xenium data) to the given sdata
"""

import os
import argparse
import pandas as pd
import spatialdata as sd
import spatialdata_io
import itertools


def open_xenium(sdata_ini_path, img_key):

    if img_key=='morphology_mip':
        sdata_ini = spatialdata_io.xenium(path=sdata_ini_path,
                                          n_jobs=1, transcripts=True, nucleus_boundaries=True, cells_boundaries=True,
                                          morphology_mip=True, morphology_focus=False,
                                          cells_table=False, aligned_images=False, nucleus_labels=False, cells_labels=False, cells_as_circles=False)
    
    elif img_key=='morphology_focus':
        sdata_ini = spatialdata_io.xenium(path=sdata_ini_path,
                                          n_jobs=1, transcripts=True, nucleus_boundaries=True, cells_boundaries=True,
                                          morphology_mip=False, morphology_focus=True,
                                          cells_table=False, aligned_images=False, nucleus_labels=False, cells_labels=False, cells_as_circles=False)
        
    return sdata_ini



def compare_nucleus_boundaries(sdata_ini, sdata_final):

    geometries_ini = set(sdata_ini.shapes['nucleus_boundaries']['geometry'])
    geometries_final = set(sdata_final.shapes['nucleus_boundaries']['geometry'])

    geometries_only_in_ini = geometries_ini - geometries_final
    geometries_only_in_final = geometries_final - geometries_ini

    if geometries_only_in_ini:
        print(f"Geometries in 'sdata_ini' but not in 'sdata_final': {len(geometries_only_in_ini)}")
    else:
        print("All geometries in 'sdata_ini' are present in 'sdata_final'.")

    if geometries_only_in_final:
        print(f"Geometries in 'sdata_final' but not in 'sdata_ini': {len(geometries_only_in_final)}")
    else:
        print("All geometries in 'sdata_final' are present in 'sdata_ini'.")



def cell_ids_matching(sdata_ini, sdata_final):

    merged_gdf = pd.merge(sdata_ini.shapes['nucleus_boundaries'].reset_index(names='cell_id_ini'), 
                      sdata_final.shapes['nucleus_boundaries'].reset_index(names='cell_id_final'), 
                      on='geometry', how='outer')
    
    ini2final = dict(zip(merged_gdf['cell_id_ini'], merged_gdf['cell_id_final']))

    return ini2final



def main(args):

    # Load sdata
    print("\n### Loading sdata... ###")
    sdata_ini = open_xenium(args.sdata_ini_path, args.img_key)
    print("\nsdata_ini loaded: ", sdata_ini)
    sdata_final = sd.read_zarr(args.sdata_final_path, selection=('tables','shapes'))
    print("\nsdata_final loaded: ", sdata_final)

    # Compare nucleus boundaries between sdatas
    print("\n### Comparing nucleus boundaries between sdatas... ###")
    compare_nucleus_boundaries(sdata_ini, sdata_final)

    # Get cell ids matching
    print("\n### Getting cell ids matching... ###")
    ini2final = cell_ids_matching(sdata_ini, sdata_final)
    print("E.g. ini2final: ", dict(itertools.islice(ini2final.items(), 15)))

    # Modify cell_id in sdata_ini
    print("\n### Modifying cell_id in sdata_ini... ###")
    
    print("\nE.g. initial cell_id for cell_boundaries: ", sdata_ini.shapes['cell_boundaries'][0:15])
    print("\nE.g. initial cell_id for table: ", sdata_ini.tables['table'].obs[0:15])
    
    sdata_ini.shapes['cell_boundaries'].index = sdata_ini.shapes['cell_boundaries'].index.map(ini2final)
    sdata_ini.tables['table'].obs['cell_id'] = sdata_ini.tables['table'].obs['cell_id'].map(ini2final)

    # Remove NA values in cell_id due to the fact that some cell boundaries can sometimes have no nucleus boundaries
    sdata_ini.shapes['cell_boundaries'] = sdata_ini.shapes['cell_boundaries'][~sdata_ini.shapes['cell_boundaries'].index.isna()]
    sdata_ini.tables['table'] = sdata_ini.tables['table'][~sdata_ini.tables['table'].obs['cell_id'].isna()]

    print("\nE.g. final cell_id for cell_boundaries: ", sdata_ini.shapes['cell_boundaries'][0:15])
    print("\nE.g. final cell_id for table: ", sdata_ini.tables['table'].obs[0:15])

    # Adding cells boundaries and cells table to sdata_final
    print("\n### Adding cells boundaries and cells table to sdata_final... ###")
    sdata_final.tables['table_nuclei'] = sdata_final.tables.pop('table')
    sdata_final.shapes['cell_boundaries'] = sdata_ini.shapes['cell_boundaries']
    sdata_final.tables['table_cells'] = sdata_ini.tables['table']
    print("\nsdata_final updated: ", sdata_final)

    # Save on disk
    print("\n### Saving on disk... ###")
    sdata_final.delete_element_from_disk('table')
    sdata_final.write_element('table_nuclei')
    sdata_final.write_element('cell_boundaries')
    sdata_final.write_element('table_cells')
    print("Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add cells (table and boundaries) in sdata")
    
    # Data path
    parser.add_argument("--sdata_ini_path", type=str, default=r"/Volumes/SAUV_FGS/MICS/data_HE2CellType/CT_DS/data_xenium10X/lung_s3/Xenium_Prime_Human_Lung_Cancer_FFPE_outs", help="Path to initial Xenium data")
    parser.add_argument("--sdata_final_path", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/sdata_processed/sdata_lung_s3.zarr", help="Path to sdata tangram annotated")
    parser.add_argument("--img_key", type=str, default="morphology_focus", help="Image key in sdata_ini")
    
    args = parser.parse_args()
    main(args)