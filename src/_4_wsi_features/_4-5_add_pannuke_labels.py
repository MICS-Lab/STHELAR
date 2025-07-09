"""
Add PanNuke labels to sdata
"""

import argparse
import os
import pyarrow.parquet as pq
import spatialdata as sd



def main(args):

    print(f'\n==== Processing {args.slide_id} ====')

    # Load the sdata object
    print('\nLoading sdata...')
    sdata_path = os.path.join(args.sdata_folder, f'sdata_{args.slide_id}.zarr')
    sdata = sd.read_zarr(sdata_path, selection=('tables',))

    # Load the PanNuke labels
    print('Loading PanNuke labels...')
    table = pq.read_table(os.path.join(args.pannuke_labels_folder, f'pannuke_labels_{args.slide_id}.parquet'))
    pannuke_data = table.to_pydict()

    print('Adding labels to sdata...')
    
    cellid2pannuke = dict(zip(pannuke_data['cell_id'], pannuke_data['pannuke_cell_type']))
    # sdata.tables['table_nuclei'].obs['PanNuke_label'] = sdata.tables['table_nuclei'].obs['cell_id'].map(cellid2pannuke).fillna('Unknown')
    sdata.tables['table_cells'].obs['PanNuke_label'] = sdata.tables['table_cells'].obs['cell_id'].map(cellid2pannuke).fillna('Unknown')

    cellid2proba = dict(zip(pannuke_data['cell_id'], pannuke_data['pannuke_proba']))
    # sdata.tables['table_nuclei'].obs['PanNuke_proba'] = sdata.tables['table_nuclei'].obs['cell_id'].map(cellid2proba).fillna(0)
    sdata.tables['table_cells'].obs['PanNuke_proba'] = sdata.tables['table_cells'].obs['cell_id'].map(cellid2proba).fillna(0)


    # Checking
    print('\nChecking:')
    print('\nNuclei:')
    print(sdata.tables['table_nuclei'].obs['PanNuke_label'].value_counts())
    print('\nCells:')
    print(sdata.tables['table_cells'].obs['PanNuke_label'].value_counts())

    # Save on disk
    print('\nSaving on disk...')
    # sdata.delete_element_from_disk("table_nuclei")
    # sdata.write_element('table_nuclei')
    sdata.delete_element_from_disk("table_cells")
    sdata.write_element('table_cells')
    print('Done.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add metrics to check alignment patches in sdata")
    
    parser.add_argument("--slide_id", type=str, default='heart_s0', help="Slide id")
    parser.add_argument("--pannuke_labels_folder", type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/HE_features/PanNuke_predictions', help="Path to folder with PanNuke labels")
    parser.add_argument("--sdata_folder", type=str, default='/Volumes/SAUV_FGS/MICS/data_HE2CellType/CT_DS/sdata_final', help="Folder containing final sdata")

    args = parser.parse_args()
    main(args)
