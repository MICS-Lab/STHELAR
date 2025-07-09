"""Make a CSV file comparing annotations for nuclei, cells, and combined labels, and also PanNuke labels and probabilities."""

import os
import argparse
import spatialdata as sd
import pandas as pd
import gc


def process_and_save_individual(sdata_dir, temp_dir):
    """
    Process individual .zarr files and save their results as intermediate CSVs.

    Args:
        sdata_dir (str): Directory containing .zarr files.
        temp_dir (str): Directory to store intermediate CSV files.

    Returns:
        None
    """
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Get all .zarr file paths
    sdata_paths = [os.path.join(sdata_dir, f) for f in os.listdir(sdata_dir) if f.endswith('.zarr')]
    print(f"Number of .zarr files to process: {len(sdata_paths)}\n")

    for i, sdata_path in enumerate(sdata_paths):
        try:
            slide_id = os.path.basename(sdata_path).replace('sdata_', '').replace('.zarr', '')
            print(f"\n* Processing {slide_id} ({i+1}/{len(sdata_paths)})")

            # Read the spatial data file
            sdata = sd.read_zarr(sdata_path, selection=('tables',))

            # Extract observations from each table
            adata_obs_nuclei = sdata.tables['table_nuclei'].obs
            adata_obs_cells = sdata.tables['table_cells'].obs
            adata_obs_combined = sdata.tables['table_combined'].obs

            # Add slide_id for traceability
            adata_obs_nuclei['slide_id'] = slide_id
            adata_obs_cells['slide_id'] = slide_id
            adata_obs_combined['slide_id'] = slide_id

            # Merge the data from this slide
            partial_merge = pd.merge(
                adata_obs_nuclei[['slide_id', 'cell_id', 'label1', 'final_label']],
                adata_obs_cells[['slide_id', 'cell_id', 'label1', 'final_label', 'PanNuke_label', 'PanNuke_proba']],
                on=['slide_id', 'cell_id'],
                suffixes=('_nuclei', '_cells'),
                how='outer'
            )
            partial_merge = pd.merge(
                partial_merge,
                adata_obs_combined[['slide_id', 'cell_id', 'final_label_combined']],
                on=['slide_id', 'cell_id'],
                suffixes=('', '_combined'),
                how='outer'
            )

            # Save the intermediate result to a CSV file
            partial_csv_path = os.path.join(temp_dir, f"{slide_id}_partial.csv")
            partial_merge.to_csv(partial_csv_path, index=False)
            print(f"-> Saved.")

            # Clear memory
            del sdata, adata_obs_nuclei, adata_obs_cells, adata_obs_combined, partial_merge
            gc.collect()

        except Exception as e:
            print(f"Error processing {sdata_path}: {e}")


def merge_partial_results(temp_dir, output_csv):
    """
    Merge all intermediate CSVs into a final output file.

    Args:
        temp_dir (str): Directory containing intermediate CSV files.
        output_csv (str): Path to save the final merged CSV file.

    Returns:
        None
    """
    partial_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('_partial.csv')]
    print(f"Number of partial files to merge: {len(partial_files)}\n")

    # Open the final CSV file for writing in chunks
    with open(output_csv, 'w') as output_file:
        for i, partial_file in enumerate(partial_files):
            print(f"* Merging {os.path.basename(partial_file)} ({i+1}/{len(partial_files)})")
            with open(partial_file, 'r') as f:
                if i == 0:
                    # Write the header for the first file
                    output_file.write(f.read())
                else:
                    # Skip the header for subsequent files
                    next(f)  # Skip header line
                    output_file.write(f.read())

    print(f"Done.")


def main(args):
    """
    Main function to execute the processing pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None
    """
    temp_dir = os.path.join(os.path.dirname(args.output_csv), "temp_partial_results")

    print("\nStep 1: Processing individual .zarr files...")
    process_and_save_individual(args.sdata_dir, temp_dir)

    print("\nStep 2: Merging intermediate results...")
    merge_partial_results(temp_dir, args.output_csv)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a CSV file comparing annotations.")

    parser.add_argument(
        "--sdata_dir",
        type=str,
        help="Directory containing sdata (.zarr files)",
        default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/sdata_final"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to save the final merged CSV file",
        default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/annots/adata_compare_annots_v2.csv"
    )

    args = parser.parse_args()
    main(args)