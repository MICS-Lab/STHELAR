"""
Extract trained labels for future comparison with our own labels
"""

import os
import argparse
import yaml
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm



# Function to open the YAML file and extract the nuclei types
def get_nuclei_types(yaml_path):
    """
    Load nuclei types and their indices from a YAML file.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        Dict: Mapping from indices to nuclei type names.
    """
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    nuclei_types = yaml_data['nuclei_types']  # Mapping for nuclei types
    index_to_nuclei_type = {v: k for k, v in nuclei_types.items()}
    return index_to_nuclei_type



def process_trained_labels(count_pixels_path, index_to_nuclei_type):
    """
    Process the pixel counts and determine the most likely class for each cell.

    Args:
        count_pixels_path (str): Path to the .pth file containing pixel counts.
        index_to_nuclei_type (dict): Mapping of indices to nuclei type names.

    Returns:
        pd.DataFrame: Dataframe containing cell_id, trained_cell_type, and trained_proba.
    """
    # Load the pixel counts
    pixel_counts = torch.load(count_pixels_path)

    cell_ids = []
    trained_cell_types = []
    trained_probas = []

    print("Processing trained labels...")
    # Iterate over each cell_id and its associated pixel counts
    for cell_id, counts in tqdm(pixel_counts.items()):
        
        total_pixels = sum(counts)
        
        # Ignore cells with zero total pixels
        if total_pixels == 0:
            print(f"Cell {cell_id} has zero total pixels. Skipping...")
            continue

        # Find the indices of the two highest pixel counts
        sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
        primary_index = sorted_indices[0]
        secondary_index = sorted_indices[1] if len(sorted_indices) > 1 else None

        # Determine the primary class (handle "Background" case)
        if index_to_nuclei_type[primary_index] == "Background" and secondary_index is not None and counts[secondary_index] > 0:
            primary_index = secondary_index

        # Calculate the probability of the chosen class
        trained_cell_type = index_to_nuclei_type[primary_index]
        trained_proba = counts[primary_index] / total_pixels

        # Append results
        cell_ids.append(cell_id)
        trained_cell_types.append(trained_cell_type)
        trained_probas.append(trained_proba.item())

    return cell_ids, trained_cell_types, trained_probas



def main(args):
    """
    Main function to extract trained labels and save them to a parquet file.
    """

    print(f"\n==== Processing {args.slide_id} ====")
    # Paths
    print("Loading data...")
    dataset_config_path = os.path.join(args.dataset_config_path)
    count_pixels_path = os.path.join(args.output_cellvit_folder, args.slide_id, 'pannuke_labels_gt.pth')

    # Load the nuclei types from the YAML file
    index_to_nuclei_type = get_nuclei_types(dataset_config_path)

    # Process trained labels
    cell_ids, trained_cell_types, trained_probas = process_trained_labels(count_pixels_path, index_to_nuclei_type)

    print("Checking for one example:")
    print(f"Cell ID: {cell_ids[0]}")
    print(f"trained cell type: {trained_cell_types[0]}")
    print(f"trained probability: {trained_probas[0]}")

    # Save directly to Parquet format
    os.makedirs(args.output_path, exist_ok=True)
    output_parquet_path = os.path.join(args.output_path, f'trained_labels_{args.training_id}_{args.slide_id}.parquet')
    print(f"Saving results to {output_parquet_path}...")
    table = pa.Table.from_pydict({
        "cell_id": cell_ids,
        f"{args.training_id}_cell_type": trained_cell_types,
        f"{args.training_id}_proba": trained_probas,
    })
    pq.write_table(table, output_parquet_path, compression="snappy")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get trained labels for gt segmentation")

    parser.add_argument('--output_cellvit_folder', type=str, default='/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/analyze_trained_model/training_28/output_model', help="Path to folder with output from Cellvit")
    parser.add_argument('--data_path', type=str, default='/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat', help="Path to folder with prepared patches for Xenium")
    parser.add_argument('--dataset_config_path', type=str, default='/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/training_datasets/ds_4/dataset_config.yaml', help="Path to dataset_config.yaml")
    parser.add_argument('--slide_id', type=str, default='heart_s0', help="Slide ID")
    parser.add_argument('--training_id', type=str, default='training_28', help="Training ID")
    parser.add_argument('--output_path', type=str, default='/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/analyze_trained_model/training_28/trained_predictions', help="Path to folder to save the trained predictions")
    
    args = parser.parse_args()
    main(args)