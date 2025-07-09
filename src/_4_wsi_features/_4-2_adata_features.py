'''Convert features for each model into AnnData format and save them into .h5ad files'''

import argparse
import os
import numpy as np
import anndata as ad
from tqdm import tqdm


def process_features_to_adata(features, feature_prefix):
    """
    Convert features dictionary directly to AnnData format without using Pandas DataFrame.
    """
    # Prepare lists for obs and feature matrix (X)
    cell_ids = []
    patch_ids_list = []
    feature_matrix = []

    for cell_id, (features_array, patch_ids) in tqdm(features.items()):
        # Process patch_ids: Convert to string and remove `.png`
        processed_patch_ids = [str(patch_id).replace('.png', '') for patch_id in patch_ids]
        
        # Collect data
        cell_ids.append(cell_id)
        patch_ids_list.append(','.join(processed_patch_ids))  # Join list into a string for compatibility
        feature_matrix.append(features_array)

    # Convert feature matrix to NumPy array
    feature_matrix = np.array(feature_matrix, dtype=np.float32)

    # Create obs (observations) with cell_id and patch_ids
    obs = {
        'cell_id': cell_ids,
        'patch_ids': patch_ids_list
    }

    # Create var (variables) with feature names
    num_features = feature_matrix.shape[1]
    var = {
        'index': [f"{feature_prefix}{i}" for i in range(num_features)]
    }

    # Create AnnData object
    adata = ad.AnnData(X=feature_matrix, obs=obs, var=var)

    return adata



def main(args):
    print(f'\n==== Processing {args.slide_id} ====')

    for model in ['phikonv2', 'vit_google', 'cellvit']:
        print(f"\n###### Adding features for model {model} ######")

        # Load the features
        print('* Loading features...')
        if model in ['phikonv2', 'vit_google']:
            features_model_path = os.path.join(
                args.features_path, args.slide_id, f"cell_features_{args.slide_id}_{model}.npy"
            )
        elif model == 'cellvit':
            features_model_path = os.path.join(
                args.cellvit_features_path, args.slide_id, f"cell_features_{model}.npy"
            )
        features = np.load(features_model_path, allow_pickle=True).item()

        # Convert features to AnnData
        print('* Converting to adata...')
        adata = process_features_to_adata(features, feature_prefix=f"HE{model}_")

        # Checking
        print(f'\n* Checking adata for features_{model}:\n', adata)

        # Save adata into a .h5ad file
        print('\n* Saving adata into a .h5ad file...')
        os.makedirs(os.path.join(args.output_path, args.slide_id), exist_ok=True)
        adata.write_h5ad(os.path.join(args.output_path, args.slide_id, f"adata_features_{args.slide_id}_{model}.h5ad"))

        print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save features for each model into adata")
    
    parser.add_argument("--slide_id", type=str, default='heart_s0', help="Slide id")
    parser.add_argument("--features_path", type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/HE_features/features/features_extraction', help="Path to folder containing .npy files with features for phikonv2 and vit_google")
    parser.add_argument("--cellvit_features_path", type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/output_cellvit', help="Path to folder containing .npy files with features for cellvit")
    parser.add_argument("--output_path", type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/HE_features/features/adata_HEfeatures', help="Path to save the adata files for features for each model")

    args = parser.parse_args()
    main(args)