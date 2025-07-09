"""Merge inference results from two parts of a dataset"""

## NB ##
# It means that during inference we put :
    # - in HE2CT/cell_segmentation/datasets/pannuke.py :
        # ### CHOOSE ONLY FOR INFERENCE IN SEVERAL TIMES ###
        # # DO NOT FORGET TO REMOVE MEAN FOR CELL TOKENS IN THE INFERENCE SCRIPT (cf. to comment)
        # split_idx = len(self.images) // 2
        # self.images = self.images[split_idx:]
        # self.masks = self.masks[split_idx:]
        # self.img_names = self.img_names[split_idx:]
        # ### END CHOOSE ONLY FOR SEVERAL TIMES INFERENCE ###
    # - and comment the fowolling line in HE2CT/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py :
        # # self.all_cell_tokens = self.compute_mean_features(self.all_cell_tokens)  # TO COMMENT IF INFERENCE IN SEVERAL TIMES FOR SAME SLIDE (FOR INSTANCE BREAST_S1)

# This is only the case if our hardware does not allow to do inference on the whole slide at once. (Not the case on JeanZay for instance, where we can do inference on the whole slide at once)

import numpy as np
import torch
import argparse
import os
import h5py
import json
import gc
from tqdm import tqdm


def compute_mean_features(cell_features):

    # Prepare to store all sum tensors and patch id lists
    cell_ids = list(cell_features.keys())  # List of cell ids

    # Collect all summed feature tensors and patch id lists for each cell
    sums = []
    patch_counts = []
    
    for cell_id_str in cell_ids:
        
        sum_features, patch_ids = cell_features[cell_id_str]
        sums.append(sum_features)
        patch_counts.append(len(patch_ids))  # Count of patches

    # Convert lists to tensors (this allows vectorized computation)
    sums = torch.stack(sums)
    patch_counts = torch.tensor(patch_counts, dtype=torch.float32)

    # Calculate means by dividing sums by the number of patches
    means = sums / patch_counts.unsqueeze(1)  # Unsqueeze for broadcasting

    # Convert means to numpy arrays
    means = means.numpy()

    # Rebuild dictionaries with the computed means
    cell_features_mean = {
        cell_id_str: [means[i], cell_features[cell_id_str][1]]
        for i, cell_id_str in enumerate(cell_ids)
    }

    return cell_features_mean



def merge_cell_features(n_split, slide_id, cellvit_path):

    # Base split
    final = np.load(os.path.join(cellvit_path, slide_id, "inference_part1", "cell_features_cellvit.npy"), allow_pickle=True).item()

    # Merge base split with the other splits
    for split in range(2, n_split+1):

        split = np.load(os.path.join(cellvit_path, slide_id, f"inference_part{split}", "cell_features_cellvit.npy"), allow_pickle=True).item()

        for cell_id, (features, patch_ids) in tqdm(split.items()):
            
            if cell_id in final:
                final[cell_id][0] += features.clone().detach()
                final[cell_id][1].extend(patch_ids)
            else:
                final[cell_id] = [features.clone().detach(), patch_ids]

    # Perform the mean of the features
    final = compute_mean_features(final)

    # Save the merged cell_features
    np.save(os.path.join(cellvit_path, slide_id, "cell_features_cellvit.npy"), final)

    # Clear memory
    del final, split
    gc.collect()



def merge_instance_map(n_split, slide_id, cellvit_path):

    final_output_path = os.path.join(cellvit_path, slide_id, "inference_instance_map_predictions.h5")

    with h5py.File(final_output_path, "w") as final_h5:

        for split_idx in range(1, n_split + 1):
            split_path = os.path.join(cellvit_path, slide_id, f"inference_part{split_idx}", "inference_instance_map_predictions.h5")

            with h5py.File(split_path, "r") as split_h5:
                for data_name in tqdm(split_h5.keys()):
                    if data_name in final_h5:
                        raise ValueError(f"Duplicate dataset found: {data_name} in split {split_idx}. Check your inputs!")
                    split_h5.copy(data_name, final_h5)
    
    # Clear memory
    del final_h5, split_h5
    gc.collect()



def merge_inference_results(n_split, slide_id, cellvit_path):
    
    # Load the base JSON file
    base_path = os.path.join(cellvit_path, slide_id, "inference_part1", "inference_results.json")
    with open(base_path, "r") as f:
        final_results = json.load(f)

    # Function to initialize values as lists
    def ensure_list(value):
        if isinstance(value, list):
            return value
        elif isinstance(value, (int, float)): 
            return [value]
        return [np.nan]

    # Ensure dataset metrics hold lists, including NaNs
    for key, value in final_results["dataset"].items():
        final_results["dataset"][key] = ensure_list(value)

    # Ensure tissue_metrics hold lists for numeric fields
    for tissue, metrics in final_results["tissue_metrics"].items():
        for metric_key, value in metrics.items():
            final_results["tissue_metrics"][tissue][metric_key] = ensure_list(value)

    # Ensure nuclei_metrics_pq and nuclei_metrics_d hold lists for numeric fields
    for key, value in final_results["nuclei_metrics_pq"].items():
        final_results["nuclei_metrics_pq"][key] = ensure_list(value)

    for nuclei, metrics in final_results["nuclei_metrics_d"].items():
        for metric_key, value in metrics.items():
            final_results["nuclei_metrics_d"][nuclei][metric_key] = ensure_list(value)

    # Iterate through other splits and merge
    for split in range(2, n_split + 1):
        split_path = os.path.join(cellvit_path, slide_id, f"inference_part{split}", "inference_results.json")
        with open(split_path, "r") as f:
            split_results = json.load(f)

        # Merge "dataset" (append all values, including NaNs)
        for key, value in split_results["dataset"].items():
            if key not in final_results["dataset"]:
                final_results["dataset"][key] = ensure_list(value)
            else:
                final_results["dataset"][key].append(value if isinstance(value, (int, float)) and not np.isnan(value) else np.nan)

        # Merge "image_metrics" (check for duplicates)
        for image_name, metrics in split_results["image_metrics"].items():
            if image_name not in final_results["image_metrics"]:
                final_results["image_metrics"][image_name] = metrics
            else:
                raise ValueError(f"Duplicate image found: {image_name}. Check your inputs!")

        # Merge "tissue_metrics" (append values, including NaNs)
        for tissue, metrics in split_results["tissue_metrics"].items():
            if tissue not in final_results["tissue_metrics"]:
                final_results["tissue_metrics"][tissue] = {
                    k: ensure_list(v) for k, v in metrics.items()
                }
            else:
                for metric_key, value in metrics.items():
                    if metric_key not in final_results["tissue_metrics"][tissue]:
                        final_results["tissue_metrics"][tissue][metric_key] = ensure_list(value)
                    else:
                        final_results["tissue_metrics"][tissue][metric_key].append(
                            value if isinstance(value, (int, float)) and not np.isnan(value) else np.nan
                        )

        # Merge "nuclei_metrics_pq" (append values, including NaNs)
        for key, value in split_results["nuclei_metrics_pq"].items():
            if key not in final_results["nuclei_metrics_pq"]:
                final_results["nuclei_metrics_pq"][key] = ensure_list(value)
            else:
                final_results["nuclei_metrics_pq"][key].append(value if isinstance(value, (int, float)) and not np.isnan(value) else np.nan)

        # Merge "nuclei_metrics_d" (append values, including NaNs)
        for nuclei, metrics in split_results["nuclei_metrics_d"].items():
            if nuclei not in final_results["nuclei_metrics_d"]:
                final_results["nuclei_metrics_d"][nuclei] = {
                    k: ensure_list(v) for k, v in metrics.items()
                }
            else:
                for metric_key, value in metrics.items():
                    if metric_key not in final_results["nuclei_metrics_d"][nuclei]:
                        final_results["nuclei_metrics_d"][nuclei][metric_key] = ensure_list(value)
                    else:
                        final_results["nuclei_metrics_d"][nuclei][metric_key].append(
                            value if isinstance(value, (int, float)) and not np.isnan(value) else np.nan
                        )

    # Save the merged results
    output_path = os.path.join(cellvit_path, slide_id, "inference_results.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # Clear memory
    del final_results, split_results
    gc.collect()



def merge_pannuke_labels_gt(n_split, slide_id, output_cellvit_folder):

    # Load the first split file as the base
    base_file_path = os.path.join(output_cellvit_folder, slide_id, "inference_part1", "pannuke_labels_gt.pth")
    merged_pannuke_labels = torch.load(base_file_path)
    
    # Iterate over the remaining splits and update the base file
    for split in range(2, n_split + 1):
        
        split_file_path = os.path.join(output_cellvit_folder, slide_id, f"inference_part{split}", "pannuke_labels_gt.pth")
        split_pannuke_labels = torch.load(split_file_path)
        
        # Update the counts for each cell_id
        for cell_id, counts in tqdm(split_pannuke_labels.items()):
            if cell_id in merged_pannuke_labels:
                merged_pannuke_labels[cell_id] += counts  # Sum the counts if cell_id exists
            else:
                merged_pannuke_labels[cell_id] = counts  # Add new cell_id

    # Save the merged result
    output_path = os.path.join(output_cellvit_folder, slide_id, "pannuke_labels_gt.pth")
    torch.save(merged_pannuke_labels, output_path)

    # Clear memory
    del merged_pannuke_labels, split_pannuke_labels
    gc.collect()




def main(args):

    if args.n_split < 2:
        raise ValueError("n_split must be greater than 1")

    # Merge cell_features_cellvit.npy
    print("\n* Merging cell_features_cellvit.npy...")
    merge_cell_features(args.n_split, args.slide_id, args.output_cellvit_folder)
    
    # Merge inference_instance_map_predictions.h5
    print("\n* Merging inference_instance_map_predictions.h5...")
    merge_instance_map(args.n_split, args.slide_id, args.output_cellvit_folder)

    # Merge inference_results.json
    print("\n* Merging inference_results.json...")
    merge_inference_results(args.n_split, args.slide_id, args.output_cellvit_folder)

    # Merge pannuke_labels_gt.pth
    print("\n* Merging pannuke_labels_gt.pth...")
    merge_pannuke_labels_gt(args.n_split, args.slide_id, args.output_cellvit_folder)

    print("\nDone.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add metrics to check alignment patches in sdata")
    
    parser.add_argument("--slide_id", type=str, default="breast_s1", help="Slide id")
    parser.add_argument("--n_split", type=int, default=2, help="Number of splits")
    parser.add_argument("--output_cellvit_folder", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/output_cellvit", help="Folder to save output cellvit results from inference")

    args = parser.parse_args()
    main(args)
