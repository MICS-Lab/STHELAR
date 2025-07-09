"""
Extract H&E features for individual cells
NB: For H&E features using phikonv2 and vit_google ⇒ training at x20 whereas our dataset is x40 
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import zipfile
from PIL import Image
import io
import os
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
from scipy.sparse import load_npz



# Function to load an image file from a zip
def load_imgfile(imgs_zip_path, img_name):
    with zipfile.ZipFile(imgs_zip_path, 'r') as img_zip:
        img_data = img_zip.read(img_name)
        img = Image.open(io.BytesIO(img_data))
    return np.array(img).astype(np.uint8)


# Function to load a mask file from a zip
def load_maskfile(masks_sparse_matrix, idx):
    mask = masks_sparse_matrix[idx].toarray().reshape(256, 256)
    return mask.astype(np.int32)



# Reverse int cell id to str cell id
def str_cell_id(cell_id: int) -> str:
    """Transforms an integer cell ID into an Xenium Explorer alphabetical cell id"""
    cell_id -= 1  # Shift by 1 to avoid having 0 as a cell ID because of background
    coefs = []
    for _ in range(8):
        cell_id, coef = divmod(cell_id, 16)
        coefs.append(coef)
    return "".join([chr(97 + coef) for coef in coefs][::-1]) + "-1"



# Global counter for the number of debug images saved
debug_image_count = 0
def save_debug_visualization(
    patch_id, cell_id, inst_map, inst_map_224, mask_224to14, checking_dir
):
    """
    Saves debug visualizations for a specific cell within a patch,
    including instance maps with the given cell highlighted.
    """
    global debug_image_count
    if debug_image_count >= 50:
        return  # Do not save more than 50 images

    # Create binary masks for the specific cell ID
    cell_mask_256 = (inst_map == cell_id).astype(float)  # Highlight the cell in 256x256
    cell_mask_224 = (inst_map_224 == cell_id).astype(float)  # Highlight the cell in 224x224

    # Plot the masks and instance maps
    plt.figure(figsize=(12, 10))

    # Original Instance Map (256x256)
    plt.subplot(3, 2, 1)
    plt.title("Original Inst Map (256x256)")
    plt.imshow(inst_map, cmap="tab20b")
    plt.axis("on")

    # Resized Instance Map (224x224)
    plt.subplot(3, 2, 2)
    plt.title("Resized Inst Map (224x224)")
    plt.imshow(inst_map_224, cmap="tab20b")
    plt.axis("on")

    # Highlighted Cell in Inst Map (256x256)
    plt.subplot(3, 2, 3)
    plt.title(f"Inst Map (256x256) - Cell {cell_id}")
    plt.imshow(cell_mask_256, cmap="Reds")
    plt.axis("on")

    # Highlighted Cell in Resized Inst Map (224x224)
    plt.subplot(3, 2, 4)
    plt.title(f"Resized Inst Map (224x224) - Cell {cell_id}")
    plt.imshow(cell_mask_224, cmap="Reds")
    plt.axis("on")

    # Mask (16x16)
    plt.subplot(3, 2, 5)
    plt.title("Mask (16x16)")
    # Nothing to show here, as the process in done during cellvit inference, put empty plot
    plt.imshow(np.zeros((16, 16)), cmap="Blues")
    plt.axis("on")

    # Mask (14x14)
    plt.subplot(3, 2, 6)
    plt.title("Mask (14x14)")
    plt.imshow(mask_224to14, cmap="Blues")
    plt.axis("on")

    # Save the figure
    plt.tight_layout()
    file_name = f"cell_masks_{patch_id}_{str_cell_id(cell_id)}.png"
    plt.savefig(os.path.join(checking_dir, file_name))
    plt.close()

    debug_image_count += 1  # Increment the debug image counter



# Custom Dataset for Cells Patches and Masks
class CellsDataset(Dataset):

    def __init__(self, imgs_zip_path, masks_path, patch_ids, phikon_processor, vit_google_processor):
        self.imgs_zip_path = imgs_zip_path
        self.masks_path = masks_path
        print(f"\nLoading masks from {masks_path}...")
        self.masks_sparse_matrix = load_npz(masks_path)
        self.patch_ids = patch_ids
        self.phikon_processor = phikon_processor
        self.vit_google_processor = vit_google_processor

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):

        patch_id = self.patch_ids[idx]

        # Load the image and mask
        img_name = f"{patch_id}.png"
        image = load_imgfile(self.imgs_zip_path, img_name)
        inst_map = load_maskfile(self.masks_sparse_matrix, idx)

        # Preprocess the image for both models here
        image_pil = Image.fromarray(image)
        inputs_phikon = self.phikon_processor(images=image_pil, return_tensors="pt")
        inputs_vit_google = self.vit_google_processor(images=image_pil, return_tensors="pt")

        return inst_map, patch_id, inputs_phikon, inputs_vit_google
    


def custom_collate_fn(batch):
    return batch



def process_patch_batch(batch, phikon_model, vit_google_model, device, checking_dir, cell_features_phikon, cell_features_vit_google):
    """
    Processes a batch of image patches to extract features for Phikon-v2 and ViT-Google.
    """

    # Unpack the batch and move images to a batched tensor
    inst_maps, patch_ids, inputs_phikon, inputs_vit_google = zip(*batch)

    # Convert inst_maps to a batched tensor (N, 256, 256)
    inst_maps = torch.stack([torch.tensor(inst_map, device=device) for inst_map in inst_maps])  # Shape (batch_size, 256, 256)

    # Resize inst_maps to 224x224 for Phikon and ViT-Google in parallel
    inst_maps_224 = F.interpolate(inst_maps.unsqueeze(1).float(), size=(224, 224), mode="nearest").squeeze(1)

    # Get unique cell IDs per patch (batch processing)
    unique_cell_ids = [torch.unique(inst_map[inst_map != 0]) for inst_map in inst_maps] # Exclude background

    # Create binary masks for each cell ID and batch them (224x224 to 14x14)
    masks_224to14 = torch.zeros(len(batch), max([len(cells) for cells in unique_cell_ids]), 14, 14, device=device)
    for b_idx, cell_ids in enumerate(unique_cell_ids):
        # Create masks for all cell IDs in the batch
        cell_masks = (inst_maps[b_idx].unsqueeze(0) == cell_ids.unsqueeze(-1).unsqueeze(-1)).float()
        # Apply pooling to all masks simultaneously
        pooled_masks = F.adaptive_max_pool2d(cell_masks, (14, 14))
        masks_224to14[b_idx, :len(cell_ids)] = pooled_masks

    # Extract spatial features for Phikon and ViT-Google in batch
    with torch.no_grad():
        spatial_features_phikon = phikon_model(**{k: torch.cat([x[k] for x in inputs_phikon]).to(device) for k in inputs_phikon[0]}).last_hidden_state[:, 1:, :]
        spatial_features_phikon = spatial_features_phikon.view(len(batch), 14, 14, -1).permute(0, 3, 1, 2)  # (batch_size, 1024, 14, 14)

        spatial_features_vit_google = vit_google_model(**{k: torch.cat([x[k] for x in inputs_vit_google]).to(device) for k in inputs_vit_google[0]}).last_hidden_state[:, 1:, :]
        spatial_features_vit_google = spatial_features_vit_google.view(len(batch), 14, 14, -1).permute(0, 3, 1, 2)  # (batch_size, 768, 14, 14)

    # Matrix multiplication for features extraction
    phikon_features = torch.einsum('bchw,bnhw->bnc', spatial_features_phikon, masks_224to14)  # (batch_size, num_cells, 1024)
    vit_google_features = torch.einsum('bchw,bnhw->bnc', spatial_features_vit_google, masks_224to14)  # (batch_size, num_cells, 768)

    # Collect features for each cell in the batch
    for b_idx, cell_ids in enumerate(unique_cell_ids):
        patch_id = patch_ids[b_idx]
        
        for c_idx, cell_id in enumerate(cell_ids):
            cell_id_str = str_cell_id(cell_id.item())
            
            # For Phikon features
            if cell_id_str not in cell_features_phikon:
                cell_features_phikon[cell_id_str] = [torch.zeros(phikon_features.size(-1), device='cpu'), []]
            cell_features_phikon[cell_id_str][0] += phikon_features[b_idx, c_idx].cpu()
            cell_features_phikon[cell_id_str][1].append(patch_id)

            # For ViT-Google features
            if cell_id_str not in cell_features_vit_google:
                cell_features_vit_google[cell_id_str] = [torch.zeros(vit_google_features.size(-1), device='cpu'), []]
            cell_features_vit_google[cell_id_str][0] += vit_google_features[b_idx, c_idx].cpu()
            cell_features_vit_google[cell_id_str][1].append(patch_id)       
    
    
    # Randomly pick one cell for debugging visualization
    if len(unique_cell_ids) > 0:
        b_idx = np.random.choice(len(unique_cell_ids))  # Select a random patch
        if len(unique_cell_ids[b_idx]) > 0:
            random_idx = np.random.choice(len(unique_cell_ids[b_idx]))  # Select a random cell
            random_cell_id = unique_cell_ids[b_idx][random_idx].item()

            # Visualize masks for the selected cell
            save_debug_visualization(
                patch_ids[b_idx],
                random_cell_id,
                inst_maps[b_idx].cpu().numpy(),
                inst_maps_224[b_idx].cpu().numpy(),
                masks_224to14[b_idx, random_idx].cpu().numpy(),
                checking_dir,
            )




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
    




def main(args):

    print(f"\n==== Processing {args.slide_id} ====")

    # Load the patch_ids
    patch_ids = np.load(args.patch_ids_path)

    # Create the output directory
    save_dir = os.path.join(args.output_path, args.slide_id)
    os.makedirs(save_dir, exist_ok=True)
    checking_dir = os.path.join(save_dir, "checking")
    os.makedirs(os.path.join(checking_dir), exist_ok=True)

    # Load the Phikon-v2 and vit_google models and their image processors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phikon_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    phikon_model = AutoModel.from_pretrained("owkin/phikon-v2").to(device).eval()
    # => hidden_size=1024; image_size=224; patch_size=16; thus subpatch_size=224/16=14

    assert phikon_model.config.hidden_size == 1024
    assert phikon_model.config.image_size == 224
    assert phikon_model.config.patch_size == 16

    vit_google_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_google_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()
    # => hidden_size=768; image_size=224; patch_size=16; thus subpatch_size=224/16=14

    assert vit_google_model.config.hidden_size == 768
    assert vit_google_model.config.image_size == 224
    assert vit_google_model.config.patch_size == 16

    # Instantiate the dataset and DataLoader
    dataset = CellsDataset(args.imgs_zip_path, args.masks_path, patch_ids, phikon_processor, vit_google_processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=3, collate_fn=custom_collate_fn)
    print(f"There are {len(dataset)} patches.")

    # Initialize dictionaries for features
    cell_features_phikon = {}
    cell_features_vit_google = {}

    for batch in tqdm(dataloader):

        process_patch_batch(
            batch, phikon_model, vit_google_model, device, checking_dir, cell_features_phikon, cell_features_vit_google
        )
        torch.cuda.empty_cache()
    
    # Compute mean features for each cell across all patches
    print("Computing mean features for each cell across all patches...")
    cell_features_phikon = compute_mean_features(cell_features_phikon)
    cell_features_vit_google = compute_mean_features(cell_features_vit_google)

    # Save features
    print(f"Saving extracted features for Phikon-v2 and Vit_google in {save_dir}...")
    np.save(os.path.join(save_dir, f"cell_features_{args.slide_id}_phikonv2.npy"), cell_features_phikon)
    np.save(os.path.join(save_dir, f"cell_features_{args.slide_id}_vit_google.npy"), cell_features_vit_google)
    print("Done")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Extract cell features from patches using Phikon-v2 and ViT-Google")
    
    parser.add_argument('--imgs_zip_path', type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium/heart_s0/images.zip', help="Path to the images ZIP file")
    parser.add_argument('--masks_path', type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/patches_xenium/heart_s0/masks_cells.npz', help="Path to the npz file with cells masks")
    parser.add_argument('--patch_ids_path', type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/patches_xenium/heart_s0/patch_ids.npy', help="Path to the patch_ids .npy file")
    parser.add_argument('--slide_id', type=str, default='heart_s0', help="Slide ID")

    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for processing patches")
    parser.add_argument('--output_path', type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/HE_features/features/features_extraction', help="Path to folder to save the extracted cell features dictionary")
    
    args = parser.parse_args()
    main(args)