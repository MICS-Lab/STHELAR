#!/bin/bash
#SBATCH --job-name=extract_cells_features
#SBATCH --output=%x.o%j 
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=45G
#SBATCH --mail-type=ALL

# Load necessary modules
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Activate anaconda environment
source activate transformers_fgs

# List of slide IDs
slide_ids=(breast_s1 breast_s0 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4 pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 brain_s0 bone_marrow_s0 bone_marrow_s1 bone_s0 prostate_s0 cervix_s0)

# Loop through each slide ID
for slide_id in "${slide_ids[@]}"; do
    echo "Processing slide: $slide_id"

    # Run the .py file
    time python /gpfs/users/user/HE2CellType/CT_DS/src/extract_cells_features.py \
        --imgs_zip_path "/gpfs/workdir/user/HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium/$slide_id/fold2/images.zip" \
        --masks_path "/gpfs/workdir/user/HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium/$slide_id/fold2/masks_cells.npz" \
        --patch_ids_path "/gpfs/workdir/user/HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium/$slide_id/fold2/patch_ids.npy" \
        --slide_id "$slide_id" \
        --batch_size 16 \
        --output_path /gpfs/workdir/user/HE2CellType/CT_DS/HE_features/features

    echo "Finished processing slide: $slide_id"
    echo "======================================================="
    echo " "
done