#!/bin/bash
#SBATCH --job-name=pca_big_slides
#SBATCH --output=%x.o%j 
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --mem=60G
#SBATCH --mail-type=ALL

# Load necessary modules
module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Activate anaconda environment
source activate rapids_cluster

# List of slide IDs
slide_ids=(nuclei_breast_s6 nuclei_lymph_node_s0 nuclei_ovary_s1 nuclei_cervix_s0 cyto_breast_s6 cyto_lymph_node_s0 cyto_ovary_s1 cyto_cervix_s0)

# Loop through each slide ID
for slide_id in "${slide_ids[@]}"; do
    echo "Processing slide: $slide_id"

    # Run the .py file
    time python /gpfs/users/user/HE2CellType/CT_DS/src/pca_for_big_slides.py \
        --slide_id "$slide_id"

    echo "Finished processing slide: $slide_id"
    echo "======================================================="
    echo " "
done