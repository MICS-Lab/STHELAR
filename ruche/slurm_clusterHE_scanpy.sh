#!/bin/bash
#SBATCH --job-name=clustering_gpu_HEfeatures_1
#SBATCH --output=%x.o%j 
#SBATCH --time=03:00:00 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --mem=80G
#SBATCH --mail-type=ALL

# Load necessary modules
module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Activate anaconda environment
source activate rapids_cluster

# Run python script

# time python /gpfs/users/user/HE2CellType/CT_DS/src/clustering_with_gpu_for_big_slides.py

# List of slide IDs
slide_ids=(breast_s0 breast_s3 breast_s6)

# Loop through each slide ID
for slide_id in "${slide_ids[@]}"; do
    echo "Processing slide: $slide_id"

    # Run the .py file
    time python /gpfs/users/user/HE2CellType/CT_DS/src/cluster_features.py \
        --slide_id "$slide_id"

    echo "Finished processing slide: $slide_id"
    echo "======================================================================"
    echo " "
done