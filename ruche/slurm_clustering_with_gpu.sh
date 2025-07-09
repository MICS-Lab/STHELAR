#!/bin/bash
#SBATCH --job-name=clustering_gpu_lymph_node_s0
#SBATCH --output=%x.o%j 
#SBATCH --time=00:15:00 
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpua100
#SBATCH --mem=90G
#SBATCH --mail-type=ALL

# Load necessary modules
module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Activate anaconda environment
source activate rapids_cluster

# Run python script
time python /gpfs/users/user/HE2CellType/CT_DS/src/clustering_with_gpu_for_big_slides.py