#!/bin/bash
#SBATCH --job-name=get_scvi_models_3
#SBATCH --output=%x.o%j 
#SBATCH --time=03:00:00 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=13G
#SBATCH --mail-type=ALL

# Load necessary modules
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.2.1/gcc-11.2.0

# Activate anaconda environment
source activate scvi

# Run python script
time python /gpfs/users/user/HE2CellType/CT_DS/src/get_scvi_models_ruche.py