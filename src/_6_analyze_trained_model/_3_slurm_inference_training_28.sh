#!/bin/bash
#SBATCH --job-name=inference_t28         # nom du job
#SBATCH --output=./output/%x_%j.out     # fichier de sortie (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # fichier d’erreur (%j = job ID)
#SBATCH --constraint=h100               # demander des GPU A100 80 Go ou des GPU H100 80 Go
#SBATCH --nodes=1                       # reserver 1 nœuds
#SBATCH --ntasks=1                      # reserver 1 taches (ou processus)
#SBATCH --gres=gpu:1                    # reserver 1 GPU par noeud
#SBATCH --cpus-per-task=16               # reserver 4 CPU par tache (IMPORTANT pour la memoire dispo)
#SBATCH --time=20:00:00                 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread            # desactiver l’hyperthreading
#SBATCH --account=user@h100              # comptabilite A100 ou H100
#SBATCH --mail-type=ALL                 # When to send email notifications
#SBATCH --mail-user=mail  # Your email address

# Nettoyer les modules herites par defaut
module purge 

# Selectionner les modules compiles pour les A100 ou H100
module load arch/h100 

# Charger les modules
# No module to load for cuda or cudnn because already included in the conda env
module load miniforge/24.9.0

# Desactiver les environnements herites par defaut
conda deactivate 

# Activer environnement conda
conda activate cellvit_env

# Executer script inference

# List of slide IDs
# breast_s0 breast_s1 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4 pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 prostate_s0 cervix_s0
slide_ids=(ovary_s0 ovary_s1 prostate_s0 cervix_s0 breast_s0 breast_s3 breast_s6 breast_s1)

# Loop through each slide ID
for slide_id in "${slide_ids[@]}"; do
    echo "Processing slide: $slide_id"

    # Set the run directory based on the slide ID
    run_dir="/lustre/fswork/projects/rech/user/ubu16ws/HE2CellType/HE2CT/inferences/training_28/$slide_id"

    # Run the inference command
    time python /linkhome/rech/genrce01/ubu16ws/HE2CellType/HE2CT/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py \
        --run_dir "$run_dir" \
        --checkpoint_name checkpoint_32.pth \
        --gpu 0 \
        --magnification 40 \
        --cell_tokens nucleus

    echo "Finished processing slide: $slide_id"
    echo "======================================================="
    echo " "
done