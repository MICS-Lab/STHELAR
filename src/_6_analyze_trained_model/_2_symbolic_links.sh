#!/bin/bash

# Usage:
# ./symbolic_links.sh <machine>
# Example:
# ./symbolic_links.sh local

# Check that the machine argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <machine: local|jeanzay>"
  exit 1
fi

machine="$1"

# Set HE2CT base path based on the machine
if [ "$machine" == "local" ]; then
  base_path="/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT"
elif [ "$machine" == "jeanzay" ]; then
  base_path="/lustre/fswork/projects/rech/bry/ubu16ws/HE2CellType/HE2CT"
else
  echo "Unknown machine: $machine"
  exit 1
fi

# Variables
cell_cat_id="ct_1"   # TO CHOOSE
training_dataset_id="ds_1"   # TO CHOOSE
training_id="training_27"    # TO CHOOSE
training_timestamp="2025-03-25T100556_training_27"  # TO CHOOSE
checkpoint_name="checkpoint_40.pth"  # TO CHOOSE

# Slides to loop over
slide_ids=("breast_s0" "breast_s1" "breast_s3" "breast_s6" "lung_s1" "lung_s3" "skin_s1" "skin_s2" "skin_s3" "skin_s4" "pancreatic_s0" "pancreatic_s1" "pancreatic_s2" "heart_s0" "colon_s1" "colon_s2" "kidney_s0" "kidney_s1" "liver_s0" "liver_s1" "tonsil_s0" "tonsil_s1" "lymph_node_s0" "ovary_s0" "ovary_s1" "prostate_s0" "cervix_s0")

# Source directory
source_dir_ct="${base_path}/prepared_datasets_cat"
source_dir_ds="${base_path}/training_datasets/${training_dataset_id}"

# Files to link for slide-specific training (machine-dependent path for labels)
if [ "$machine" == "local" ]; then
  labels_path="${source_dir_ct}/${cell_cat_id}/ALL/labels.zip"
  types_path="${source_dir_ct}/${cell_cat_id}/ALL/types.csv"
elif [ "$machine" == "jeanzay" ]; then
  labels_path="${source_dir_ct}/${cell_cat_id}/labels.zip"
  types_path="${source_dir_ct}/${cell_cat_id}/types.csv"
fi

files=(
    "${source_dir_ct}/images.zip"
    "${labels_path}"
    "${types_path}"
    "${source_dir_ct}/masks_cell_ids_nuclei.zip"
    "${source_dir_ds}/dataset_config.yaml"
    "${source_dir_ds}/weight_config.yaml"
)

# Loop over slide IDs
for slide_id in "${slide_ids[@]}"; do
  # Slide-specific training dataset directory
  target_dir="${base_path}/training_datasets/slide_specific_${training_dataset_id}/${slide_id}"
  mkdir -p "${target_dir}"

  for file in "${files[@]}"; do
    filename=$(basename "${file}")
    target="${target_dir}/${filename}"

    if [ ! -e "${target}" ]; then
      ln -s "${file}" "${target}"
      echo "Created symbolic link: ${target} -> ${file}"
    else
      echo "Link already exists: ${target}"
    fi
  done

  # Inference structure setup
  inference_slide_dir="${base_path}/inferences/${training_id}/${slide_id}"
  mkdir -p "${inference_slide_dir}/checkpoints"

  # Copy and modify config.yaml
  config_source="${base_path}/trainings/${training_id}/log/${training_timestamp}/config.yaml"
  config_target="${inference_slide_dir}/config.yaml"
  if [ ! -e "${config_target}" ]; then
    cp "${config_source}" "${config_target}"
    echo "Copied config.yaml for ${slide_id}"

    # Replace dataset_path line in the copied config
    slide_specific_path="${base_path}/training_datasets/slide_specific_${training_dataset_id}/${slide_id}"
    sed -i "s|^\( *dataset_path: \).*|\1${slide_specific_path}|" "${config_target}"
    echo "Modified dataset_path in config.yaml for ${slide_id}"
  else
    echo "config.yaml already exists for ${slide_id}"
  fi

  # Link to checkpoint inside "checkpoints" folder
  checkpoint_source="${base_path}/trainings/${training_id}/log/${training_timestamp}/checkpoints/${checkpoint_name}"
  checkpoint_target="${inference_slide_dir}/checkpoints/${checkpoint_name}"
  if [ ! -e "${checkpoint_target}" ]; then
    ln -s "${checkpoint_source}" "${checkpoint_target}"
    echo "Linked checkpoint for ${slide_id}"
  else
    echo "Checkpoint link already exists for ${slide_id}"
  fi
done