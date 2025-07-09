#!/bin/bash

# Define the lists of slide IDs and tissue types
slide_ids=("lymph_node_s0" "ovary_s0" "ovary_s1" "prostate_s0" "cervix_s0")
tissue_types=("LymphNode" "Ovarian" "Ovarian" "Prostate" "Cervix")


# Loop through both lists
for i in "${!slide_ids[@]}"; do
    
    slide_id=${slide_ids[$i]}
    tissue_type=${tissue_types[$i]}
    
    # Define the paths based on slide_id and tissue_type
    sdata_path="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/sdata_final/sdata_${slide_id}.zarr"
    already_images_path="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium/${slide_id}/images.zip"
    output_dir="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/ds_slides_cat/ct_1/${slide_id}"
    
    # Execute the Python script
    python3 /Users/felicie-giraud-sauveur/Documents/HE2CellType/code/CT_DS/src/_5_build_final_dataset/_5-3_get_ds_per_slide_CAT.py \
        --sdata_path "$sdata_path" \
        --tissue_type "$tissue_type" \
        --already_images_path "$already_images_path" \
        --slide_id "$slide_id" \
        --output_dir "$output_dir"
done