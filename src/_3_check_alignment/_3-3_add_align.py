"""
Before running this script, you need to applgy CellVit on the patches to predict the comparison segmentation mask:
In HE2CT folder:
- Prepare dataset using cell_segmentation/datasets/prepare_pannuke.py using ['Neoplastic','Inflammatory','Connective','Dead','Epithelial'] as classes
- Make the Macenko normalization using cell_segmentation/datasets/macenko_normalization.py (OR NOT ????)
- Convert the dataset to zip using cell_segmentation/datasets/convert_into_zip.py
- Apply CellVit inference (file cell_segmentation/inference/inference_cellvit_experiment_pannuke.py):
    - modifying config.yaml
    - de-commenting the part to get the predictions for instance_map and pixel count predictions for pannuke label for gt in inference_cellvit_experiment_pannuke.py (cf. See where there is CHOOSE OR NOT)
    - In case we want to use real cell type mask instead of fake one : Dynamically adapt mask to handle PanNuke categories in datasets/pannuke.py by uncommenting type_map[type_map != 0] = 1 at the end of the load_maskfile function (cf. CHOOSE)
    - and --cell_tokens nucleus will be useful for H&E features after
    - and using inference_cellvit_experiment_pannuke.py with in the terminal : python3 cell_segmentation/inference/inference_cellvit_experiment_pannuke.py --run_dir /Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/output_cellvit/heart_s0 --checkpoint_name CellViT-SAM-H-x40.pth --gpu mps --magnification 40 --cell_tokens nucleus
    OR use ruche with slurm_cellvit_checkalign.sh
Then using this file, add metrics for each patch in the sdata object.
!!!! If slide was too big to do inference in one time, use before the script optional_group_output_cellvit.py to group the output of CellVit in one unique files for the given slide. !!!!
"""

import argparse
import os
import json
import pandas as pd
import spatialdata as sd



def open_json_metrics(output_cellvit_folder, slide_id):
    
    json_path = os.path.join(output_cellvit_folder, f'{slide_id}/inference_results.json')
    with open(json_path, 'r') as file:
        metric_json_file = json.load(file)
    
    return metric_json_file




def build_df_metrics(metric_json_file):

    df_metrics = pd.DataFrame.from_dict(metric_json_file['image_metrics'], orient='index')

    df_metrics.reset_index(inplace=True)
    df_metrics.rename(columns={'index': 'image'}, inplace=True)

    return df_metrics




def add_metrics_in_sdata(sdata, df_metrics):

    he_patches = sdata.shapes['he_patches'].copy()

    df_metrics['patch_id'] = df_metrics['image'].str.replace('.png', '').astype(int)

    he_patches = he_patches.merge(df_metrics[['patch_id', 'Dice', 'Jaccard', 'bPQ']],
                                on='patch_id', how='left')

    he_patches[['Dice', 'Jaccard', 'bPQ']] = he_patches[['Dice', 'Jaccard', 'bPQ']].fillna(-1) # -1 will correspond to no cell in xenium mask

    sdata.shapes['he_patches']['Dice'] = he_patches['Dice']
    sdata.shapes['he_patches']['Jaccard'] = he_patches['Jaccard']
    sdata.shapes['he_patches']['bPQ'] = he_patches['bPQ']
    print(sdata.shapes['he_patches'].head())
    print("\n\n")
    print(sdata)

    print("\n\nSaving on disk...")
    sdata.delete_element_from_disk("he_patches")
    sdata.write_element("he_patches")
    print("Done.")




def main(args):

    print(f"\n==== Proccessing {args.slide_id} ====")

    # Get metrics output from CellVit
    print("Loading metrics...")
    metric_json_file = open_json_metrics(args.output_cellvit_folder, args.slide_id)
    df_metrics = build_df_metrics(metric_json_file)

    # Load sdata
    print("Loading sdata...")
    sdata_path = os.path.join(args.sdata_folder, f'sdata_{args.slide_id}.zarr')
    sdata = sd.read_zarr(sdata_path, selection=('shapes',))

    try:
        del sdata.shapes['he_patches']['Dice']
        del sdata.shapes['he_patches']['Jaccard']
        del sdata.shapes['he_patches']['bPQ']
    except:
        pass

    print(sdata.shapes['he_patches'].head())

    # Add metrics in sdata
    print("\n\nAdding metrics in sdata...")
    add_metrics_in_sdata(sdata, df_metrics)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add metrics to check alignment patches in sdata")
    
    parser.add_argument("--slide_id", type=str, default="heart_s0", help="Slide id")
    parser.add_argument("--output_cellvit_folder", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/output_cellvit", help="Output folder of CellVit for align checking")
    parser.add_argument("--sdata_folder", type=str, default="/Volumes/SAUV_FGS/MICS/data_HE2CellType/CT_DS/sdata_final", help="Folder containing final sdata")

    args = parser.parse_args()
    main(args)
