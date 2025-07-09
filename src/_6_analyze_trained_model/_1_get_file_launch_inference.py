"""Get cell count csv for each slide to perform inference then"""

import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json


def apply_grouping(df, grouping):
    df_new = df.copy()
    for new_cat, cols in grouping.items():
        missing_cols = [c for c in cols if c not in df_new.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} are missing in df_cell_count for grouping '{new_cat}'.")
        df_new[new_cat] = df_new[cols].sum(axis=1)
        df_new.drop(columns=cols, inplace=True)
    return df_new


def make_final_df(slide_id, path_prepared_dataset, grouping=None):
    print("- Loading data...")
    df_cell_count = pd.read_csv(os.path.join(path_prepared_dataset, "ALL/cell_count.csv"))
    df_types = pd.read_csv(os.path.join(path_prepared_dataset, "ALL/types.csv"))
    df_patch_metrics = pd.read_csv(os.path.join(path_prepared_dataset, "ALL/patch_metrics.csv"))

    if grouping:
        print("- Applying grouping using dict:", grouping)
        df_cell_count = apply_grouping(df_cell_count, grouping)

    print("- Merging data...")
    df_final = pd.merge(df_cell_count, df_types, left_on='Image', right_on='img')
    df_final.drop(columns=['Image'], inplace=True)
    df_final = pd.merge(df_final, df_patch_metrics, left_on='img', right_on='patch_id')
    df_final.drop(columns=['patch_id'], inplace=True)
    df_final['slide_id'] = df_final['img'].str.rsplit('_', n=1).str[0]

    df_final_filtered = df_final[df_final['slide_id'] == slide_id].copy()
    cell_types_cols = [col for col in df_cell_count.columns if col != 'Image']

    df_final_filtered['set'] = 'test'
    return df_final_filtered, cell_types_cols, df_cell_count


def create_plots(df_final_filtered, cell_types_cols, save_dir):
    df = df_final_filtered.copy()

    # FIGURE 1: Tissues
    print("- Making fig1...")
    fig1, axs = plt.subplots(1, 2, figsize=(20, 7))
    
    sns.countplot(data=df, x="type", hue="set", order=df['type'].unique(), ax=axs[0])
    axs[0].set_title("Patch count per tissue")
    axs[0].set_xlabel("Tissue type")
    axs[0].set_ylabel("Patch count")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=3)

    tissue_counts = df.groupby(['set', 'type'], observed=True).size().reset_index(name='count')
    tissue_totals = tissue_counts.groupby('set', observed=True)['count'].transform('sum')
    tissue_counts['percent'] = (tissue_counts['count'] / tissue_totals) * 100
    pivot_tissue = tissue_counts.pivot(index="set", columns="type", values="percent").fillna(0)
    pivot_tissue.plot(kind="bar", stacked=True, ax=axs[1])
    axs[1].set_title("Percentage of tissue type")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("Percentage of patches")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/tissues_distribution.png")

    # FIGURE 2: Slide_ids
    print("- Making fig2...")
    fig2, axs = plt.subplots(1, 2, figsize=(20, 7))
    sns.countplot(data=df, x="slide_id", hue="set", order=df['slide_id'].unique(), ax=axs[0])
    axs[0].set_title("Patch count per slide_id")
    axs[0].set_xlabel("Slide ID")
    axs[0].set_ylabel("Patch Count")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.38), ncol=3)

    slide_counts = df.groupby(['set', 'slide_id'], observed=True).size().reset_index(name='count')
    slide_totals = slide_counts.groupby('set', observed=True)['count'].transform('sum')
    slide_counts['percent'] = (slide_counts['count'] / slide_totals) * 100
    pivot_slide = slide_counts.pivot(index="set", columns="slide_id", values="percent").fillna(0)
    pivot_slide.plot(kind="bar", stacked=True, ax=axs[1])
    axs[1].set_title("Percentage per slide_id")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("Percentage of patches")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/slide_ids_distribution.png")

    # FIGURE 3: Cell types
    print("- Making fig3...")
    fig3, axs = plt.subplots(1, 2, figsize=(14, 6))
    cell_counts = df.groupby(['set'], observed=True)[cell_types_cols].sum().reset_index()
    ax1 = cell_counts.set_index("set")[cell_types_cols].plot(kind="bar", stacked=True, ax=axs[0], colormap="tab20", legend=False)
    axs[0].set_title("Cell types count")
    axs[0].set_xlabel("Fold")
    axs[0].set_ylabel("Cell count")

    cell_totals = cell_counts[cell_types_cols].sum(axis=1)
    cell_percentages = cell_counts[cell_types_cols].div(cell_totals, axis=0) * 100
    cell_percentages['set'] = cell_counts['set']
    ax2 = cell_percentages.set_index("set").plot(kind="bar", stacked=True, ax=axs[1], colormap="tab20", legend=False)
    axs[1].set_title("Cell types percentage")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("Cell percentage")

    handles, labels = ax1.get_legend_handles_labels()
    fig3.legend(handles, labels, loc='lower center', ncol=4, title="Cell Types")
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.savefig(f"{save_dir}/cell_types_distribution.png")


def main(args):

    print(f"\n-------- Processing slide {args.slide_id} --------")

    grouping_dict = json.loads(args.grouping) if args.grouping else None
    save_dir = os.path.join(args.output_dir, args.dataset_id, "informations")
    os.makedirs(save_dir, exist_ok=True)

    print("\n** Making the final DataFrame...")
    df_final, cell_types_cols, df_cell_count = make_final_df(args.slide_id, args.path_prepared_dataset, grouping=grouping_dict)

    print("\n** Creating plots...")
    create_plots(df_final, cell_types_cols, save_dir)

    print("\n** Saving the final DataFrame...")
    df_final.to_csv(f"{save_dir}/infos_{args.dataset_id}.csv", index=False)

    print("\n** Saving the cell count...")
    df_cell_count[df_cell_count['Image'].isin(df_final['img'])].to_csv(os.path.join(args.output_dir, args.dataset_id, 'cell_count_test.csv'), index=False)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset info for a single slide")
    parser.add_argument("--slide_id", type=str, required=True, help="Slide ID to process")
    parser.add_argument("--path_prepared_dataset", type=str, default="/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/ct_1", help="Path to the prepared_datasets directory")
    parser.add_argument("--output_dir", type=str, default="/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/training_datasets/slide_specific_ds_1", help="Directory to save output")
    parser.add_argument("--dataset_id", type=str, required=True, help="Training dataset ID")
    parser.add_argument("--grouping", type=str, default=None, help="Optional JSON string for grouping cell types")  # used for ds_4

    args = parser.parse_args()
    main(args)



# -----------------------------------------
# Command line usage for this file :

# ds_1 => use slide_specific_ds_1 for output_dir
# for slide_id in breast_s0 breast_s1 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4 pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 prostate_s0 cervix_s0; do python3 src/_6_analyze_trained_model/_1_get_file_launch_inference.py --slide_id "$slide_id" --dataset_id "$slide_id"; done

# ds_4 => use slide_specific_ds_4 for output_dir
# for slide_id in breast_s0 breast_s1 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4 pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 prostate_s0 cervix_s0; do python3 src/_6_analyze_trained_model/_1_get_file_launch_inference.py --slide_id "$slide_id" --dataset_id "$slide_id" --grouping '{"Immune": ["T_NK", "B_Plasma", "Myeloid"], "Stromal": ["Blood_vessel", "Fibroblast_Myofibroblast"], "Other": ["Specialized", "Dead"]}'; done




