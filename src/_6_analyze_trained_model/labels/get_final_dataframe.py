"""
Merge slide‑level cell‑type predictions from several training runs into the global annotation CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq




def prepare_prediction_columns(df: pd.DataFrame, col_prefix: str) -> Tuple[str, str]:
    """
    Add empty columns to *df* that will later receive the predictions
    and return their names.
    """
    cell_type_col = f"{col_prefix}_cell_type"
    proba_col = f"{col_prefix}_proba"

    df[cell_type_col] = pa.scalar(None, type=pa.string())
    df[proba_col] = pa.scalar(None, type=pa.float32())
    return cell_type_col, proba_col


def parquet_file_for_slide(
    parquet_root: Path, training_id: str, slide_id: str
) -> Path:
    """
    Build the path to the Parquet file for a given slide_id and training_id.
    """
    return (
        parquet_root
        / training_id
        / "trained_predictions"
        / f"trained_labels_{training_id}_{slide_id}.parquet"
    )


def read_slide_parquet(
    parquet_path: Path,
    parquet_cell_type_col: str,
    parquet_proba_col: str,
    dest_cell_type_col: str,
    dest_proba_col: str,
) -> pd.DataFrame:
    """
    Read the Parquet file for one slide, and rename columns to match destination.
    """
    table = pq.read_table(
        parquet_path,
        columns=["cell_id", parquet_cell_type_col, parquet_proba_col],
    )
    p_df = table.to_pandas(types_mapper=pd.ArrowDtype)

    # Drop duplicate cell_id values, keeping the first occurrence
    dup_mask = p_df.duplicated(subset="cell_id", keep=False)
    if dup_mask.any():
        n_dups = dup_mask.sum()
        print(f"{parquet_path.name}: found {n_dups} duplicate cell_id(s) in Parquet – keeping the first")
        p_df = p_df.drop_duplicates(subset="cell_id", keep="first")

    # Rename to match destination column names
    p_df = p_df.rename(columns={
        parquet_cell_type_col: dest_cell_type_col,
        parquet_proba_col: dest_proba_col,
    })

    return p_df


def merge_slide_predictions(
    master_df: pd.DataFrame,
    slide_id: str,
    slide_predictions: pd.DataFrame,
    cell_type_col: str,
    proba_col: str,
) -> None:
    """
    Merge the Arrow‑backed slide_predictions into the slice of master_df that
    corresponds to this slide_id.
    """
    mask = master_df["slide_id"] == slide_id
    if not mask.any():
        print(f"No rows found in CSV for slide_id='{slide_id}' – skipping")
        return

    join_df = (
        master_df.loc[mask, ["cell_id"]]
        .merge(slide_predictions, on="cell_id", how="left", copy=False, validate="1:1")
    )

    master_df.loc[mask, cell_type_col] = join_df[cell_type_col].to_numpy()
    master_df.loc[mask, proba_col] = join_df[proba_col].to_numpy()





def main(args: argparse.Namespace) -> None:

    # Load the annotation CSV once
    print(f"\n** Loading CSV file: {args.csv_path}")
    USECOLS = [
        "slide_id",
        "cell_id",
        "final_label_nuclei",
        "final_label_cells",
        "PanNuke_label",
        "PanNuke_proba",
        "final_label_combined",
    ]
    df = pd.read_csv(Path(args.csv_path), usecols=USECOLS, dtype_backend="pyarrow", low_memory=False)
    print(df.head())

    # Prepare lists from comma‑separated CLI arguments
    training_ids = [t.strip() for t in args.training_ids.split(",") if t.strip()]
    slide_ids = [s.strip() for s in args.slide_ids.split(",") if s.strip()]
    parquet_root = Path(args.parquet_root)

    # For each training_id : add columns & merge predictions
    print(f"\n** Processing {len(training_ids)} training IDs:")
    for training_id in training_ids:
        
        col_prefix = training_id.replace("training_", "t") or training_id
        print(f"\n>>> Processing training_id '{training_id}'")

        cell_type_col, proba_col = prepare_prediction_columns(df, col_prefix)

        for slide in tqdm(slide_ids):
            
            parquet_path = parquet_file_for_slide(parquet_root, training_id, slide)
            if not parquet_path.is_file():
                print(f"Parquet file not found for slide_id='{slide}' ({parquet_path})")
                continue

            parquet_cell_type_col = f"{training_id}_cell_type"
            parquet_proba_col = f"{training_id}_proba"

            print(f"Merging {training_id} predictions for slide {slide}")
            slide_df = read_slide_parquet(
                            parquet_path,
                            parquet_cell_type_col=parquet_cell_type_col,
                            parquet_proba_col=parquet_proba_col,
                            dest_cell_type_col=cell_type_col,
                            dest_proba_col=proba_col,
                        )
            merge_slide_predictions(df, slide, slide_df, cell_type_col, proba_col)

    # Write the merged table
    print(f"\n** Writing merged CSV file: {args.output_path}")
    print(df.head())
    df.to_csv(Path(args.output_path), index=False)
    print(f"\nDone.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge slide‑specific trained predictions from multiple training runs into the main annotation CSV."
    )

    parser.add_argument(
        "--training_ids",
        type=str,
        default="training_27,training_28",
        help="Comma‑separated list of training run identifiers (e.g. 'training_27,training_28').",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/annots/adata_compare_annots_v2.csv",
        help="Path to adata_compare_annots_v2.csv.",
    )
    parser.add_argument(
        "--parquet_root",
        type=str,
        default="/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/analyze_trained_model",
        help=("Root folder that contains '{training_id}/trained_predictions'."),
    )
    parser.add_argument(
        "--slide_ids",
        type=str,
        default="breast_s0,breast_s1,breast_s3,breast_s6,lung_s1,lung_s3,skin_s1,skin_s2,skin_s3,skin_s4,pancreatic_s0,pancreatic_s1,pancreatic_s2,heart_s0,colon_s1,colon_s2,kidney_s0,kidney_s1,liver_s0,liver_s1,tonsil_s0,tonsil_s1,lymph_node_s0,ovary_s0,ovary_s1,prostate_s0,cervix_s0",
        help="Comma‑separated list of slide_id values to process.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/analyze_trained_model/trained_labels_comparison.csv",
        help="Output file path (.csv).",
    )

    arguments = parser.parse_args()
    main(arguments)