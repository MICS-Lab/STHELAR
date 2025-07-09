"""
Annotate cell type using Tangram => save annotated sdata
"""

import argparse
import os

import anndata
import spatialdata as sd
from sopa.annotation.tangram import tangram_annotate



def ct_annots_tangram(sdata, ref_path, ct_key):

    adata_ref = anndata.read_h5ad(ref_path)

    # # SAMPLING !!
    # adata_500 = sdata.table[:100]
    # del sdata.table
    # sdata.table = adata_500
    # #
    
    tangram_annotate(sdata, adata_ref, cell_type_key=ct_key, device="mps")
    sdata.table.obs['ct_tangram'] = sdata.table.obs[ct_key]
    del sdata.table.obs[ct_key]





def main(args):
    
    # Load ST data and rename
    print("\n## Loading data ##")
    sdata = sd.read_zarr(args.sdata_path, selection=('tables',))
    print(sdata)

    # Annotate cell types using Tangram
    print("\n## Cell types annotation using Tangram ##")
    ct_annots_tangram(sdata, args.ref_path, args.ct_key)

    # Checking
    print("\n## Checking ##")
    print(sdata.table.obs)
    
    # Save sdata
    print("\n## Saving tangram annotated table ##")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    sdata.write(args.save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate cell types using Tangram")
    
    # Data path
    parser.add_argument("--sdata_path", type=str, default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/sdata_processed/sdata_lung_s3.zarr", help="Path to processed sdata")

    # Cell type annotation using Tangram
    parser.add_argument("--ref_path", type=str, default='/Users/felicie-giraud-sauveur/Documents/HE2CellType/code/CT_DS/data/sc_atlas/disco/disco_lung_v2.0.h5ad', help="Path to the cell type reference file")
    parser.add_argument("--ct_key", type=str, default="ct", help="Key for cell type in ref file")

    # Results
    parser.add_argument("--save_path", type=str, default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/sdata_att/sdata_lung_s3.zarr", help="Path to save the annotated sdata")
    
    args = parser.parse_args()
    main(args)

