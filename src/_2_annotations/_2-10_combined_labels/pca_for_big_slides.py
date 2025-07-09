"""
Do the PCA using GPU for big slides
"""

import argparse
import rapids_singlecell as rsc
import cupy as cp
import anndata as ad
import os

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

cgroup_gpu_ids = os.getenv('CUDA_VISIBLE_DEVICES', '')
gpu_devices = [int(id) for id in cgroup_gpu_ids.split(',')]

rmm.reinitialize(
    managed_memory=True,
    pool_allocator=False,
    devices=gpu_devices,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)


def pca_rapid(adata):
    '''GPU accelerated implementation'''

    adata.X = adata.X.astype('float32')

    # Normalize
    print("Normalization")
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)

    # Scale each gene to unit variance. Clip values exceeding standard deviation 10.
    print("Scaling")
    rsc.pp.scale(adata, max_value=10)

    # Reduce the dimensionality of the data by running principal component analysis (PCA), which reveals the main axes of variation and denoises the data.
    print("PCA")
    rsc.pp.pca(adata, use_highly_variable=False)

    # Move all adata to CPU
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    



def main(args):

    # Load sdata
    print(f"\nLoading adata")
    adata = ad.read_h5ad(os.path.join(args.adatas_folder, f"adata_{args.slide_id}.h5ad"))
    print(f"adata: {adata}")

    # PCA
    print("\n-> Using GPU <-")
    print(f'Available gpu devices: {gpu_devices}')
    pca_rapid(adata)

    print("\nResults:", adata)

    # Save
    print("\nSaving...")
    os.makedirs(args.save_folder, exist_ok=True)
    adata.write_h5ad(os.path.join(args.save_folder, f"adata_pca_{args.slide_id}.h5ad"))
    print(adata)
    print("Done.")

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Clustering using GPU for big slides")
    
    parser.add_argument("--slide_id", type=str, help="Slide id and type (nuclei or cyto)", default="nuclei_breast_s6")
    parser.add_argument("--adatas_folder", type=str, help="Folder to adata files", default=r"/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/adata_big_pca")
    parser.add_argument("--save_folder", type=str, help="Folder to save clustering results", default=r"/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/adata_big_pca")
    
    args = parser.parse_args()
    main(args)