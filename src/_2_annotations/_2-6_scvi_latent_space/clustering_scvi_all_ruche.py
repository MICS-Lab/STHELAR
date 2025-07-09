"""
UMAP + clustering on scVI embeddings for adata_all
"""

import argparse
import scanpy as sc
import scanpy.external as sce
import rapids_singlecell as rsc
import cupy as cp
import anndata as ad
import os
import logging

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

cgroup_gpu_ids = os.getenv('CUDA_VISIBLE_DEVICES', '')
gpu_devices = [int(id) for id in cgroup_gpu_ids.split(',')]
print(gpu_devices)

rmm.reinitialize(
    managed_memory=True,
    pool_allocator=False,
    devices=gpu_devices,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)


logging.basicConfig(
    filename='/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/adata_scvi/clustering_scvi_all_filtered.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def cluster_rapid(adata_all):
    '''GPU accelerated implementation'''

    # Move all adata to GPU
    rsc.get.anndata_to_GPU(adata_all, convert_all=True)

    # Neighbors graph
    print(f"\n### 2. Neighbors graph ###\n")
    logging.info(f"\n### 2. Neighbors graph ###\n")
    rsc.pp.neighbors(adata_all, n_neighbors=10, use_rep="X_scVI", key_added='all_scVI_n10')
    print("Done")
    logging.info("Done")

    # UMAP
    print(f"\n### 3. UMAP ###\n")
    logging.info(f"\n### 3. UMAP ###\n")
    rsc.tl.umap(adata_all, neighbors_key='all_scVI_n10')
    print("Done")
    logging.info("Done")

    # Clustering
    print(f"\n### 4. Clustering ###\n")
    logging.info(f"\n### 4. Clustering ###\n")
    
    # print("Resolution=0.2")
    # logging.info("Resolution=0.2")
    # rsc.tl.leiden(adata_all, resolution=0.2, random_state=0, key_added='all_scVI_n10_leiden_res0.2', neighbors_key='all_scVI_n10')
    
    print("Resolution=0.4")
    logging.info("Resolution=0.4")
    rsc.tl.leiden(adata_all, resolution=0.4, random_state=0, key_added='all_scVI_n10_leiden_res0.4', neighbors_key='all_scVI_n10')
    
    print("Resolution=0.6")
    logging.info("Resolution=0.6")
    rsc.tl.leiden(adata_all, resolution=0.6, random_state=0, key_added='all_scVI_n10_leiden_res0.6', neighbors_key='all_scVI_n10')
    
    # print("Resolution=1.0")
    # logging.info("Resolution=1.0")
    # rsc.tl.leiden(adata_all, resolution=1.0, random_state=0, key_added='all_scVI_n10_leiden_res1.0', neighbors_key='all_scVI_n10')
    
    print("Done")
    logging.info("Done")

    # Move all adata to CPU
    rsc.get.anndata_to_CPU(adata_all, convert_all=True)

    return adata_all
    



def main(args):

    # Load sdata
    print(f"\n### 1. Loading sdata ###\n")
    logging.info(f"\n### 1. Loading sdata ###\n")
    adata_all = ad.read_h5ad(args.adata_all_path)
    print(adata_all)
    logging.info(adata_all)

    # Clustering
    print(f'Available gpu devices: {gpu_devices}')
    logging.info(f'Available gpu devices: {gpu_devices}')
    adata_all = cluster_rapid(adata_all)

    # Save
    print(f"\n### 5. Saving ###\n")
    logging.info(f"\n### 5. Saving ###\n")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata_all.write_h5ad(args.save_path)
    print(adata_all)
    logging.info(adata_all)
    print("Done")
    logging.info("Done")

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="UMAP + clustering on scVI embeddings")
    
    parser.add_argument("--adata_all_path", type=str, help="Path to adata_all file", default=r"/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/adata_scvi/adata_scvi_all_filtered.h5ad")
    parser.add_argument("--save_path", type=str, help="Path to save the adata_all file with clustering results", default=r"/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/adata_scvi_output/adata_scvi_all_filtered.h5ad")
    
    args = parser.parse_args()
    main(args)