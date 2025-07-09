"""
Do the clustering using GPU for big slides
"""

import argparse
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp
import anndata as ad
import os
import logging
import decoupler as dc

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
    filename='/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/clustering.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def cluster_rapid(adata):
    '''GPU accelerated implementation'''

    # Ensure X is of type float32 before moving to GPU
    adata.X = adata.X.astype('float32')

    # Move all adata to GPU
    rsc.get.anndata_to_GPU(adata, convert_all=True)

    # Save the raw data
    adata.layers['count'] = adata.X.copy()

    # Normalize
    logging.info("Normalization")
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    adata.layers['log_norm'] = adata.X.copy()

    # Scale each gene to unit variance. Clip values exceeding standard deviation 10
    logging.info("Scaling")
    rsc.pp.scale(adata, max_value=10)

    # Reduce the dimensionality of the data by running principal component analysis (PCA), which reveals the main axes of variation and denoises the data
    logging.info("PCA")
    rsc.pp.pca(adata, use_highly_variable=False)

    # Before swapping layers, move data to CPU
    logging.info("Moving to CPU for decoupler operation")
    rsc.get.anndata_to_CPU(adata, convert_all=True)

    # Restore X to be log norm counts
    # /!\ ALL the following analysis will use the 'log_norm' layer as input /!\
    dc.swap_layer(adata, 'log_norm', X_layer_key=None, inplace=True)

    # Move data back to GPU for further operations
    logging.info("Moving data back to GPU")
    rsc.get.anndata_to_GPU(adata, convert_all=True)

    # Checking if values are different (important)
    logging.info(f"Sum count: {adata.layers['count'].sum()}")
    logging.info(f"Sum log_norm: {adata.layers['log_norm'].sum()}")
    logging.info(f"Sum X: {adata.X.sum()}")

    # Neighborhood graph of cells using the PCA representation of the data matrix
    logging.info("Neighborhood graph")
    rsc.pp.neighbors(adata, n_neighbors=20, n_pcs=16, use_rep="X_pca", key_added='pca_n20_pcs16')
    
    # Embedding the neighborhood graph using UMAP
    logging.info("UMAP")
    rsc.tl.umap(adata, neighbors_key='pca_n20_pcs16', min_dist=0.6)
    
    # Leiden clustering directly clusters the neighborhood graph of cells
    logging.info("Leiden clustering / res0.2")
    rsc.tl.leiden(adata, resolution=0.2, random_state=0, key_added='pca_n20_pcs16_leiden_res0.2', neighbors_key='pca_n20_pcs16')
    logging.info("Leiden clustering / res0.4")
    rsc.tl.leiden(adata, resolution=0.4, random_state=0, key_added='pca_n20_pcs16_leiden_res0.4', neighbors_key='pca_n20_pcs16')
    logging.info("Leiden clustering / res0.6")
    rsc.tl.leiden(adata, resolution=0.6, random_state=0, key_added='pca_n20_pcs16_leiden_res0.6', neighbors_key='pca_n20_pcs16')

    # Move all adata to CPU
    rsc.get.anndata_to_CPU(adata, convert_all=True)

    # Checking number of clusters
    logging.info(f"N clusters for res0.2: {len(adata.obs['pca_n20_pcs16_leiden_res0.2'].unique())}")
    logging.info(f"N clusters for res0.4: {len(adata.obs['pca_n20_pcs16_leiden_res0.4'].unique())}")
    logging.info(f"N clusters for res0.6: {len(adata.obs['pca_n20_pcs16_leiden_res0.6'].unique())}")

    return adata
    



def main(args):

    # Load sdata
    logging.info(f"Loading adata")
    adata = ad.read_h5ad(args.adata_path)
    logging.info(adata)

    # Clustering
    logging.info("\n-> Using GPU <-")
    logging.info(f'Available gpu devices: {gpu_devices}')
    adata = cluster_rapid(adata)

    # Save
    logging.info("Saving")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    adata.write_h5ad(args.save_path)
    logging.info(adata)
    logging.info("Done")

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Clustering using GPU for big slides")
    
    parser.add_argument("--adata_path", type=str, help="Path to adata file", default=r"/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/adata_for_gpu_clustering_lymph_node_s0.h5ad")
    parser.add_argument("--save_path", type=str, help="Path to save the adata file with clustering results", default=r"/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/adata_after_gpu_clustering_lymph_node_s0.h5ad")
    
    args = parser.parse_args()
    main(args)