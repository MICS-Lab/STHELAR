"""
Do the clustering using GPU for big slides
"""

import argparse
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp
import anndata as ad
import os
import decoupler as dc
import numpy as np
import pandas as pd
import logging


logging.basicConfig(
    filename='/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/att_big_combined/clustering.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

cgroup_gpu_ids = os.getenv('CUDA_VISIBLE_DEVICES', '')
gpu_devices = [int(id) for id in cgroup_gpu_ids.split(',')]
logging.info(gpu_devices)

rmm.reinitialize(
    managed_memory=True,
    pool_allocator=False,
    devices=gpu_devices,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)


def cluster_rapid(adata_nuclei_ini, adata_cells_ini):
    '''GPU accelerated implementation'''

    # Define nucleus weight
    nucleus_weight = 1   ## /!\/!\/!\ CHOOSE NUCLEUS WEIGHT /!\/!\/!\ ## NB: final formula is like this: cytoplasmic & nucleus_counts + (nucleus weight * nucleus counts)

    # Align the cell_id in both AnnData objects
    # Create a mapping of cell_id to index position in both objects
    nuclei_idx = adata_nuclei_ini.obs.set_index('cell_id').index
    cells_idx = adata_cells_ini.obs.set_index('cell_id').index

    # Find the intersection of cell_id in both datasets
    common_cells = nuclei_idx.intersection(cells_idx)
    logging.info(f"Number of common cell_ids: {len(common_cells)}")

    # Subset both AnnData objects to include only common cell_ids
    adata_nuclei = adata_nuclei_ini[adata_nuclei_ini.obs['cell_id'].isin(common_cells)]
    adata_cells = adata_cells_ini[adata_cells_ini.obs['cell_id'].isin(common_cells)]

    # Merge the 'obs' dataframes based on 'cell_id' to ensure the same order in both
    obs_nuclei = adata_nuclei.obs[['cell_id']].reset_index(drop=True)
    obs_cells = adata_cells.obs[['cell_id']].reset_index(drop=True)

    # Ensure that both are in the same order
    adata_nuclei = adata_nuclei[obs_nuclei['cell_id'].argsort().values]
    adata_cells = adata_cells[obs_cells['cell_id'].argsort().values]

    # Check that the cell_ids match now
    assert np.array_equal(adata_nuclei.obs['cell_id'].values, adata_cells.obs['cell_id'].values), "Cell IDs do not match after alignment."

    # Find common genes
    common_genes = adata_nuclei.var_names.intersection(adata_cells.var_names)
    logging.info(f"Number of common genes: {len(common_genes)}")

    # Subset both AnnData objects to include only the common genes
    adata_nuclei = adata_nuclei[:, common_genes]
    adata_cells = adata_cells[:, common_genes]

    # Ensure X is of type float32 before moving to GPU
    adata_nuclei = adata_nuclei.copy()
    adata_cells = adata_cells.copy()
    adata_nuclei.X = adata_nuclei.X.astype('float32')
    adata_cells.X = adata_cells.X.astype('float32')

    # Move all adata to GPU
    rsc.get.anndata_to_GPU(adata_nuclei, convert_all=True)
    rsc.get.anndata_to_GPU(adata_cells, convert_all=True)

    # Normalize the RNA counts
    rsc.pp.normalize_total(adata_nuclei, target_sum=1e4)
    rsc.pp.normalize_total(adata_cells, target_sum=1e4)

    # Log-transform the RNA counts
    rsc.pp.log1p(adata_nuclei)
    rsc.pp.log1p(adata_cells)

    # Get the matrix from both datasets
    nucleus = adata_nuclei.X
    cytoplasmic_nucleus = adata_cells.X

    # Compute the weighted RNA counts
    weighted_rna_counts = cytoplasmic_nucleus + (nucleus_weight * nucleus)

    # Create a new AnnData object with the weighted RNA counts
    adata_combined = sc.AnnData(X=weighted_rna_counts, 
                                obs=adata_nuclei.obs.copy(), 
                                var=pd.DataFrame(index=common_genes))

    # Keep the 'spatial' information
    adata_combined.obsm['spatial'] = adata_nuclei.obsm['spatial']

    # Checking
    logging.info(f"adata_combined: {adata_combined}")
    logging.info(f"adata_combined.obs: {adata_combined.obs}")
    logging.info(f"adata_combined.var: {adata_combined.var}")

    # Ensure X is of type float32 before moving to GPU
    adata_combined = adata_combined.copy()
    adata_combined.X = adata_combined.X.astype('float32')

    # Normalize
    logging.info("Normalization")
    rsc.get.anndata_to_GPU(adata_combined, convert_all=True)
    rsc.pp.normalize_total(adata_combined, target_sum=1e4)
    rsc.pp.log1p(adata_combined)
    adata_combined.layers['log_norm'] = adata_combined.X.copy()

    # Scale each gene to unit variance. Clip values exceeding standard deviation 10.
    logging.info("Scaling")
    rsc.pp.scale(adata_combined, max_value=10)

    # Reduce the dimensionality of the data by running principal component analysis (PCA), which reveals the main axes of variation and denoises the data.
    logging.info("PCA")
    rsc.pp.pca(adata_combined, use_highly_variable=False)

    # Before swapping layers, move data to CPU
    logging.info("Moving to CPU for decoupler operation")
    rsc.get.anndata_to_CPU(adata_combined, convert_all=True)

    # Restore X to be log norm counts
    # /!\ ALL the following analysis will use the 'log_norm' layer as input /!\
    logging.info("Restore X to be log norm counts")
    dc.swap_layer(adata_combined, 'log_norm', X_layer_key=None, inplace=True)

    # Move data back to GPU for further operations
    logging.info("Moving data back to GPU")
    rsc.get.anndata_to_GPU(adata_combined, convert_all=True)

    # Checking if values are different (important)
    logging.info(f"Sum log_norm: {adata_combined.layers['log_norm'].sum()}")
    logging.info(f"Sum X: {adata_combined.X.sum()}")

    # Neighborhood graph of cells using the PCA representation of the data matrix
    logging.info("Neighborhood graph")
    rsc.pp.neighbors(adata_combined, n_neighbors=20, n_pcs=16, use_rep="X_pca", key_added='pca_n20_pcs16')
    
    # Embedding the neighborhood graph using UMAP
    logging.info("UMAP")
    rsc.tl.umap(adata_combined, neighbors_key='pca_n20_pcs16', min_dist=0.6)
    
    # Leiden clustering directly clusters the neighborhood graph of cells
    logging.info("Leiden clustering / res0.2")
    rsc.tl.leiden(adata_combined, resolution=0.2, random_state=0, key_added='pca_n20_pcs16_leiden_res0.2', neighbors_key='pca_n20_pcs16')
    logging.info("Leiden clustering / res0.4")
    rsc.tl.leiden(adata_combined, resolution=0.4, random_state=0, key_added='pca_n20_pcs16_leiden_res0.4', neighbors_key='pca_n20_pcs16')
    logging.info("Leiden clustering / res0.6")
    rsc.tl.leiden(adata_combined, resolution=0.6, random_state=0, key_added='pca_n20_pcs16_leiden_res0.6', neighbors_key='pca_n20_pcs16')

    # Move all adata to CPU
    rsc.get.anndata_to_CPU(adata_combined, convert_all=True)

    # Checking number of clusters
    logging.info(f"N clusters for res0.2: {len(adata_combined.obs['pca_n20_pcs16_leiden_res0.2'].unique())}")
    logging.info(f"N clusters for res0.4: {len(adata_combined.obs['pca_n20_pcs16_leiden_res0.4'].unique())}")
    logging.info(f"N clusters for res0.6: {len(adata_combined.obs['pca_n20_pcs16_leiden_res0.6'].unique())}")

    return adata_combined
    



def main(args):

    # Load sdata
    logging.info(f"Loading adatas")
    adata_nuclei_ini = ad.read_h5ad(os.path.join(args.adatas_folder, f"adata_combined_nuclei_for_gpu_clustering_{args.slide_id}.h5ad"))
    adata_cells_ini = ad.read_h5ad(os.path.join(args.adatas_folder, f"adata_combined_cells_for_gpu_clustering_{args.slide_id}.h5ad"))
    logging.info(f"adata_nuclei_ini: {adata_nuclei_ini}")
    logging.info(f"adata_cells_ini: {adata_cells_ini}")

    # Clustering
    logging.info("\n-> Using GPU <-")
    logging.info(f'Available gpu devices: {gpu_devices}')
    adata_combined = cluster_rapid(adata_nuclei_ini, adata_cells_ini)

    # Save
    logging.info("Saving")
    os.makedirs(args.save_folder, exist_ok=True)
    adata_combined.write_h5ad(os.path.join(args.save_folder, f"adata_combined_gpu_clustering_{args.slide_id}.h5ad"))
    logging.info(adata_combined)
    logging.info("Done")

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Clustering using GPU for big slides")
    
    parser.add_argument("--slide_id", type=str, help="Slide id", default="breast_s6")
    parser.add_argument("--adatas_folder", type=str, help="Folder to adata files", default=r"/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/att_big_combined")
    parser.add_argument("--save_folder", type=str, help="Folder to save clustering results", default=r"/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/att_big_combined")
    
    args = parser.parse_args()
    main(args)