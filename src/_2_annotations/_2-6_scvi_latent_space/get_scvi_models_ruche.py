import scanpy as sc
import argparse
import scvi
import torch
import os

scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")


def get_scvi_models(path_adata_ini, path_adata_final, path_model, slide_id):

    adata_scvi = sc.read_h5ad(path_adata_ini)
    print("\nadata_scvi:", adata_scvi)

    adata_scvi.layers["counts"] = adata_scvi.X.copy()  # preserve counts

    ## TODO: COMMENT THIS FOR ADATA_SCVI_ALL ##
    sc.pp.normalize_total(adata_scvi, target_sum=1e4)
    sc.pp.log1p(adata_scvi)
    adata_scvi.raw = adata_scvi  # freeze the state in `.raw`
    ## END ##

    scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts")   # TODO # ADD batch_key="sample" for adata_scvi_all

    model = scvi.model.SCVI(adata_scvi)
    print("\nmodel:", model)

    print("\ntraining:")

    if slide_id in ['pancreatic_s0']:
        model.train(accelerator='gpu',
                    early_stopping=True,              # Enable early stopping
                    early_stopping_monitor='reconstruction_loss_validation',  # Monitor the reconstruction_loss_validation
                    early_stopping_min_delta=0.1,    # Minimum improvement needed to continue
                    early_stopping_patience=10,       # Number of epochs to wait without improvement
                    early_stopping_mode='min',        # We want to minimize the reconstruction_loss_validation
                    enable_progress_bar=True,
                    max_epochs=100,
                    learning_rate_monitor=True,
                    plan_kwargs={"optimizer": 'AdamW', "n_epochs_kl_warmup":400, "reduce_lr_on_plateau":True, "lr_patience": 5, "lr": 1e-4},
                    datasplitter_kwargs={"drop_last": True})
    else:
        model.train(accelerator='gpu',
                    early_stopping=True,              # Enable early stopping
                    early_stopping_monitor='reconstruction_loss_validation',  # Monitor the reconstruction_loss_validation
                    early_stopping_min_delta=0.1,    # Minimum improvement needed to continue
                    early_stopping_patience=10,       # Number of epochs to wait without improvement
                    early_stopping_mode='min',        # We want to minimize the reconstruction_loss_validation
                    enable_progress_bar=True,
                    max_epochs=100,
                    learning_rate_monitor=True,
                    plan_kwargs={"optimizer": 'AdamW', "n_epochs_kl_warmup":400, "reduce_lr_on_plateau":True, "lr_patience": 5, "lr": 1e-4})        

    model.save(path_model, overwrite=True)

    adata_scvi.write_h5ad(path_adata_final)



def main(args):

    os.makedirs(args.folder_adata_final, exist_ok=True)
    os.makedirs(args.folder_model, exist_ok=True)

    slide_ids = [namefile.replace("adata_scvi_", "").replace(".h5ad", "") for namefile in os.listdir(args.folder_adata_ini)]
    already_done = ["bone_s0", "brain_s0", "breast_s0", "breast_s1", "kidney_s0", "kidney_s1", "ovary_s0", "ovary_s1", "pancreatic_s1", "pancreatic_s2", "skin_s4", "tonsil_s0"]

    for slide_id in slide_ids:

        if slide_id in already_done:
            print(f"Already done {slide_id}")
        
        else:

            print(f"\nProcessing slide {slide_id}:")
            path_adata_ini = os.path.join(args.folder_adata_ini, f"adata_scvi_{slide_id}.h5ad")
            path_adata_final = os.path.join(args.folder_adata_final, f"adata_scvi_{slide_id}.h5ad")
            path_model = os.path.join(args.folder_model, f"scvi_model_{slide_id}")

            get_scvi_models(path_adata_ini, path_adata_final, path_model, slide_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # WARNING: if adata_scvi_all then add batch_key="sample" in scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key="sample") + comment preprocessing
    parser.add_argument("--folder_adata_ini", type=str, default="/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/adata_scvi", help="Folder containing the initial scvi adata")
    parser.add_argument("--folder_adata_final", type=str, default="/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/adata_scvi_output", help="Folder containing the final scvi adata")
    parser.add_argument("--folder_model", type=str, default="/gpfs/workdir/giraudsfe/HE2CellType/CT_DS/data/models_scvi", help="Folder containing the scvi models")
    
    args = parser.parse_args()
    main(args)