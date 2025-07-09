"""
To process spatial transcriptomic data by segmenting cell nuclei and aggregating transcripts per cell
and aligning H&E for each xenium data => save processed sdata
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import tiffslide
import xarray
from xarray import DataArray

import anndata
from spatialdata.transformations import set_transformation, Identity, Scale
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from datatree import DataTree
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
import spatialdata_io

from sopa.io import align
from sopa.segmentation import StainingSegmentation, Patches2D, Aggregator
from sopa.segmentation.methods import cellpose_patch
from sopa.segmentation.shapes import solve_conflicts
from sopa._sdata import get_intrinsic_cs, get_spatial_image




#### To open WSI - code adatpted from sopa for histopathology ####
# -> adapted to get only scale 0 and 1 of images because of corruption to save in zarr file


def _get_scale_transformation(scale_factor: float):
    if scale_factor == 1:
        return Identity()
    return Scale([scale_factor, scale_factor], axes=("x", "y"))


def _open_wsi(path: str | Path) -> tuple[str, xarray.Dataset, Any, dict]:

    image_name = Path(Path(path).stem).stem
    print(image_name)

    tiff = tiffslide.open_slide(path)
    img = xarray.open_zarr(
        tiff.zarr_group.store,
        consolidated=False,
        mask_and_scale=False,
    )

    tiff_metadata = {
        "properties": tiff.properties,
        "dimensions": tiff.dimensions,
        "level_count": tiff.level_count,
        "level_dimensions": tiff.level_dimensions,
        "level_downsamples": tiff.level_downsamples,
    }
    return image_name, img, tiff, tiff_metadata


def wsi(
    path: str | Path, chunks: tuple[int, int, int] = (3, 256, 256), as_image: bool = False
) -> SpatialData:
    """Read a WSI into a `SpatialData` object

    Args:
        path: Path to the WSI
        chunks: Tuple representing the chunksize for the dimensions `(C, Y, X)`.
        as_image: If `True`, returns a, image instead of a `SpatialData` object

    Returns:
        A `SpatialData` object with a multiscale 2D-image of shape `(C, Y, X)`
    """
    image_name, img, tiff, tiff_metadata = _open_wsi(path)

    images = {}
    for level, key in enumerate(list(img.keys())):

        suffix = key if key != "0" else ""

        if img[key].dims==("C", f"Y{suffix}", f"X{suffix}"):
                    scale_image = SpatialImage(
                        img[key],
                        dims=("c", "y", "x"),
                        ).chunk(chunks)

        else:
            scale_image = SpatialImage(
                img[key].transpose("S", f"Y{suffix}", f"X{suffix}"),
                dims=("c", "y", "x"),
            ).chunk(chunks)

        scale_factor = tiff.level_downsamples[level]

        scale_image = Image2DModel.parse(
            scale_image,
            transformations={"pixels": _get_scale_transformation(scale_factor)},
            c_coords=("r", "g", "b"),
        )
        scale_image.coords["y"] = scale_factor * scale_image.coords["y"]
        scale_image.coords["x"] = scale_factor * scale_image.coords["x"]

        images[f"scale{key}"] = scale_image

    multiscale_image = DataTree.from_dict(images)
    multiscale_image.attrs["metadata"] = tiff_metadata

    if as_image:
        multiscale_image.name = image_name
        return multiscale_image

    sdata = SpatialData(images={image_name: multiscale_image})
    sdata[image_name].attrs["metadata"] = tiff_metadata

    return sdata


#### End of code to open WSI ####



def segmentation(sdata, diameter, channels, cellpose_model, flow_threshold, cellprob_threshold, min_area, patch_width, patch_overlap, num_processes):

    channels = [channel for channel in channels]
    print(channels)

    method = cellpose_patch(diameter=diameter, channels=channels, model_type=cellpose_model, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
    segmentation = StainingSegmentation(sdata, method, channels, min_area=min_area)

    patches = Patches2D(sdata, 'morpho', patch_width=patch_width, patch_overlap=patch_overlap)
    patches.write()

    cellpose_temp_dir = 'cellpose_temp'
    def process_patch(patch_index):
        segmentation.write_patch_cells(cellpose_temp_dir, patch_index)
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_patch, range(len(patches)))  # range(153,155) FOR SAMPLING !!!

    cells = StainingSegmentation.read_patches_cells(cellpose_temp_dir)
    cells = solve_conflicts(cells)

    StainingSegmentation.add_shapes(sdata, cells, 'morpho', 'nucleus_boundaries')
    sdata.shapes['seg_patches'] = sdata.shapes.pop('sopa_patches')



def aggregate_transcripts(sdata, aggr, gene_key):

    if aggr=='yes':
        try:
            del sdata.table
        except:
            pass
    
    aggregator = Aggregator(sdata, image_key="morpho", shapes_key="nucleus_boundaries")
    aggregator.compute_table(gene_column=gene_key, average_intensities=False)



def align_he_image(sdata, he_path, align_matrix_path):

    he = wsi(he_path, as_image=True)

    if align_matrix_path is None:
        default_image = get_spatial_image(sdata, "morpho")
        pixel_cs = get_intrinsic_cs(sdata, default_image)
        set_transformation(he, {pixel_cs: Identity()}, set_all=True)
        sdata.images["he"] = he
    else:
        align(sdata, he, transformation_matrix_path=align_matrix_path, image_key="morpho")
        sdata.images["he"] = sdata.images.pop(he.name)





def main(args):
    
    # Load ST data and rename
    print("\n## Loading data ##")
    sdata = spatialdata_io.xenium(path=args.st_dir,
                                    n_jobs=1, transcripts=True, nucleus_boundaries=True, cells_boundaries=False,
                                    morphology_mip=False, morphology_focus=True,
                                    cells_table=False, aligned_images=False, nucleus_labels=False, cells_labels=False, cells_as_circles=False)
    sdata.images['morpho'] = sdata.images.pop(args.img_key)
    sdata.points['st'] = sdata.points.pop(args.st_key)
    print(sdata)

    if args.seg=='yes':
        try:
            sdata.shapes.pop('nucleus_boundaries')
            sdata.shapes.pop('cell_boundaries')
        except:
            pass
        # Perform segmentation
        print("\n## Performing segmentation ##")
        segmentation(sdata, args.diameter, args.channels, args.model_type,
                    args.flow_threshold, args.cellprob_threshold, args.min_area,
                    args.seg_patch_width, args.seg_patch_overlap, args.num_processes)

    # Aggregate transcripts
    print("\n## Transcripts aggregation ##")
    aggregate_transcripts(sdata, args.aggr, args.gene_key)

    # Load and align H&E image with spatial data
    print("\n## Loading and aligning H&E image ##")
    align_he_image(sdata, args.he_path, args.align_matrix_path)

    # Remove unnecessary data
    try:
        sdata.shapes.pop('cell_boundaries')
    except:
        pass
    try:
        sdata.shapes.pop('seg_patches')
    except:
        pass
    
    # Checking sdata
    print("\n## Checking sdata ##")
    print(sdata)

    # Save sdata
    print("\n## Saving processed sdata ##")
    os.makedirs(args.output_dir, exist_ok=True)
    sdata.write(os.path.join(args.output_dir, f"sdata_{args.slide_id}.zarr"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sdata (seg, aggr, align)")
    
    # Data paths
    parser.add_argument("--st_dir", type=str, help="Path to ST data directory",
                        default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/data_xenium10X/lung_s3/Xenium_Prime_Human_Lung_Cancer_FFPE_outs")
    
    parser.add_argument("--he_path", type=str, help="Path to H&E image", 
                        default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/data_xenium10X/lung_s3/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image.ome.tif")
    
    parser.add_argument("--align_matrix_path", type=lambda x : None if x == 'None' else str(x), help="Path to alignment matrix",
                        default=r"/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/data_xenium10X/lung_s3/Xenium_Prime_Human_Lung_Cancer_FFPE_he_imagealignment.csv")

    # Keys
    parser.add_argument("--img_key", type=str, default="morphology_focus", help="Image key in sdata")
    parser.add_argument("--st_key", type=str, default="transcripts", help="Transcripts key in sdata")
    parser.add_argument("--gene_key", type=str, default="feature_name", help="Gene column in st_key in sdata")
    
    # Cellpose segmentation
    parser.add_argument("--seg", type=str, default='no', help="Performing cellpose segmentation, choose 'yes' or 'no'")
        ## patching 
    parser.add_argument("--seg_patch_width", type=int, default=1200, help="Patch width")
    parser.add_argument("--seg_patch_overlap", type=int, default=50, help="Patch overlap")
        ## segmentation
    parser.add_argument("--channels", nargs='+', default=['0'], help="Channels for segmentation, either one or two integers")
    parser.add_argument("--model_type", type=str, default='nuclei', help="Type of cellpose model")
    parser.add_argument("--diameter", type=int, default=30, help="Diameter for cellpose segmentation") # see sopa config for xenium
    parser.add_argument("--flow_threshold", type=int, default=2, help="Flow threshold for cellpose segmentation") # see sopa config for xenium
    parser.add_argument("--cellprob_threshold", type=int, default=-6, help="Cell probability threshold for cellpose segmentation") # see sopa config for xenium
    parser.add_argument("--min_area", type=int, default=400, help="Minimum area (in pixels^2) for a cell to be kept") # cell 10µm and then convert with the given method resolution (here xenium)
    parser.add_argument("--num_processes", type=int, default=3, help="Number of processes for segmentation")

    # Transcripts aggregation
    parser.add_argument("--aggr", type=str, default='yes', help="Performing aggregation, choose 'yes' or 'no'")

    # Save results
    parser.add_argument("--output_dir", type=str, default='/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/sdata_processed', help="Output directory")
    parser.add_argument("--slide_id", type=str, default='lung_s3', help="Slide ID")
    
    args = parser.parse_args()
    main(args)



# NB for mac: go in the morphology_focus folder and run the following command to delete weird files: rm ._*