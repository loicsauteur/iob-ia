import os
from typing import Optional
from zipfile import ZIP_DEFLATED

import numpy as np
import tifffile
from bioio_base.types import PhysicalPixelSizes
from bioio_ome_tiff.writers import OmeTiffWriter
from tifffile import imwrite

from iob_ia.custom_image import CustomImage


def save_image_channel(
    img: np.ndarray,
    path: str,
    channel_order: str = "ZYX",
    voxel_size: Optional[tuple] = None,
    channel_name: Optional[str] = None,
):
    """
    Save an image as tif file.

    Only if voxel_size is provided, the image will be saved as ome.tiff file,
    otherwise as plane tif (using the TIFFFILE library).

    :param img: image
    :param path: save path
    :param channel_order: channel order. Default is "ZYX"
    :param voxel_size: (optional) tuple of voxel size
    :param channel_name: list of str channel names
    :return:
    """
    if voxel_size is not None:
        if len(voxel_size) != 3:
            raise ValueError(
                f"Voxel size requires a tuple of length 3 (for X, Y, Z voxel sizes). "
                f"Got = {voxel_size}."
            )
        path = path.replace(".tif", ".ome.tif")
        OmeTiffWriter.save(
            img,
            path,
            dim_order=channel_order,
            channel_names=[channel_name],
            physical_pixel_sizes=PhysicalPixelSizes(*voxel_size),
        )

    else:
        imwrite(path, img)
    print(f"Saved image to: {path}")


def save_labels(labels: np.ndarray, path: str):
    """
    Save labels to .tif file (compressed).

    :param labels:
    :param path:
    :return:
    """
    imwrite(path, labels, compression=ZIP_DEFLATED)
    print(f"Saved labels to: {path}")


def gen_out_path(path: str, name: str = "output") -> str:
    """
    Generate output TIF-file path from input path and name.

    Create a save path with similar filename, e.g.:
    p = gen_out_path('C:/images/test.nd2', 'my_mask')
    p = 'C:/images/test_my_mask.tif'

    :param path: str path to an input file
    :param name: str addition to name of the output file
    :return: str path to output file
    """
    folder = os.path.dirname(path)
    # Check for relative path
    if folder == "":
        folder = "./"

    # adjust the file name
    file_name = os.path.basename(path).split(".")[:-1]
    file_name[-1] = file_name[-1] + "_" + name + ".tif"
    file_name = ".".join(file_name)

    return os.path.join(folder, file_name)


def read_image(path: str) -> CustomImage:
    """
    Checks for supported file types (for this library) and reads the image.

    tif and vsi files are read with bioio_bioformats.

    :param path: path to file
    :return: basically a BioImage object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    supported_types = ["tif", "tiff", "nd2", "vsi"]
    file_ext = path.split(".")[-1]
    if file_ext not in supported_types:
        raise NotImplementedError(
            f"File not supported: {path}. Supported types: {supported_types}"
        )
    if file_ext == "nd2":
        import bioio_nd2

        return CustomImage(path, reader=bioio_nd2.Reader)

    # Support for ome.tif
    if path.endswith("ome.tiff") or path.endswith("ome.tif"):
        import bioio_ome_tiff

        return CustomImage(path, reader=bioio_ome_tiff.Reader)

    if file_ext == "vsi" or file_ext == "tif" or file_ext == "tiff":
        import bioio_bioformats

        return CustomImage(path, reader=bioio_bioformats.Reader)


def read_labels(path: str) -> np.ndarray:
    """
    Read a labels image from .tif file, using the TIFFFILE library.

    :param path: path to file
    :return: plain np.ndarray
    """
    if not path.endswith(".tif") and not path.endswith(".tiff"):
        raise ValueError(f"File not a .tif: {path}")
    return tifffile.imread(path)
