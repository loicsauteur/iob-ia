import os
from zipfile import ZIP_DEFLATED

import numpy as np
import tifffile
from tifffile import imwrite

from iob_ia.custom_image import CustomImage


def save_image_channel(img: np.ndarray, path: str):
    """
    Save an image as tif file.

    :param img: image
    :param path: save path
    :return:
    """
    imwrite(path, img)
    print(f"Saved image to: {path}")


def save_labels(labels: np.ndarray, path: str):
    """
    Save labels to .tif file.

    :param labels:
    :param path:
    :return:
    """
    imwrite(path, labels, compression=ZIP_DEFLATED)
    print(f"Saved labels to: {path}")


def gen_out_path(path: str, name: str = "output") -> str:
    """
    Generate output TIF-file path from input path and name.

    E.g. to create a save path with similar filename...

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
    if file_ext == "vsi" or file_ext == "tif" or file_ext == "tiff":
        import bioio_bioformats

        return CustomImage(path, reader=bioio_bioformats.Reader)


def read_labels(path: str) -> np.ndarray:
    """
    Read a labels image from .tif file.

    :param path: path to file
    :return: plain np.ndarray
    """
    if not path.endswith(".tif") and not path.endswith(".tiff"):
        raise ValueError(f"File not a .tif: {path}")
    return tifffile.imread(path)
