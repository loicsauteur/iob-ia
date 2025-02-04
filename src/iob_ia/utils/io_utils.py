import os

import tifffile
from bioio import BioImage
import bioio_bioformats
import numpy as np
from tifffile import TiffFile, imwrite
from zipfile import ZIP_DEFLATED


def save_image_channel(img: np.ndarray, path: str):
    """
    Save an image as tif file.

    :param img: image
    :param path: save path
    :return:
    """
    imwrite(path, img)
    print(f'Saved image to: {path}')


def save_labels(labels: np.ndarray, path: str):
    """
    Save labels to .tif file

    :param labels:
    :param path:
    :return:
    """
    imwrite(path, labels, compression=ZIP_DEFLATED)
    print(f'Saved labels to: {path}')


def gen_out_path(path: str, name: str = 'output') -> str:
    """
    Generate output TIF-file path from input path and name.

    E.g. to create a save path with similar filename...

    :param path: str path to an input file
    :param name: str addition to name of the output file
    :return: str path to output file
    """
    folder = os.path.dirname(path)
    # Check for relative path
    if folder == '':
        folder = './'

    # adjust the file name
    file_name = os.path.basename(path).split('.')[:-1]
    file_name[-1] = file_name[-1] + '_' + name + '.tif'
    file_name = '.'.join(file_name)

    return os.path.join(folder, file_name)


def read_vsi(path: str) -> (np.ndarray, tuple):
    """
    Read image from .vsi file.

    Returns the image as CZYX
    :param path: str path to file
    :return: data-array, voxel_size
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    if not path.endswith('.vsi'):
        raise ValueError(f'File not a .vsi: {path}')

    b_img = BioImage(path, reader=bioio_bioformats.Reader)
    # remove axes of size 1
    img = np.squeeze(b_img.data)
    # get the voxel size
    voxel_size = (
        b_img.physical_pixel_sizes.Z,
        b_img.physical_pixel_sizes.Y,
        b_img.physical_pixel_sizes.X
    )

    return img, voxel_size


def read_tif(path: str) -> (np.ndarray, tuple):
    """
    Read image from .tif file.

    Only imageJ tif files supported.

    Returns the image as CZYX
    :param path:
    :return:
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    if not path.endswith('.tif'):
        raise ValueError(f'File not a .tif: {path}')
    t_file = TiffFile(path)
    data = t_file.asarray()
    if len(data.shape) < 3:
        raise RuntimeError(f'3D image is expected, but it is a 2D image: {path}')
    elif len(data.shape) > 4:
        raise RuntimeError(f'Too many dimensions: image has '
                           f'{len(data.shape)} dimensions: {path}')
    elif len(data.shape) == 4:
        # Multichannel image has shape: ZCYX need to convert to CZYX
        data = np.swapaxes(data, 0, 1)

    # Find voxel-size
    # To find the Z step it is a bit more difficult, need to use IJ metadata
    if not t_file.is_imagej:
        raise Warning(f'Only imagej tif files supported.')
    # Find the Z incrementValue for the z-step size
    ij_description = t_file.pages[0].tags['IJMetadata'].value
    info = ij_description['Info'].split('\n')
    z = ''
    for line in info:
        if line.startswith('Z incrementValue'):
            z = line.split('=')[1]
            z = z.strip()
            break
    if z == '':
        raise ValueError(f'Could not find Z step value in IJ metadata.')
    # Find the YX voxel size
    tags = t_file.pages[0].tags
    x = tags['XResolution'].value
    y = tags['YResolution'].value
    x = x[1] / x[0]
    y = y[1] / y[0]

    return data, (float(z), y, x)


def read_labels(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    if not path.endswith('.tif'):
        raise ValueError(f'File not a .tif: {path}')
    return tifffile.imread(path)
