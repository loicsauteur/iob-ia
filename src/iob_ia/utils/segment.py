import numpy as np
import skimage
from skimage.segmentation import expand_labels
from skimage.measure import label

from skimage.measure import regionprops_table
from typing import List, Optional, Union
from tqdm import tqdm
from time import time


def create_cell_cyto_masks(
    lbl: np.ndarray,
    expansion: float,
    voxel_size: Union[float, tuple] = 1
) -> (np.ndarray, np.ndarray):
    """
    Create cell and cyto masks from the labels.

    Allows expansion for anisotropic data.
    :param lbl: nuclear label mask
    :param expansion: desired expansion in microns
    :param voxel_size: ZYX voxel size
    :return: cell mask, cytoplasm mask
    """
    if len(voxel_size) != 3:
        raise RuntimeError(
            f'Voxel size must be 3D. You have {len(voxel_size)} dimensions. '
        )
    if voxel_size[1] != voxel_size[2]:
        raise ValueError(
            f'Voxel size in Y and X must be equal. Got: {voxel_size[-2:]}'
        )
    # skimage has anisotropic expand labels from v0.23.0 on
    # (also requires scipy>=1.8, but I don't think this will be a problem)
    start = time()
    cells = cell_expansion(lbl, spacing=voxel_size, expansion=expansion)
    print('Creating cells took:', time() - start)
    start = time()
    # create cyto mask
    cyto = np.subtract(cells, lbl)
    print('Creating cytoplasm took:', time() - start)
    return cells, cyto


def cell_expansion(
    label_image: np.ndarray, spacing: Union[float, tuple] = 1, expansion: float = 1
) -> np.ndarray:
    """
    Basically skimage's expand_labels.

    But since anisotropic expansion is only available since skimage v0.23.0,
    re-implement it here: copied from:
    https://github.com/scikit-image/scikit-image/blob/v0.25.1/skimage/segmentation/_expand_labels.py
    :param label_image:
    :param spacing: usually a tuple of the voxel-size,
                    used to calculate the distance map with anisotropy
    :param expansion: distance in microns (if the spacing tuple is in microns)
    :return:
    """
    if check_skimage_version(0, 22, 9):
        return expand_labels(label_image, distance=expansion, spacing=spacing)
    # Re-implementation
    from scipy.ndimage import distance_transform_edt
    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, sampling=spacing, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= expansion
    # build the coordinates to find the nearest labels,
    # in contrast to 'cellprofiler' this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask] for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out


def segment_3d_cellpose(
    img: np.ndarray,
    model_path: str = 'cyto3',
    anisotropy: float = None
) -> np.ndarray:
    """
    Use cellpose cyto3 to segment the 3D image.


    :param img: single channel image
    :param model_path: path to cellpose model. Default = 'cyto3'
    :param anisotropy: Optional, anisotropy rescaling factor
    :return: (np.ndarray for mask, list for flows)
    """
    # check the image first
    if img.ndim != 3:
        raise ValueError(f'Image must be 3D. You have {img.ndim} dimensions. '
                         f'With an image shape of: {img.shape}.')

    # Local imports
    from cellpose import models, core
    from cellpose.io import logger_setup

    use_gpu = core.use_gpu()
    logger_setup()
    print('>>> GPU activated:', use_gpu)

    # grayscale image
    channels = [0, 0]

    # Set up the model
    # FIXME if not cyto3, i need to check if the path can be used...?!?!
    if model_path == 'cyto3':
        model = models.Cellpose(gpu=use_gpu, model_type=model_path)
        # Run 3D cellpose
        masks, flows, _, _ = model.eval(
            img, channels=channels,
            diameter=12, do_3D=True,
            anisotropy=anisotropy,
            z_axis=0
        )
    else:
        model = models.CellposeModel(gpu=True, pretrained_model=model_path)
        # TODO: Not sure how important the anisotropy is here
        #  Also need to try with smoothing the flows:
        #  -> dP_smooth (sigma for gaussian filter
        # Run 3D cellpose
        masks, flows, _ = model.eval(
            img, channels=channels,
            diameter=12, do_3D=True,
            anisotropy=anisotropy,
            z_axis=0
        )

    return masks


def filter_shape(
    labels: np.ndarray, prop: str,
    min_val=float('-inf'), max_val=float('inf'),
    return_all_labels: bool = False,
    labels_to_remove: Optional[List] = None
):
    """
    Filter a label image by a shape property.
    Generally, min/max value are supposed to be in pixels.

    Only 'area, 'euler_number' and 'extent' are supported for 3D images.
    :param labels: label image
    :param prop: str of property to filter on
    :param min_val: minimum value (exclusive). Default -inf
    :param max_val: maximum value (exclusive). Default inf
    :param return_all_labels: whether to return all labels. Default False .
    :param labels_to_remove: Optional, list of labels to remove
    :return: list of labels to remove.
             Will append new labels to remove if labels_to_remove is not None
    """
    supported_props = [
        "area",
        "euler_number",
        "extent",
    ]
    if prop not in supported_props:
        raise NotImplementedError(
            f'Unsupported property "{prop}". '
            f'Supported properties are: {supported_props}'
        )
    table = regionprops_table(labels, properties=['label', prop])
    labels_to_remove = get_label_list(
        labels=table['label'], values=table[prop],
        min_val=min_val, max_val=max_val,
        labels_to_remove=labels_to_remove
    )
    if return_all_labels:
        return labels_to_remove, table['label']
    return labels_to_remove


def filter_intensity(
    labels: np.ndarray, img: np.ndarray,
    prop: str, min_val=float('-inf'), max_val=float('inf'),
    return_all_labels: bool = False,
    labels_to_remove: Optional[List] = None
):
    """
    Filter a label image on intensity features.

    Only "intensity_max", "intensity_mean", "intensity_min" supported.
    For skimage version > 0.23.1, also "intensity_std" supported.
    :param labels: label image
    :param img: intensity channel image
    :param prop: property to filter on
    :param min_val: minimum value (exclusive). Default = -inf
    :param max_val: maximum value (exclusive). Default = inf
    :param return_all_labels: whether to return all labels. Default False.
    :param labels_to_remove: Optional, list of labels to remove
    :return: List of labels to remove
    """
    supported_props = [
        "intensity_max",
        "intensity_mean",
        "intensity_min",
    ]
    if check_skimage_version():
        supported_props.append('intensity_std')
    if prop not in supported_props:
        raise NotImplementedError(
            f'Unsupported property "{prop}". '
            f'Supported properties are: {supported_props}'
        )
    table = regionprops_table(
        label_image=labels,
        intensity_image=img,
        properties=['label', prop]
    )
    labels_to_remove = get_label_list(
        labels=table['label'], values=table[prop],
        min_val=min_val, max_val=max_val,
        labels_to_remove=labels_to_remove
    )
    if return_all_labels:
        return labels_to_remove, table['label']
    return labels_to_remove


def check_skimage_version(
    major: int = 0,
    minor: int = 23,
    micro: int = 1
) -> bool:
    """
    Check if the installed skimage version is bigger than major.minor.micro.

    Default minimal skimage version = 0.23.1
    :param major: minimal skimage major. Default = 0
    :param minor: minimal skimage minor. Default = 23
    :param micro: minimal skimage micro. Default = 1
    :return: boolean
    """
    v = skimage.__version__.split('.')
    if int(v[0]) > major:
        return True
    elif int(v[0]) < major:
        return False
    else:
        if int(v[1]) > minor:
            return True
        elif int(v[1]) < minor:
            return False
        else:
            try:
                v3 = int(v[2])
            except ValueError as e:
                return False
            return v3 > micro


def get_label_list(
    labels: List,
    values: List,
    min_val,
    max_val,
    labels_to_remove: Optional[List] = None
) -> List:
    """
    Get a list of labels to remove based on min and max values.

    :param labels: List of labels (same order as values, use regionprops_table)
    :param values: List of values
    :param min_val: minimum value (exclusive)
    :param max_val: maximum value (exclusive)
    :param labels_to_remove: Optional, list of labels to remove
    :return: list of labels to remove.
             Will append new labels to remove if labels_to_remove is not None
    """
    if labels_to_remove is None:
        labels_to_remove = []
    for i in range(len(values)):
        if values[i] < min_val:
            # check if the label is not already in the list from a previous call
            if labels[i] not in labels_to_remove:
                labels_to_remove.append(labels[i])
        if values[i] > max_val:
            # check if the label is not already in the list from a previous call
            if labels[i] not in labels_to_remove:
                labels_to_remove.append(labels[i])
    return labels_to_remove


def remove_label_objects(
    img_label: np.ndarray,
    labels: List,
    all_labels: Optional[List] = None,
) -> np.ndarray:
    """
    Remove labels from an image.

    :param img_label: label image
    :param labels: list of labels to remove
    :param all_labels: list of all labels in the image
    :return: label image with labels removed
    """
    # Convert the lists to numpy arrays
    if all_labels is None:
        all_labels = regionprops_table(img_label, properties=['label'])['label']
    # List of all labels
    input_vals = np.array(all_labels, dtype=int)

    # output_vals is the same len(input_vals) and maps the values of input_vals
    # i.e. set labels (labels to remove) in output_vals to 0
    output_vals = np.copy(input_vals)
    for i in labels:
        output_vals = np.where(output_vals == i, 0, output_vals)

    copy = skimage.util.map_array(
        img_label,
        input_vals=input_vals,
        output_vals=output_vals
    )
    # Re-label the new label image
    return label(copy)


def calc_pixel_size(size_um: float, voxel_size: tuple) -> float:
    """
    Calculate a calibrated volume to number of pixels.

    E.g. for getting a value for filtering on size.

    :param size_um: desired size in um^3
    :param voxel_size: ZYX voxel size
    :return: volume in number of pixels
    """
    if len(voxel_size) != 3:
        raise ValueError(f'Expected 3D voxel size,'
                         f'but got {len(voxel_size)} dimensions.')
    return size_um / (voxel_size[0] * voxel_size[1] * voxel_size[2])
