import numpy as np
import skimage
from skimage.morphology import remove_small_objects

from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
from typing import List, Optional
from tqdm import tqdm

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

    # Set up the model
    # FIXME if not cyto3, i need to check if the path can be used...?!?!
    model = models.Cellpose(gpu=use_gpu, model_type=model_path)

    # grayscale image
    channels = [0, 0]

    # Run 3D cellpose
    masks, flows, _, _ = model.eval(
        img, channels=channels,
        diameter=12, do_3D=True,
        anisotropy=anisotropy,
        z_axis=0
    )
    return masks


def filter_shape(
    labels: np.ndarray, prop: str,
    min_val=float('-inf'), max_val=float('inf'),
    labels_to_remove: Optional[List] = None
) -> List:
    """
    Filter a label image by a shape property.

    Only 'area, 'euler_number' and 'extent' are supported for 3D images.
    :param labels: label image
    :param prop: str of property to filter on
    :param min_val: minimum value (exclusive). Default -inf
    :param max_val: maximum value (exclusive). Default inf
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
    return labels_to_remove


def filter_intensity(
    labels: np.ndarray, img: np.ndarray,
    prop: str, min_val=float('-inf'), max_val=float('inf'),
    labels_to_remove: Optional[List] = None
) -> List:
    """
    Filter a label image on intensity features.

    Only "intensity_max", "intensity_mean", "intensity_min" supported.
    For skimage version > 0.23.1, also "intensity_std" supported.
    :param labels: label image
    :param img: intensity channel image
    :param prop: property to filter on
    :param min_val: minimum value (exclusive). Default = -inf
    :param max_val: maximum value (exclusive). Default = inf
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
    return labels_to_remove


def check_skimage_version(
    major: int = 0,
    minor: int = 23,
    micro: int = 1
) -> bool:
    """
    Check if the installed skimage version is bigger than major.minor.micro
    Default minimal skimage version = 0.23.1
    :param major:
    :param minor:
    :param micro:
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
    labels: List
) -> np.ndarray:
    copy = np.copy(img_label)
    for lbl in tqdm(labels):
        # find indices that correspond to the label
        a = copy == lbl
        # set the image where indices True to 0
        copy[a] = 0
    return copy


def filter_labels_size(mask: np.ndarray, min_size: int, max_size: int):
    """
    @Deprecated

    :param mask:
    :param min_size:
    :param max_size:
    :return:
    """
    lbl = label(mask)
    props_ori = regionprops(lbl)
    table_ori = regionprops_table(lbl, properties=['label', 'area'])
    lbl = remove_small_objects(lbl, min_size=min_size)
    table_small = regionprops_table(lbl, properties=['label', 'area'])
    lbl = remove_big_objects(lbl, max_size=max_size)
    table_big = regionprops_table(lbl, properties=['label', 'area'])

    #print(table_ori['label'])
    #print(table_small['label'])
    #print(table_big['label'])
    #print(type(table_big))

    table_small['small'] = ['small'] * len(table_small['label'])
    table_big['big'] = ['big'] * len(table_big['label'])

    table_ori = pd.DataFrame.from_dict(table_ori)
    table_small = pd.DataFrame.from_dict(table_small)
    table_big = pd.DataFrame.from_dict(table_big)

    final = pd.merge(table_ori, table_small, on='label', how='left')
    final = pd.merge(final, table_big, on='label', how='left')
    print(final)
    return final


def filter_label_prop(mask: np.ndarray, prop: str, value: float):
    """
    @Deprecated

    :param mask:
    :param prop:
    :param value:
    :return:
    """
    # maybe i can use np.where to set label values to zero after checking regionprops?
    pass


def remove_big_objects(ar: np.ndarray, max_size: int = 64) -> np.ndarray:
    """
    @Deprecated
    :param ar:
    :param max_size:
    :return:
    """
    # Similar to skimage.morphology.remove_small_objects
    out = ar.copy()
    css = out
    try:
        component_sizes = np.bincount(css.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )
    # print(component_sizes)
    too_big = component_sizes > max_size
    too_big_mask = too_big[css]
    out[too_big_mask] = 0

    return out

