from time import time
from typing import Optional, Union

import numpy as np
import skimage
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels, relabel_sequential

import iob_ia.utils.extra_props as ep


def create_cell_cyto_masks(
    lbl: np.ndarray,
    expansion: float,
    shrinking: float = 0,
    voxel_size: Union[float, tuple] = 1,
) -> (np.ndarray, np.ndarray):
    """
    Create cell and cyto masks from the labels.

    Allows expansion for anisotropic data.
    :param lbl: nuclear label mask
    :param expansion: desired expansion in voxel_size units
    :param shrinking: desired shrinking in voxel_size units
    :param voxel_size: ZYX voxel size
    :return: nuclei mask, cell mask, cytoplasm mask
    """
    if len(voxel_size) != 3:
        raise RuntimeError(
            f"Voxel size must be 3D. You have {len(voxel_size)} dimensions. "
        )
    if voxel_size[1] != voxel_size[2]:
        raise ValueError(
            f"Voxel size in Y and X must be equal. Got: {voxel_size[-2:]}"
        )
    # skimage has anisotropic expand labels from v0.23.0 on
    # (also requires scipy>=1.8, but I don't think this will be a problem)
    if shrinking > 0:
        start = time()
        lbl = label_shrinking(lbl, spacing=voxel_size, shrink=shrinking)
        print("Shrinking nuclei took:", time() - start)
    start = time()
    cells = label_expansion(lbl, spacing=voxel_size, expansion=expansion)
    print("Creating cells took:", time() - start)
    start = time()
    # create cyto mask
    cyto = np.subtract(cells, lbl)
    print("Creating cytoplasm took:", time() - start)
    return lbl, cells, cyto


def label_expansion(
    label_image: np.ndarray,
    spacing: Union[float, tuple] = 1,
    expansion: float = 1,
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
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out


def label_shrinking(
    label_image: np.ndarray,
    spacing: Union[float, tuple] = 1,
    shrink: float = 1,
) -> np.ndarray:
    """
    To shrink labels in ZYX, by 'shrink' units.

    Uses scipy.ndimage.distance_transform_edt to calculate distances (supports
    anisotropic spacing).

    :param label_image: e.g. nuclei
    :param spacing: voxel/pixel size
    :param shrink: distance to shrink in spacing units
    :return: shrunk labels of label_image
    """
    from scipy.ndimage import distance_transform_edt

    distances = distance_transform_edt(label_image != 0, sampling=spacing)
    erode_mask = distances > shrink
    return np.where(erode_mask, label_image, 0)  # this is fast...


def measure_props(
    img_label: np.ndarray,
    img_intensity: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    voxel_size: Union[float, tuple] = (1, 1, 1),
) -> dict:
    """
    Measure the properties of the labels.

    Allow props including optional (multi-channel) intensity image.

    :param img_label: 3D label image
    :param img_intensity:  a list of 3D channels or a single 3D CZYX image
    :param voxel_size: ZYX voxel size for calibrated shape measurements
    :return: regionprops_table dictionary
    """
    # Make sure that the image is a 3D label image
    if len(img_label.shape) != 3:
        raise ValueError(
            f"Image must be 3D. You have {img_label.ndim} dimensions. "
            f"With an image shape of: {img_label.shape}."
        )
    # Define the properties to measure
    if img_intensity is None:
        props = [
            "label",
            "area",
            "euler_number",
            "extent",
        ]
    else:
        props = [
            "label",
            "area",
            "euler_number",
            "extent",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
        ]
        # Check the intensity image
        if isinstance(img_intensity, list):
            # Check that each image has the same shape as the label image
            for img in img_intensity:
                if img.shape != img_label.shape:
                    raise ValueError(
                        f"The intensity images must have the same shape. "
                        f"Got {img.shape}, but should be {img_label.shape}."
                    )
            # Convert list to multichannel ZYXC image
            img_intensity = np.stack(img_intensity, axis=-1)
        elif isinstance(img_intensity, np.ndarray):
            # Make sure that the image has the same shape as the label image
            if img_intensity.shape != img_label.shape:
                # Check if it is a CZYX image
                if (
                    len(img_intensity.shape) == 4
                    and img_intensity.shape[1:] == img_label.shape
                ):
                    # Swap axes to ZYXC for region props
                    img_intensity = np.moveaxis(img_intensity, 0, -1)
                else:
                    raise ValueError(
                        f"Intensity image must be a CZYX image. "
                        f"You have {img_intensity.shape}."
                    )
        else:
            raise ValueError(
                f"Intensity image must be a list or single numpy array. "
                f"You have {type(img_intensity)}."
            )
    # Measure properties
    table = regionprops_table(
        label_image=img_label,
        intensity_image=img_intensity,
        properties=props,
        extra_properties=(
            ep.projected_area,
            ep.projected_convex_area,
            ep.projected_perimeter,
            ep.projected_circularity,
        ),
        spacing=voxel_size,
    )
    # Calibrate the extra_properties
    table = ep.calibrate_extra_properties(table, voxel_size=voxel_size)
    return table


def segment_3d_cellpose(
    img: np.ndarray,
    model_path: str = "cpsam",
    anisotropy: Optional[float] = None,
    flow3D_smooth: int = 0,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
    """
    Use cellpose cpsam to segment the 3D image.

    :param img: single channel image
    :param model_path: path to cellpose model. Default = 'cpsam'
    :param anisotropy: Optional, anisotropy rescaling factor
    :param flow3D_smooth: gaussian sigma for smoothing 3D flows. Default = 0
                    Note: This only helps a bit, fusing smaller pieces...
    :param cellprob_threshold:
    :return: (np.ndarray for mask, list for flows)
    """
    # check the image first
    if img.ndim != 3:
        raise ValueError(
            f"Image must be a dingle channel in 3D. "
            f"You have {img.ndim} dimensions. "
            f"With an image shape of: {img.shape}."
        )

    # Local imports
    from cellpose import core, models
    from cellpose.io import logger_setup

    use_gpu = core.use_gpu()
    logger_setup()
    print(">>> GPU activated:", use_gpu)

    # grayscale image
    channels = [0, 0]

    # Set up the model
    if model_path == "cpsam":
        model = models.Cellpose(gpu=use_gpu, model_type=model_path)
        # Run 3D cellpose
        result = model.eval(
            img,
            channels=channels,  # FIXME this is deprecated
            diameter=12,
            do_3D=True,
            anisotropy=anisotropy,
            # newer version calls it flow3D_smooth not dP_smooth
            flow3D_smooth=flow3D_smooth,
            cellprob_threshold=cellprob_threshold,
            # z_axis=0,
        )
    else:
        model = models.CellposeModel(gpu=True, pretrained_model=model_path)
        # TODO: Not sure how important the anisotropy is here -->
        #  probably because z-stage movement not precise...(cels in z are 40um)
        #  Also need to try with smoothing the flows:
        #  -> dP_smooth/flow3D_smooth (sigma for gaussian filter) -->
        #  Doesnt seem too have a big impact
        #  cellprob_threshold maybe decrease to -3
        #  (range -6 to + 6, default 0, smaller=more cells)
        # Run 3D cellpose
        result = model.eval(
            img,
            channels=channels,  # FIXME this is deprecated
            diameter=12,
            do_3D=True,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
            flow3D_smooth=flow3D_smooth,
            # z_axis=0,
        )
    # eval returns multiple objects (number of objects may change)
    return result[0]  # (first item is the mask)


def filter_shape(
    labels: np.ndarray,
    prop: str,
    min_val=float("-inf"),
    max_val=float("inf"),
    return_all_labels: bool = False,
    labels_to_remove: Optional[list] = None,
    props_table: Optional[dict] = None,
    voxel_size: Union[float, tuple] = (1, 1, 1),
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
    :param props_table: Optional, table already measured of properties
    :param voxel_size: voxel size in um
    :return: list of labels to remove.
             Will append new labels to remove if labels_to_remove is not None
    """
    from iob_ia.utils.extra_props import __all_extra_props__

    supported_props = [
        "area",
        "euler_number",
        "extent",
    ]
    for p in __all_extra_props__:
        supported_props.append(p)

    if prop not in supported_props:
        raise NotImplementedError(
            f'Unsupported property "{prop}". '
            f"Supported properties are: {supported_props}"
        )
    if props_table is None or prop not in props_table:
        if labels.ndim != len(voxel_size):
            raise ValueError(
                f"Expected a 3D label image but got {labels.ndim}D. "
                f"If not a 3D label image, try providing the voxel_size."
            )
        props_table = regionprops_table(
            labels,
            properties=["label", prop],
            extra_properties=(
                ep.projected_area,
                ep.projected_convex_area,
                ep.projected_perimeter,
                ep.projected_circularity,
            ),
            spacing=voxel_size,
        )
        # Calibrate extra props
        props_table = ep.calibrate_extra_properties(
            props_table, voxel_size=voxel_size
        )
    labels_to_remove = get_label_list(
        labels=props_table["label"],
        values=props_table[prop],
        min_val=min_val,
        max_val=max_val,
        labels_to_remove=labels_to_remove,
    )
    if return_all_labels:
        return labels_to_remove, props_table["label"]
    return labels_to_remove


def filter_intensity(
    labels: np.ndarray,
    img: np.ndarray,
    prop: str,
    min_val=float("-inf"),
    max_val=float("inf"),
    return_all_labels: bool = False,
    labels_to_remove: Optional[list] = None,
    props_table: Optional[dict] = None,
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
    :param props_table: Optional, table already measured of properties
    :return: List of labels to remove
    """
    supported_props = [
        "intensity_max",
        "intensity_mean",
        "intensity_min",
    ]
    if check_skimage_version():
        supported_props.append("intensity_std")
    if prop not in supported_props:
        raise NotImplementedError(
            f'Unsupported property "{prop}". '
            f"Supported properties are: {supported_props}"
        )
    if props_table is None or prop not in props_table:
        props_table = regionprops_table(
            label_image=labels, intensity_image=img, properties=["label", prop]
        )
    labels_to_remove = get_label_list(
        labels=props_table["label"],
        values=props_table[prop],
        min_val=min_val,
        max_val=max_val,
        labels_to_remove=labels_to_remove,
    )
    if return_all_labels:
        return labels_to_remove, props_table["label"]
    return labels_to_remove


def check_skimage_version(
    major: int = 0, minor: int = 23, micro: int = 1
) -> bool:
    """
    Check if the installed skimage version is bigger than major.minor.micro.

    Default minimal skimage version = 0.23.1
    :param major: minimal skimage major. Default = 0
    :param minor: minimal skimage minor. Default = 23
    :param micro: minimal skimage micro. Default = 1
    :return: boolean
    """
    v = skimage.__version__.split(".")
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
                print("Error:", e)
                return False
            return v3 > micro


def get_label_list(
    labels: list,
    values: list,
    min_val,
    max_val,
    labels_to_remove: Optional[list] = None,
) -> list:
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
            # check if the label is not already in the list
            if labels[i] not in labels_to_remove:
                labels_to_remove.append(labels[i])
        if values[i] > max_val:
            # check if the label is not already in the list
            if labels[i] not in labels_to_remove:
                labels_to_remove.append(labels[i])
    return labels_to_remove


def remove_label_objects(
    img_label: np.ndarray,
    label_map: Optional[dict] = None,
    labels: Optional[list] = None,
    all_labels: Optional[list] = None,
    relabel: bool = False,
) -> np.ndarray:
    """
    Remove labels from an image.

    Either use the 'label_map', or a list of 'labels' to remove.
    Basically uses the:
    napari_filter_labels_by_prop.utils.remove_labels function

    :param img_label: label image
    :param labels: list of labels to remove
    :param label_map: dictionary of labels to remove,
               key = label ID, value = label ID (keep) or 0 (remove)
    :param all_labels: list of all labels in the image
    :param relabel: whether to relabel the image or keep the
                    original label ids. Default = False
    :return: label image with labels removed
    """
    # Check if label_map or labels are provided
    if label_map is None and labels is None:
        raise ValueError("Either label_map or labels must be provided.")
    # Check that not both, label_map and labels are provided
    if label_map is not None and labels is not None:
        raise ValueError(
            "Please provide only one, either label_map or labels, not both."
        )
    if label_map is not None:
        from napari_filter_labels_by_prop.utils import remove_labels

        return remove_labels(
            img=img_label, label_map=label_map, relabel=relabel
        )
    # Convert the lists to numpy arrays
    if all_labels is None:
        all_labels = regionprops_table(img_label, properties=["label"])[
            "label"
        ]
    # List of all labels
    input_vals = np.array(all_labels, dtype=int)

    # output_vals is the same len(input_vals) and maps the values of input_vals
    # i.e. set labels (labels to remove) in output_vals to 0
    output_vals = np.copy(input_vals)
    for i in labels:
        output_vals = np.where(output_vals == i, 0, output_vals)

    copy = skimage.util.map_array(
        img_label, input_vals=input_vals, output_vals=output_vals
    )
    # Re-label the new label image
    if relabel:
        copy, _, _ = relabel_sequential(copy)
    return copy


@DeprecationWarning
def calc_pixel_size(size_um: float, voxel_size: tuple) -> float:
    """
    Calculate a calibrated volume to number of pixels.

    # DEPRECATED / Unused
    E.g. for getting a value for filtering on size.

    :param size_um: desired size in um^3
    :param voxel_size: ZYX voxel size
    :return: volume in number of pixels
    """
    if len(voxel_size) != 3:
        raise ValueError(
            f"Expected 3D voxel size," f"but got {len(voxel_size)} dimensions."
        )
    return size_um / (voxel_size[0] * voxel_size[1] * voxel_size[2])
