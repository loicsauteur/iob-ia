"""A (hopefully) better collection of label filtering functions."""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.util import map_array

import iob_ia.utils.extra_props as ep
from iob_ia.utils.classify import estimate_threshold
from iob_ia.utils.segment import check_skimage_version


def measure_label_props(
    img_label: np.ndarray,
    img_intensity: Optional[np.ndarray] = None,
    voxel_size: Union[float, tuple] = (1, 1, 1),
    verbose: bool = False,
) -> dict:
    """
    Measure properties of labels in a label image.

    Used to create a measurements list for subsequent label filtering.
    All the current supported properties are calculated in calibrated units.

    :param img_label: 3D label image
    :param img_intensity: Optional 3D intensity channel
    :param voxel_size: tuple for calibrated measurements
    :param verbose: whether to print out the information
    :return: regionprops_table dictionary
    """
    # Define which properties to measure
    props_to_measure = [
        "label",
        "area",
        "euler_number",
        "extent",
    ]

    if img_intensity is not None:
        props_to_measure.append("intensity_mean")
        props_to_measure.append("intensity_min")
        props_to_measure.append("intensity_max")

    if check_skimage_version():
        props_to_measure.append("intensity_std")

    # Measure properties
    props_table = regionprops_table(
        label_image=img_label,
        intensity_image=img_intensity,
        properties=props_to_measure,
        extra_properties=(
            ep.projected_area,
            ep.projected_convex_area,
            ep.projected_perimeter,
            ep.projected_circularity,
        ),
        spacing=voxel_size,
    )
    # Calibrate the extra properties in the table
    props_table = ep.calibrate_extra_properties(
        props_table, voxel_size=voxel_size
    )

    # Inform about the properties:
    if verbose:
        print(f'Measured properties of {len(props_table["label"])} labels.')
        for k, v in props_table.items():
            if k != "label":
                print(f"{k}: mean = {np.mean(v)}; median = {np.median(v)}")

    return props_table


def filter_by_property(
    props_table: dict,
    prop: str,
    min_val: float = float("-inf"),
    max_val: float = float("inf"),
    labels_to_remove: Optional[list] = None,
    verbose: bool = False,
) -> tuple[dict, list[Any]]:
    """
    Filter a regionprops table based on a property min/max values.

    If min and max are not specified, the min value is estimated
    via an iterative approach.

    :param props_table: regionprops_table
    :param prop: String property to filter on
    :param min_val: Minimum value. Default = -inf
    :param max_val: Maximum value. Default = inf
    :param labels_to_remove: list of labels from a previous call. Default empty list.
    :param verbose: whether to print out the information
    :return: modified regionprops_table
    :return: list of labels to remove
    """
    if prop not in props_table.keys():
        raise KeyError(f'Property "{prop}" not found in props_table.')

    df = pd.DataFrame.from_dict(props_table)

    # Automatically determine if min_val, if min and max not specified
    if min_val == float("-inf") and max_val == float("inf"):
        min_val = estimate_threshold(prop, df)
        if verbose:
            print(f"Estimated minimal threshold value: {min_val}")
    if verbose:
        print(f"Filtering on {prop} between {min_val} and {max_val}")

    # Get a list of dataframe indices to remove
    indices_to_drop = []
    if labels_to_remove is None:
        labels_to_remove = []
    for row in df.iterrows():
        if row[1][prop] < min_val or row[1][prop] > max_val:
            indices_to_drop.append(row[0])
            labels_to_remove.append(row[1]["label"])

    # Remove the rows from the dataframe
    for i in indices_to_drop:
        df = df.drop(i)

    if verbose:
        print(
            f"Removed {len(indices_to_drop)} labels. "
            f'Remaining labels: {len(df["label"])} '
            f'(previously {len(props_table["label"])})'
        )
    return df.to_dict(), labels_to_remove


def filter_labels_by_property(
    img_label: np.ndarray,
    props_table: dict,
    props: dict[str, [float, float]],
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Filter labels from a label image.

    Create a new label image by filtering on label properties.
    Multiple properties to filter on can be passed as a dictionary of keys = properties,
    and values = [min, max]. E.g.:
    props_table = {
        'area': [100, 200],
        'intensity_mean': [1000.5, float("inf")],
        'projected_perimeter': [None, None],
        ...
    }

    Expected min/max values are float, but can be None.
    If None, they will be set to -inf and inf. If both are -inf/inf or None, the min
    value will be estimated via an iterative approach.

    :param img_label: label image
    :param props_table: regionprops_table (e.g. by using measure_label_props)
    :param props: dictionary of properties to filter on mapped with
                  their min/max thresholds
    :param verbose: whether to print out the information
    :return: label image filtered
    :return: modified regionprops_table (only remaining labels)
    """
    # Check that the properties to filter on are in the table
    for p, vals in props.items():
        if p not in props_table.keys():
            raise KeyError(
                f'Property "{p}" not found in props_table. '
                f"Available are properties: {props_table.keys()}"
            )
        if len(vals) != 2:
            raise ValueError(
                f'Expected 2 values for property "{p}" '
                f"but got {len(vals)} values."
            )

    labels_to_remove = []
    for k, v in props.items():
        if v[0] is None:
            v[0] = float("-inf")
        if v[1] is None:
            v[1] = float("inf")
        mod_props_table, labels_to_remove = filter_by_property(
            props_table=props_table,
            prop=k,
            min_val=v[0],
            max_val=v[1],
            labels_to_remove=labels_to_remove,
            verbose=verbose,
        )

    # Get a list of all labels
    input_vals = np.unique(img_label)
    # Set all labels to be removed to 0 (in output_vals)
    output_vals = np.copy(input_vals)
    for label in labels_to_remove:
        output_vals = np.where(output_vals == label, 0, output_vals)

    # Create filtered label image
    img_filtered = map_array(
        input_arr=img_label, input_vals=input_vals, output_vals=output_vals
    )
    return img_filtered, props_table
