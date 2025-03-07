import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from typing import Optional, List, Dict
import iob_ia.utils.extra_props as ep
import iob_ia.utils.segment as seg


def create_base_table(
    img: np.ndarray,  # expected C(Z)YX
    nuclei: np.ndarray,
    cyto: np.ndarray = None,
    cells: np.ndarray = None,
    channel_names: Optional[List] = None,
    scale: Optional[tuple] = (1, 1, 1),
) -> pd.DataFrame:
    """
    Create a pandas Dataframe of the skimage regionprops_table.

    :param img: multichannel image (C(Z)YX)
    :param nuclei: mask of nuclei
    :param cyto: mask of cytoplasm (label IDs should correspond to the nuclei mask)
    :param cells: mask of cells (label IDs should correspond to the nuclei mask)
    :param channel_names: list of channel names
    :param scale: voxel size for shape measurements in calibrated units
    :return:
    """
    # For multichannel images, the channels should be the last axis
    if img.ndim != nuclei.ndim:
        # Ensure that the channels are the first axis
        if img.shape[1:] != nuclei.shape:
            raise ValueError(
                f'Image (shape={img.shape}) and labels (shape={nuclei.shape}) '
                f'must have the same shape.'
            )
        # For region props the channels should be the last axis
        img = np.moveaxis(img, 0, -1)
        n_channels = img.shape[-1]
    else:
        n_channels = 1
    # Check the channel names
    if channel_names is None:
        channel_names = [f'ch_{i}' for i in range(1, n_channels + 1)]
    if len(channel_names) != n_channels:
        raise ValueError(
            f'The channel names {channel_names} do not match '
            f'the number of image channels {n_channels}.'
        )
    # Get the region props table for the nuclei  --------------------------------------
    properties = [
        'label', 'area', 'centroid', 'bbox_area',
        'intensity_mean', 'intensity_min', 'intensity_max',
    ]
    if img.ndim == 4:
        extra_properties = [
            ep.projected_area,
            ep.projected_convex_area,
            ep.projected_circularity,
            ep.projected_perimeter,
        ]
    else:
        extra_properties = None
    table = regionprops_table(
        nuclei,
        intensity_image=img,
        properties=properties,
        extra_properties=extra_properties,
        spacing=scale,
    )
    table = pd.DataFrame.from_dict(table)
    # Convert projected properties to real units
    table = ep.calibrate_extra_properties(table, voxel_size=scale)
    # Re-order columns
    new_order = reorder_columns(table.columns)
    table = table.reindex(columns=new_order)
    # Rename the table headers (i.e. add 'Nucleus:')
    table = adjust_table_headers(
        table=table,
        mask_name='Nucleus',
        channel_names=channel_names
    )

    # Get the region props for cytoplasm and cell masks  ------------------------------
    properties = [
        'label', 'area',
        'intensity_mean', 'intensity_min', 'intensity_max',
    ]
    if cyto is not None:
        cyto_table = regionprops_table(
            cyto,
            intensity_image=img,
            properties=properties,
            spacing=scale,
        )
        cyto_table = adjust_table_headers(
            table=pd.DataFrame.from_dict(cyto_table),
            mask_name='Cytoplasm',
            channel_names=channel_names
        )
        table = pd.merge(table, cyto_table, on='label')

    if cells is not None:
        cell_table = regionprops_table(
            cells,
            intensity_image=img,
            properties=properties,
            spacing=scale,
        )
        cell_table = adjust_table_headers(
            table=pd.DataFrame.from_dict(cell_table),
            mask_name='Cell',
            channel_names=channel_names
        )
        table = pd.merge(table, cell_table, on='label')

    return table


def classify(
    table_in: pd.DataFrame,
    prop: str,
    classification: str = None,
    min_val: float = float('-inf'),
    max_val: float = float('inf'),
) -> pd.DataFrame:
    """
    Classify list entries according to a property value.

    Will create a new column 'Classification' and add classes separated by ';'

    :param table_in: pd.DataFrame
    :param prop: str property value to classify on
    :param classification: str name for the classification (class)
    :param min_val: minimum value (inclusive). Default -inf
    :param max_val: maximum value (inclusive). Default inf
    :return: pd.DataFrame
    """
    table = table_in.copy()
    # Check that the property for classification exists
    if prop not in table.columns:
        # List available measurements
        print('Available measurements:')
        for measurement in table.columns:
            print('-', measurement)
        raise ValueError(f'Property "{prop}" not found in table')
    # Rename the classification if it is None
    if classification is None:
        classification = prop + '+'
    # Ensure a classification column exists in the table and add the classification
    if 'Classification' not in table.columns:
        table['Classification'] = ''
        table = table.assign(Classification=[
            classification + ';' if min_val <= x <= max_val else '' for x in table[prop]
        ])
    else:
        table['TEMP'] = ''
        table = table.assign(TEMP=[
            classification + ';' if min_val <= x <= max_val else '' for x in table[prop]
        ])
        # merge TEMP column to classification column
        table['Classification'] = table['Classification'] + table['TEMP']
        # remove TEMP column
        table = table.drop(columns=['TEMP'])

    # Log the result
    print(table.value_counts('Classification'))
    return table


def count_classifications(table: pd.DataFrame) -> pd.Series:
    """
    Count the number of classes in the table.

    :param table: pd.DataFrame
    :return: pd.DataFrame
    """
    return table.value_counts("Classification")


def adjust_table_headers(
    table: pd.DataFrame,
    mask_name: str,
    channel_names: List
) -> pd.DataFrame:
    """
    Adjust the table headers to include the mask name (e.g. Nucleus) and channel names.

    :param table: DataFrame to be changed
    :param mask_name: mask identifier, e.g. Nucleus
    :param channel_names: List of channel_names
    :return: pd.DataFrame
    """
    header_map = {}
    headers = table.columns
    for h in headers:
        if 'area' in h:
            header_map[h] = f'{mask_name}: {h}'
        elif 'projected_' in h:
            header_map[h] = f'{mask_name}: {h}'
        elif 'intensity' in h:
            header_map[h] = f'{mask_name}: {h[:-1]}{channel_names[int(h[-1])]}'
        else:
            header_map[h] = h
    # Rename the columns
    table = table.rename(columns=header_map, errors='raise')
    return table


def reorder_columns(columns: List) -> List:
    """
    Reorder the columns in the table, to have intensities at the end.

    :param columns: list of column names
    :return: list of sorted column names
    """
    new_order = []
    for c in columns:
        if 'intensity' in c:
            break
        if 'projected_' in c:
            break
        new_order.append(c)
    for c in columns:
        if 'projected_' in c:
            new_order.append(c)
    for c in columns:
        if 'intensity' in c:
            new_order.append(c)
    if len(new_order) != len(columns):
        raise RuntimeError(
            f'Failed to reorder the columns.\n'
            f'Input columns: {columns}\n'
            f'Output columns: {new_order}'
        )
    return new_order


def show_by_class(
    table: pd.DataFrame,
    class_names: str,
    img_nuclei: np.ndarray,
    color: Optional[str] = None,
    voxel_size: Optional[tuple] = (1, 1, 1)
) -> np.ndarray:
    """
    Create a new label mask of nuclei only of a specific class

    :param table: pd.DataFrame with all measurements
    :param class_names: Classification column class name to filter on,
                       multiple classes should be separated by ';'
    :param img_nuclei: nuclei label image
    :param color: Default None. If not None, will show the created mask in napari with
                  all objects colored in the chosen color
    :param voxel_size: voxel size (only for visualization)
    :return: nuclei_label image of nuclei of specified class
    """
    if "Classification" not in table.columns:
        raise ValueError(f'No classification column found in table')

    all_labels = table['label']
    label_dict = {}
    class_names = class_names.split(';')

    # For each classification
    for class_name in class_names:
        # For each row in the table
        for i in range(len(table['Classification'])):
            # If the class name is in the classification
            if class_name in table['Classification'][i]:
                # If the label_ID is present, don't modify it
                # Only add new entries
                if all_labels[i] not in label_dict:
                    # Assign the label_dict (key and value) = label_ID
                    label_dict[all_labels[i]] = all_labels[i]
            else:
                # Not positive: key = label_ID, value = 0, overwrites if it was 1 before
                label_dict[all_labels[i]] = 0

    # Create label image of selected labels
    class_labels = seg.remove_label_objects(
        img_label=img_nuclei, label_map=label_dict, relabel=False
    )

    if color is not None:
        import iob_ia.utils.visualise as vis
        colormap = vis.single_colormap(color=color, n_labels=max(label_dict.keys()))
        vis.add_labels(
            class_labels, name=';'.join(class_names),
            scale=voxel_size, colormap=colormap
        )
    return class_labels
