"""
Measure.py is intended to do some basic measurements.

- basically Fiji's binary feature extractor
- basic intensity measurements
"""

import numpy as np
from skimage.measure import regionprops_table
from skimage.util import map_array


def overlapping_labels(
    img_1: np.ndarray, img_2: np.ndarray, min_overlap: float = 0.5
) -> np.ndarray:
    """
    Return a mask of labels from img_1 that have >= min_overlap with img_2.

    Similar to biovoxxel's binary feature extractor.

    :param img_1: Object label mask
    :param img_2: Selector label mask
    :param min_overlap:
    :return: np.ndarray of remaining labels (keeps the label identities)
    """
    # Sanity check: img_1 and img_2 have the same shape
    if img_1.shape != img_2.shape:
        raise ValueError("img_1 and img_2 must have the same shape.")

    # Binarize img_2
    img_2 = img_2 > 0

    # Measure overlap by "intensity_mean" (ranges between 0-1)
    table = regionprops_table(
        label_image=img_1,
        intensity_image=img_2,
        properties=["label", "intensity_mean"],
    )

    # Copy list of labels to select for min overlap
    pos_labels = np.copy(table["label"])

    # Check which labels have at least min_overlap "intensity"
    for i in range(len(table["label"])):
        # if bigger than min_overlap, keep label ID
        if table["intensity_mean"][i] >= min_overlap:
            pos_labels[i] = table["label"][i]
        else:
            pos_labels[i] = 0

    return map_array(
        input_arr=img_1, input_vals=table["label"], output_vals=pos_labels
    )


def intensity_and_presence(
    img_labels: np.ndarray,
    img_intensity: np.ndarray,
    img_selected_labels: np.ndarray,
    column_header: str = "positive",
) -> dict:
    """
    Measure label mean intensity and if the object is also present in other label image.

    ## FIXME: I don't remember why I did this function...
        probably to check:
        - RFP objects that have also overlap with GPF objects
        - Measure the GFP of all objects and see if they were selected with
          the "binary feature extractor"

    :param img_labels: "All" labels image
    :param img_intensity: intensity image
    :param img_selected_labels: label image of selected objects
           (i.e. from overlapping_labels function)
    :param column_header: string for the output column for positivity
    :return: dictionary for a table
    """
    if img_labels.shape != img_intensity.shape:
        raise ValueError(
            "img_labels and img_intensity must have the same shape."
        )
    if img_intensity.shape != img_selected_labels.shape:
        raise ValueError(
            "img_intensity and img_selected_labels must have the same shape."
        )
    # measure intensities
    table = regionprops_table(
        label_image=img_labels,
        intensity_image=img_intensity,
        properties=["label", "intensity_mean"],
    )
    table[column_header] = []

    # check if the labels are in the selected label image
    for label in table["label"]:
        if label in img_selected_labels:
            table[column_header].append(1)
        else:
            table[column_header].append(0)
    return table
