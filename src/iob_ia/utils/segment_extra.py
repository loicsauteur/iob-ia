"""Functions for image specific segmentation tasks."""

from time import time

import dask.array as da
import numpy as np
from skimage.filters import threshold_li
from skimage.measure import regionprops_table
from skimage.morphology import convex_hull_image, label
from skimage.restoration import rolling_ball
from skimage.transform import resize
from tqdm import tqdm


def scaffold_segmentation(
    img: np.ndarray, voxel_size: tuple, downscale: int = 5
) -> (np.ndarray, np.ndarray, tuple):
    """
    Rough scaffold segmentation.

    Only for 3D images. Will:
    - SUM project the channels
    - Downscale the image by the factor 'downscale'
    - Do a rolling ball background subtraction (with radius 100px / downscale)
    - Calculate an average Li threshold of the slices
    - Threshold the image and keep only the biggest object
        - This may cause some problems...
    - Make slice-wise convex hull of found objects
    :param img: input multichannel image
    :param voxel_size: voxel size (tuple) of image
    :param downscale: downscale factor
    :return:
           - downscaled and SUM projected channel image
           - segmented image
           - voxel size of the downscaled and segmented images
    """
    if len(voxel_size) != 3:
        raise NotImplementedError("Only 3D images are supported.")
    # Sum projection of the channels
    if len(img.shape) == 4:
        img = np.sum(img, axis=0)

    # Downscaling image (high-res is not necessary)
    # in XY, & recalculating voxel size
    if isinstance(img, da.Array):
        start = time()
        img = img.compute()
        print("Loading image took:", time() - start)
    start = time()
    img = resize(
        image=img,
        output_shape=(
            img.shape[0],
            img.shape[1] // downscale,
            img.shape[2] // downscale,
        ),
        anti_aliasing=True,
        preserve_range=True,
    )
    scaled_voxel = (
        voxel_size[0],
        voxel_size[1] * downscale,
        voxel_size[2] * downscale,
    )
    print("SUM-Channel projection and downscaling took:", time() - start)

    # rolling ball
    blurred = np.zeros_like(img)
    print("rolling ball:")
    for i in tqdm(range(img.shape[0])):
        background = rolling_ball(img[i], radius=100 / downscale)
        blurred[i] = img[i] - background

    # calculate threshold
    thresholds = []
    print("Calculating thresholds:")
    for i in tqdm(range(blurred.shape[0])):
        thresholds.append(threshold_li(blurred[i]))
    # Calculate average li threshold (among slices)
    threshold = sum(thresholds) / len(thresholds)
    print("Average Li threshold (of channel-SUM slices) =", threshold)
    print("Li threshold of stack:", threshold_li(blurred))

    # thresholding
    start = time()
    # print('Thresholding image')
    binary = blurred > threshold
    print("Thresholding image took:", time() - start)

    # Keep only the biggest object
    start = time()
    binary = label(binary)
    props = regionprops_table(binary, properties=["label", "area"])
    biggest_label = props["label"][
        list(props["area"]).index(max(props["area"]))
    ]
    binary = np.where(binary == biggest_label, 1, 0)
    print("Keeping only the biggest object took:", time() - start)

    # convex hull
    print("Creating convex hull:")
    for i in tqdm(range(binary.shape[0])):
        binary[i] = convex_hull_image(binary[i])

    return img, binary, scaled_voxel


def get_area_and_center(img: np.ndarray, voxel_size: tuple) -> (float, tuple):
    """
    Calculate the area and centroid of a binary image.

    For 3D scaffold segmentation, only one object is allowed.
    :param img: binary image.
    :param voxel_size: ZYX voxel size tuple
    :return: - area (float)
             - centroid tuple(float)
    """
    props = regionprops_table(
        img, properties=["label", "area", "centroid"], spacing=voxel_size
    )
    if len(props["label"]) != 1:
        raise RuntimeError(
            "Binary image has not a single object but:", len(props["label"])
        )
    print(
        "Area:",
        props["area"][0],
        "Center:",
        (
            props["centroid-0"][0],
            props["centroid-1"][0],
            props["centroid-2"][0],
        ),
    )
    return props["area"][0], (
        props["centroid-0"][0],
        props["centroid-1"][0],
        props["centroid-2"][0],
    )
