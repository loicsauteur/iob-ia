import os.path
import warnings
from typing import Optional

import pandas as pd
from tqdm import tqdm

from iob_ia.utils.io_utils import read_image


def check_images(
    image_paths: list[str],
    channels_to_segment: list[int],
    voxel_size: Optional[tuple[float]] = None,
):
    """
    Check that all images have the same number of channels and voxel sizes.

    :param image_paths: list of string paths
    :param channels_to_segment: list of integers for 0-based channels to segment
    :param voxel_size: Optional, to override voxel size (e.g. None from metadata)
    :return: tuple of final voxel size or None (if images are too different)
    """
    # Collect metadata to check for all images
    img = None
    all_channel_names = []
    all_voxel_sizes = []
    for i in tqdm(range(len(image_paths))):
        img = read_image(image_paths[i])
        all_voxel_sizes.append(img.physical_pixel_sizes)
        all_channel_names.append(img.channel_names)
    # for p in image_paths:
    #    img = read_image(p)
    #    all_voxel_sizes.append(img.physical_pixel_sizes)
    #    all_channel_names.append(img.channel_names)
    del img

    # Check booleans
    channels_ok = True  # check if images have all same number of channels
    channels_equal = True  # check if channel_to_segment is available
    voxels_ok = True  # check if None in voxel sizes
    voxel_equal = True  # check if all voxel sizes are the same
    n_channels = len(all_channel_names[0])  # number of channels of first image
    vs = all_voxel_sizes[0]  # voxel size of first image
    for i in range(len(image_paths)):
        # Check that all images have the same number of channels
        if len(all_channel_names[i]) != n_channels:
            channels_ok = False
        # Check that the channel_to_segment is available
        if max(channels_to_segment) + 1 > len(all_channel_names[i]):
            channels_equal = False
        # Check that the voxel metadata is OK (not None)
        if None in all_voxel_sizes[i]:
            voxels_ok = False
        # Check that all voxel sizes are the same
        if all_voxel_sizes[i] != vs:
            voxel_equal = False

    # Check if voxel_size info has been supplied
    if not voxels_ok and voxel_size is not None:
        print("Using manually defined voxel size.")
        voxels_ok = True
    elif voxels_ok and voxel_size is not None:
        rounded_voxel_size = tuple([round(x, 3) for x in all_voxel_sizes[0]])
        print(
            "rounded vs supplied voxel size equal?:",
            rounded_voxel_size == voxel_size,
        )
        if rounded_voxel_size != voxel_size:
            print(
                f"Image voxel size is: {all_voxel_sizes[0]}, but you indicated "
                f"{voxel_size}. Using the. {all_voxel_sizes[0]}"
            )
            voxel_size = all_voxel_sizes[0]
        # in either case, voxel sizes are OK
        voxels_ok = True
    elif voxels_ok and voxel_size is None:
        voxel_size = all_voxel_sizes[0]
        voxels_ok = True

    # Check if all images are OK, otherwise print errors
    if channels_ok and channels_equal and voxels_ok and voxel_equal:
        print("All good!")
        return voxel_size
    else:
        message = "\nThere are some problems:\n"
        if not channels_ok:
            message += (
                "- some images don't have the same number of channels.\n"
            )
        if not channels_equal:
            message += "- some images lack channels to be segment.\n"
        if not voxels_ok:
            message += "- some images do not have proper voxel sizes in the metadata.\n"
        if not voxel_equal:
            message += "- the images do not have the same voxel sizes.\n"
        message += (
            "-> Please check your images and remove "
            "problematic ones from the folder"
        )
        # Show summary of supplied images with metadata
        image_names = [os.path.basename(p) for p in image_paths]
        check_sum = {
            "images": image_names,
            "channel names": all_channel_names,
            "voxel sizes": all_voxel_sizes,
        }
        check_sum = pd.DataFrame.from_dict(check_sum)
        warnings.warn(message, stacklevel=2)
        print(check_sum)
        # Return None for voxel_size
        return None
