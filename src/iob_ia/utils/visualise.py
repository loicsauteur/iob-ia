import napari
import numpy as np
from typing import Optional, List


def get_viewer():
    viewer = napari.viewer.current_viewer()
    if viewer is None:
        viewer = napari.Viewer()
    return viewer


def add_image(
    img: np.ndarray, name: str, colormap: str = 'gray',
    scale: Optional[tuple] = None
) -> None:
    """
    Add an image to the napari viewer.

    :param img: single channel image
    :param name: str for the layer name
    :param colormap: str LUT
    :param scale: tuple of voxel size, will scale for 3D view
    :return:
    """
    viewer = get_viewer()
    viewer.add_image(
        img, name=name, blending='additive',
        colormap=colormap, scale=scale
    )


def add_labels(img: np.ndarray, name: str, scale: Optional[tuple] = None) -> None:
    """
    Add labels to the napari viewer.

    :param img: label image
    :param name: str  for the layer name
    :param scale: tuple of voxel size, will scale for 3D view
    :return:
    """
    viewer = get_viewer()
    viewer.add_labels(img, name=name, scale=scale)


def add_pair(
    img: np.ndarray, labels: np.ndarray,
    name: str, colormap: str = 'gray',
    scale: Optional[tuple] = None
) -> None:
    """
    Add an image and the corresponding labels to the napari viewer.

    :param img: single channel image
    :param labels: label image
    :param name: str for the layer name
    :param colormap: str LUT
    :param scale: tuple of voxel size, will scale for 3D view
    :return:
    """
    if img.shape != labels.shape:
        raise RuntimeError(f'Image and labels should have the same shape. '
                           f'Image = {img.shape}; labels = {labels.shape}')
    viewer = get_viewer()
    viewer.add_image(
        img, name=name, blending='additive', colormap=colormap, scale=scale
    )
    viewer.add_labels(labels, name=name + '_segmentation', scale=scale)


def add_multichannel_image(
    img: np.ndarray,
    name: str,
    channel_names: Optional[List] = None,
    scale: Optional[tuple] = None
) -> None:
    """
    Add a multichannel image to the napari viewer.

    Up to 5 channels are supported.

    :param img: C(Z)YX image
    :param name: base name for the layers
    :param channel_names: Optional list of channel names
    :param scale: tuple of voxel size, will scale for 3D view
    :return:
    """
    # Check if image is 2D or 3D (plus channel axis)
    if img.ndim not in [3, 4]:
        raise ValueError(f'Image must have 2 or 3 dimensions, with channels. '
                         f'You have {img.ndim} dimensions. '
                         f'With an image shape of: {img.shape}.')
    # Check that there are not more than 5 channels
    if img.shape[0] > 5:
        raise ValueError(f'Image must have 5 or fewer channels. '
                         f'You have {img.shape[0]} channels. '
                         f'With an image shape of: {img.shape}.')
    if img.shape[0] == 1:
        add_image(img, name=name, scale=scale)
    if channel_names is None:
        channel_names = [f'ch1{i}' for i in range(1, img.shape[0] + 1)]
    if img.shape[0] != len(channel_names):
        raise ValueError(f'Channels names do not correspond to the number '
                         f'of image channels. Channels names = {channel_names}, and '
                         f'image has {img.shape[0]} channels.')
    # Add image channels with hard-coded LUTs
    colors = ["blue", "green", "red", "magenta", "gray"]
    for ch in range(img.shape[0]):
        add_image(
            img[ch],
            name=f'{name}_{channel_names[ch]}',
            colormap=colors[ch],
            scale=scale
        )

