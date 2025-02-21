import napari
import numpy as np
from typing import Optional


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
