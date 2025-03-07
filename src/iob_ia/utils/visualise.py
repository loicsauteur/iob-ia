import napari
import numpy as np
from typing import Optional, List, Union, Dict
from iob_ia.utils.segment import measure_props
from napari.utils.colormaps import DirectLabelColormap


__colors__ = {
    'gray': [1., 1., 1., 1.],
    'red': [1., 0., 0., 1.],
    'green': [0., 1., 0., 1.],
    'blue': [0., 0., 1., 1.],
    'magenta': [1., 0., 1., 1.],
    'cyan': [0., 1., 1., 1.],
    'other': [
        np.random.random(1)[0],
        np.random.random(1)[0],
        np.random.random(1)[0],
        1.
    ]
}


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


def add_labels(
    img: np.ndarray,
    name: str,
    scale: Optional[tuple] = None,
    features: Optional[Dict] = None,
    colormap: Optional = None
) -> None:
    """
    Add labels to the napari viewer.

    :param img: label image
    :param name: str  for the layer name
    :param scale: tuple of voxel size, will scale for 3D view
    :param features: dict for label properties
    :param colormap: napari DirectColorMap. Default is None
    :return:
    """
    viewer = get_viewer()
    viewer.add_labels(img, name=name, scale=scale, features=features, colormap=colormap)


def add_pair(
    img: np.ndarray, labels: np.ndarray,
    name: str, colormap: str = 'gray',
    scale: Optional[tuple] = None,
    features: Optional[Dict] = None
) -> None:
    """
    Add an image and the corresponding labels to the napari viewer.

    :param img: single channel image
    :param labels: label image
    :param name: str for the layer name
    :param colormap: str LUT
    :param scale: tuple of voxel size, will scale for 3D view
    :param features: dict for label properties
    :return:
    """
    if img.shape != labels.shape:
        raise RuntimeError(f'Image and labels should have the same shape. '
                           f'Image = {img.shape}; labels = {labels.shape}')
    viewer = get_viewer()
    viewer.add_image(
        img, name=name, blending='additive', colormap=colormap, scale=scale
    )
    viewer.add_labels(labels, name=name + '_segmentation', scale=scale,
                      features=features)


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


def create_napari_features(
    img_label: Optional[np.ndarray] = None,
    img_intensity: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    voxel_size: Union[float, tuple] = (1, 1, 1),
    props_table: Optional[Dict] = None
) -> dict:
    """
    Create a dictionary of label features, that is compatible with a napari label layer.

    :param img_label: label image
    :param img_intensity: intensity image (CZYX or list of ZYX)
    :param voxel_size: ZYX voxel size in microns for calibrated measurements
    :param props_table: regionprops_table that was already measured
    :return: features dictionary for napari label layer
    """
    if img_label is None and props_table is None:
        raise ValueError(
            f'You must supply either the label image or the regionprops_table. '
        )
    # If properties not already supplied...
    if props_table is None:
        props_table = measure_props(
            img_label=img_label,
            img_intensity=img_intensity,
            voxel_size=voxel_size
        )
    label_max = props_table['label'].max()
    features = {}
    for k, v in props_table.items():
        # Don't include the label
        if k == 'label':
            continue
        # Per measurement, create a 'default' dictionary entry for all labels + 0
        features[k] = ['none'] * (label_max + 1)
        # Assign the proper values to the feature value array
        for i, _label in enumerate(props_table['label']):
            features[k][_label] = v[i]
    return features


def single_colormap(color: str, n_labels: int = 65535) -> DirectLabelColormap:
    """
    Create a custom color map: for all labels a single color.

    If the color is 'unknown', a random color is generated.

    :param color:
    :param n_labels: max number of labels in image
    :return: DirectLabelColormap
    """
    # Check the color
    if color not in __colors__:
        print(f'Color {color} not found, using random color ', end='')
        color = __colors__['other']
        print(color)
    else:
        color = __colors__[color]
    n_labels = n_labels + 1
    # Create a dictionary mapping labels to RGBA colors
    color_dict = {}
    for i in range(1, n_labels):
        color_dict[i] = color
    color_dict[None] = "transparent"
    color_dict[0] = "transparent"
    return DirectLabelColormap(color_dict=color_dict)
