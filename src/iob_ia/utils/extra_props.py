"""
Measure extra properties for regionprops_table.

Allows to include measurements of projected properties,
which cannot be calculated in 3D.
"""

import numpy as np
from skimage.measure import regionprops

# List of all extra properties
__all_extra_props__ = [
    "projected_area",
    "projected_circularity",
    "projected_perimeter",
    "projected_convex_area",
]
__calibrated_extr_props__ = [
    "projected_area",
    "projected_perimeter",
    "projected_convex_area",
]


def projected_circularity(region_mask: np.ndarray) -> float:
    """
    Calculate the projected circularity of a region.

    Circularity = 4 * pi * Area / Perimeter^2

    :param region_mask: mask of a region
    :return: Circularity
    """
    img_proj = project_mask(region_mask)
    props = regionprops(img_proj)
    return (4 * np.pi * props[0].area) / (props[0].perimeter_crofton ** 2)


def projected_perimeter(region_mask: np.ndarray) -> float:
    """
    Calculate the projected perimeter of a region.

    :param region_mask: mask of a region
    :return: Perimeter
    """
    img_proj = project_mask(region_mask)
    props = regionprops(img_proj)
    return props[0].perimeter


def projected_convex_area(region_mask: np.ndarray) -> int:
    """
    Calculate the projected hull area of a region.

    :param region_mask: mask of a region
    :return: convex hull area
    """
    img_proj = project_mask(region_mask)
    props = regionprops(img_proj)
    return props[0].area_convex


def project_mask(region_mask: np.ndarray) -> np.ndarray:
    """
    Project the mask along the first (Z) axis.

    Helper function to calculate the extra props.
    :param region_mask: mask of a region
    :return: Z-projected mask
    """
    if len(region_mask.shape) != 3:
        raise ValueError("Input must be a 3D label image.")
    # Project along the first (Z) axis
    img_proj = np.max(region_mask, axis=0)
    return np.asarray(img_proj, dtype=np.uint8)


def projected_area(region_mask: np.ndarray) -> int:
    """
    Calculate the projected area of a region.

    :param region_mask: mask of a region
    :return: area
    """
    img_proj = project_mask(region_mask)
    props = regionprops(img_proj)
    return props[0].area


def calibrate_extra_properties(table: dict, voxel_size: tuple) -> dict:
    """
    Calibrate extra properties that should be calibrated.

    :param table: dict (regionprops_table)
    :param voxel_size: ZYX voxel size
    :return: modified regionprops_table
    """
    # Extra props should only be for 3D images
    if len(voxel_size) != 3:
        raise ValueError(
            f"Extra properties are only for 3D images. Got voxel size of {voxel_size}."
        )
    # Skip calibration if voxel size is all 1's
    if voxel_size == (1.0, 1.0, 1.0):
        return table

    if voxel_size[1] != voxel_size[2]:
        raise NotImplementedError(
            f"Different XY pixel size is not supported. Got {voxel_size[1:]}."
        )
    for prop in __calibrated_extr_props__:
        # Skip props that are not in the table
        if prop not in table:
            continue
        if "area" in prop:
            # area properties
            table[prop] = table[prop] * voxel_size[1] ** 2
        else:
            # i.e. perimeter
            table[prop] = table[prop] * voxel_size[1]
    return table


def make_sphere() -> np.ndarray:
    """
    Makes a sphere like Robert Haase.

    See:
    https://forum.image.sc/t/measure-sphericity-using-python/95826/15
    :return: 100x100x100 binary image of a sphere
    """
    img_sphere = np.zeros((100, 100, 100), dtype=np.uint8)
    center = np.array(img_sphere.shape) // 2
    z, y, x = np.ogrid[:100, :100, :100]
    sphere = (x - center[2]) ** 2 + (y - center[1]) ** 2 + (
        z - center[0]
    ) ** 2 <= 40**2
    img_sphere[sphere] = 1
    return img_sphere
