import numpy as np
from skimage.measure import label, regionprops, regionprops_table


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
    if len(region_mask.shape) != 3:
        raise ValueError('Input must be a 3D label image.')
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
        z - center[0]) ** 2 <= 40 ** 2
    img_sphere[sphere] = 1
    return img_sphere


'''
if __name__ == '__main__':
    path = 'G:\\20241120_IOB_Magdalena\\iob-ia\\resources\\wannabe_sphere_and_ellipse.tif'
    path1 = 'G:\\20241120_IOB_Magdalena\\iob-ia\\resources\\wannabe_sphere.tif'
    from tifffile import imread
    from skimage.measure import label
    img = imread(path)
    img1 = label(imread(path1)>1)
    img = img > 1
    img = label(img)
    #print(img.shape, img.dtype, img.max())
    #print(img1.shape, img1.dtype, img1.max())

    props = regionprops_table(img, extra_properties=(projected_circularity, projected_perimeter, projected_hull))
    props1 = regionprops_table(img1, extra_properties=(projected_circularity,projected_perimeter, projected_hull))
    print(props)
    print()
    print(props1)
'''
