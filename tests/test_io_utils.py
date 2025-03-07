import pytest
import numpy.testing as nt
import numpy as np


def test_read_image():
    from iob_ia.utils.io_utils import read_image
    from pathlib import Path
    path_resources = Path(__file__).parent.parent / 'resources'
    path_list = [
        path_resources / 'test_image.nd2',
        path_resources / 'test_image.tif',
        path_resources / 'does_not_exist.nd2',
        path_resources / 'test_image.jpg'
    ]

    with pytest.raises(FileNotFoundError):
        read_image(str(path_list[2].absolute()))
    with pytest.raises(NotImplementedError):
        read_image(str(path_list[3].absolute()))

    img_nd2 = read_image(str(path_list[0].absolute()))
    img_tif = read_image(str(path_list[1].absolute()))
    expected_voxels = [5.0, 0.69, 0.69]
    expected_shape = (4, 18, 40, 40)

    # Check nd2 voxel size
    nt.assert_array_almost_equal(
        np.asarray(img_nd2.physical_pixel_sizes),
        np.asarray(expected_voxels),
        decimal=2,
        err_msg=f'Reading nd2 voxel size failed, '
                f'expected {expected_voxels}, got {img_nd2.physical_pixel_sizes}'
    )
    # Check nd2 shape
    assert expected_shape == img_nd2.get_data().shape, f'Expected shape {expected_shape} but got {img_nd2.get_data().shape}'

    # Check tif voxel size
    nt.assert_array_almost_equal(
        np.asarray(img_tif.physical_pixel_sizes),
        np.asarray(expected_voxels),
        decimal=2,
        err_msg=f'Reading tif voxel size failed, '
                f'expected {expected_voxels}, got {img_tif.physical_pixel_sizes}'
    )
    # Check tif shape
    assert expected_shape == img_tif.get_data().shape, f'Expected shape {expected_shape} but got {img_tif.get_data().shape}'
