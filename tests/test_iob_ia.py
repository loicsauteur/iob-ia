import numpy as np
import pytest
import numpy.testing as nt


def test_something():
    pass


@pytest.mark.skip(reason="Deprecated")
def test_remove_big_objects():
    from iob_ia.utils.segment import remove_big_objects
    labeled_image = np.array(
        [
            [1, 0, 0, 0, 4],
            [0, 2, 2, 0, 4],
            [0, 0, 0, 0, 4],
            [0, 0, 3, 3, 4],
            [0, 0, 3, 0, 0],
        ], dtype=int
    )
    expected = np.array(
        [
            [1, 0, 0, 0, 4],
            [0, 2, 2, 0, 4],
            [0, 0, 0, 0, 4],
            [0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0],
        ], dtype=int
    )
    observed = remove_big_objects(labeled_image, max_size=3)
    np.testing.assert_array_equal(observed, expected)


def test_read_nd2():
    from iob_ia.utils.io_utils import read_nd2
    import os
    path = 'G:\\20250211_MarcDu\\20x_Testfile_405(DAPI)_488(CD45)_546(CD3)_647(' \
           'CD31)\\3-20x.nd2'
    # only do if file exists, this is for local testing
    if not path.endswith('.nd2'):
        with pytest.raises(ValueError):
            read_nd2(path)
    elif os.path.exists(path):
        data, voxel_size = read_nd2(path)
        assert data.shape == (4, 68, 2048, 2048), f'Expected shape (4, 68, 2048, ' \
                                                  f'2048) but got {data.shape}'
        expected_voxels = [2.0, 0.43, 0.43]
        nt.assert_array_almost_equal(
            expected_voxels, np.asarray(voxel_size), decimal=2,
            err_msg=f'Reading nd2 voxel size failed, '
                    f'expected {expected_voxels}, got {voxel_size}')
    else:
        with pytest.raises(FileNotFoundError):
            read_nd2(path)
