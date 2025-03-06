import numpy as np
import pytest
import numpy.testing as nt


def test_something():
    pass


def test_measure_props():
    from iob_ia.utils.segment import measure_props
    random_lbl = np.random.randint(0, 5, size=(5, 100, 100))
    img_list_correct = [
        np.random.randint(low=100, high=4000, size=(5, 100, 100)),
        np.random.randint(low=100, high=4000, size=(5, 100, 100)),
        np.random.randint(low=100, high=4000, size=(5, 100, 100)),
    ]
    img_list_wrong = [
        np.random.randint(low=100, high=4000, size=(5, 100, 100)),
        np.random.randint(low=100, high=4000, size=(5, 100, 100)),
        np.random.randint(low=100, high=4000, size=(3, 100, 100)),
    ]
    img_multich_correct = np.random.randint(low=100, high=4000, size=(5, 5, 100, 100))
    img_multich_wrong1 = np.random.randint(low=100, high=4000, size=(5, 3, 90, 100))
    img_multich_wrong2 = np.random.randint(low=100, high=4000, size=(5, 100, 100, 5))

    # Check that errors are raised
    with pytest.raises(ValueError):
        # with a list of intensity images with the wrong shape
        measure_props(img_label=random_lbl, img_intensity=img_list_wrong,
                      voxel_size=(1, 1, 1))
        # with a intensity multichannel images of the wrong shape
        measure_props(img_label=random_lbl, img_intensity=img_multich_wrong1,
                      voxel_size=(1, 1, 1))
        measure_props(img_label=random_lbl, img_intensity=img_multich_wrong2,
                      voxel_size=(1, 1, 1))
        # with a non-3D label image
        measure_props(img_label=np.random.randint(0, 10, size=(100, 100)))

    # Check that the output is correct
    assert isinstance(
        measure_props(img_label=random_lbl, img_intensity=img_list_correct,
                      voxel_size=(1, 1, 1)),
        dict
    ), f'Function did not return a dictionary'
    assert isinstance(
        measure_props(img_label=random_lbl, img_intensity=img_multich_correct,
                      voxel_size=(1, 1, 1)),
        dict
    ), f'Function did not return a dictionary'
    # check that the label is in the table
    assert 'label' in measure_props(
        img_label=random_lbl,
        img_intensity=img_multich_correct,
        voxel_size=(1, 1, 1)), f'Function did not return a dictionary with ' \
                               f'"label" in the keys'
    assert 'label' in measure_props(
        img_label=random_lbl,
        voxel_size=(1, 1, 1)), f'Function did not return a dictionary with ' \
                               f'"label" in the keys'


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
