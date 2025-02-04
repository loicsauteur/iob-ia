import numpy as np

from iob_ia.utils.segment import remove_big_objects


def test_something():
    pass


def test_remove_big_objects():
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
