import numpy as np
import numpy.testing as nt

import iob_ia.utils.measure


def test_overlapping_labels():
    img1 = np.asarray(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 2, 2, 0],
            [0, 1, 2, 2, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 3, 3, 0],
        ]
    )
    img2 = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 6, 0, 0],
            [0, 1, 6, 0, 0],
            [0, 7, 7, 0, 0],
            [0, 7, 7, 0, 0],
        ]
    )
    expected = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 2, 2, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 3, 3, 0],
        ]
    )

    actual = iob_ia.utils.measure.overlapping_labels(img1, img2)
    nt.assert_array_equal(actual, expected)


# if __name__ == "__main__":
#    test_overlapping_labels()
