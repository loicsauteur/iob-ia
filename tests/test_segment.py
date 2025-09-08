import math

import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_segment_2d_sub_projections():
    # create random image
    img = np.random.randint(0, 255, size=(10, 256, 256), dtype=np.uint8)
    voxel_size = (5, 0.5, 0.5)

    from iob_ia.utils.segment import segment_2d_sub_projections

    substack_size = 40
    mask_out, img_out, new_voxel_size = segment_2d_sub_projections(
        img, voxel_size, substack_size=substack_size
    )

    # check the output
    assert new_voxel_size == (
        substack_size,
        voxel_size[1],
        voxel_size[2],
    ), "Voxel size different than expected!"
    assert mask_out.shape == (
        math.ceil(img.shape[0] * voxel_size[0] / substack_size),
        img.shape[1],
        img.shape[1],
    ), "Output shape different than expected!"
