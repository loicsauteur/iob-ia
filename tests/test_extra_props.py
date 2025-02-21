import numpy as np
import numpy.testing as nt
import pytest
from skimage.measure import regionprops_table, regionprops
from skimage.morphology import ball, disk
import iob_ia.utils.extra_props as ep


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_projected_extra_props():
    radius = 10
    sphere = ball(radius)
    disk_props = regionprops(disk(radius))
    expected_perimeter = disk_props[0].perimeter
    expected_hull = disk_props[0].area_convex

    table = regionprops_table(
        sphere, extra_properties=(
            ep.projected_circularity,
            ep.projected_perimeter,
            ep.projected_convex_area
        )
    )

    nt.assert_array_almost_equal(table['projected_circularity'], [0.9], decimal=1)
    nt.assert_array_equal(table['projected_perimeter'], [expected_perimeter])
    nt.assert_array_equal(table['projected_convex_area'], [expected_hull])

