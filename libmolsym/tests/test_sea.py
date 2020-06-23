import numpy as np

from libmolsym import Arrangement, SEA
from libmolsym.dummy_data import get_ring


def test_c6_reg_polygon():
    sea = get_ring(r=2, atom_num=6, atom="C", rot=0.)
    print(sea)

    origin = np.zeros(3)

    atol = 1e-15
    com = sea.center_of_mass
    centroid = sea.centroid
    np.testing.assert_allclose(com, origin, atol=atol)
    np.testing.assert_allclose(centroid, origin, atol=atol)
    np.testing.assert_allclose(centroid, com, atol=atol)

    assert sea.arrangement == Arrangement.REGULAR_POLYGON
