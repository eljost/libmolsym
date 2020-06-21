import numpy as np
from pysisyphus.Geometry import Geometry
import pytest
from scipy.spatial.distance import pdist, squareform

from libmolsym.main import find_se_atoms


def pol2cart(r, phi):
    rad = np.deg2rad(phi)
    return (r*np.cos(rad), r*np.sin(rad))


def get_hexagonal_geom(r=2, distort=False):
    phi = np.array((30, 90, 150, 210, 270, 330))
    x, y = pol2cart(2, phi)
    z = np.zeros_like(x)
    if distort:
        y[1] = 3

    atoms = "C" * x.size
    coords3d = np.stack((x, y, z), axis=1)
    geom = Geometry(atoms, coords3d.flatten())
    return geom


@pytest.mark.parametrize(
    "distort, sea_groups", [
        (False, 1),
        (True, 4),
    ]
)
def test_sea(distort, sea_groups):
    geom = get_hexagonal_geom(distort=distort)
    geom.standard_orientation()
    geom.coords3d -= geom.center_of_mass
    cdm = pdist(geom.coords3d)
    dist_mat = squareform(cdm)
    se_atoms = find_se_atoms(geom.atoms, dist_mat)

    assert len(se_atoms.keys()) == sea_groups
    print(se_atoms)


# fn = "lib:h2o.xyz"
# bla(fn)

# fn = "lib:benzene.xyz"
# fn = "xtbopt.xyz"
# bla(fn)
