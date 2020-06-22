import numpy as np
from pysisyphus.Geometry import Geometry
import pytest
from scipy.spatial.distance import pdist, squareform

from libmolsym.constants import BOHR2ANG
from libmolsym.main import center_of_mass, find_se_atoms, inertia_tensor


# [1] https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.23493
#     Algorithms for computer detection of symmetry elements in molecular systems
#     Beruski, Vidal, 2013


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


@pytest.fixture
def ammonia():
    coords3d = np.array((
                    ( 0.000000,  0.000000,  0.000000),
                    ( 0.000000,  0.000000,  1.008000),
                    ( 0.950353,  0.000000, -0.336000),
                    (-0.475176, -0.823029, -0.336000),
    )) / BOHR2ANG  # From Å to Bohr
    atoms = "N H H H".split()
    masses = np.array((14.003074, 1.007825, 1.007825, 1.007825))
    return (atoms, coords3d, masses)


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


def test_center_of_mass(ammonia):
    """See [1], Numerical Examples, p. 296, top left column."""
    atoms, coords3d, masses = ammonia
    com = center_of_mass(coords3d, masses)

    com_ref = np.array((0.0281, -0.0487, 0.0199)) / BOHR2ANG
    np.testing.assert_allclose(com, com_ref, atol=1e-4)


def test_inertia_tensor(ammonia):
    """See [1], Numerical Examples, p. 296, top left column."""
    atoms, coords3d, masses = ammonia
    coords3d = coords3d - center_of_mass(coords3d, masses)

    I = inertia_tensor(coords3d, masses)

    I_ref = np.array((
                ( 1.8871, -0.4175,  0.1704),
                (-0.4175,  2.3692, -0.2952),
                ( 0.1704, -0.2952,  1.7666),
    ))  # in Å²/amu
    I_ref = I_ref / BOHR2ANG**2  # to Bohr²/amu
    np.testing.assert_allclose(I, I_ref, atol=1.21e-4)


# fn = "lib:h2o.xyz"
# bla(fn)

# fn = "lib:benzene.xyz"
# fn = "xtbopt.xyz"
# bla(fn)
