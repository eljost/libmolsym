import numpy as np
from pysisyphus.Geometry import Geometry
from pysisyphus.elem_data import MASS_DICT
import pytest
from scipy.spatial.distance import pdist, squareform

from libmolsym.constants import BOHR2ANG
from libmolsym.Arrangement import Arrangement
from libmolsym.main import center_of_mass, find_se_atoms, inertia_tensor, \
                           find_proper_rotations, get_arrangement, \
                           diagonalized_inertia_tensor


# [1] https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.23493
#     Algorithms for computer detection of symmetry elements in molecular systems
#     Beruski, Vidal, 2013


def pol2cart(r, phi):
    rad = np.deg2rad(phi)
    return (r*np.cos(rad), r*np.sin(rad))


def get_ring(r=2, atom_num=6, atom="C", rot=0.):
    step = 360. / atom_num
    phi = (np.arange(1, atom_num+1) * step) + rot
    x, y = pol2cart(r, phi)
    z = np.zeros_like(x)

    atoms = [atom] * x.size
    coords3d = np.stack((x, y, z), axis=1)
    masses = np.array([MASS_DICT[atom.lower()] for atom in atoms])
    return atoms, coords3d, masses


def ferrocene_dummy(top_rot=0):
    a, c, m = get_ring(atom_num=5)

    distance = 8.

    c_down = c.copy()
    c_down += np.array(((0, 0, -distance/2), ))

    a_top, c_top, m_top = get_ring(atom_num=5, rot=top_rot)
    c_top += np.array(((0, 0, +distance/2), ))

    # Iron
    fe_c = np.array(((0., 0., 0.), ))

    # Full
    cf = np.concatenate((c_down, fe_c, c_top), axis=0)
    fa = a + ["Fe"] + a
    masses = np.concatenate((m, (55.845, ), m_top))
    # g = Geometry(fa, cf)
    # g.jmol()
    return fa, cf, masses


def twisted_ferrocene_dummy():
    return ferrocene_dummy(top_rot=36)


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


def hydrogen_atom_func():
    coords3d = np.array(((0.000000,  0.000000,  0.000000), ))
    atoms = ("H", )
    masses = np.array((1.007825, ))
    return atoms, coords3d, masses


def hydrogen_molecule_func():
    coords3d = np.array((
                (0.000000,  0.000000,  0.000000),
                (0.000000,  0.000000,  1.000000),
    ))
    atoms = ("H", "H")
    masses = np.array((1.007825, 1.007825))
    return atoms, coords3d, masses


def c3_carbons_func():
    bond_length = 1.51 / BOHR2ANG
    coords3d = np.array((
                (0.000000, 0.000000, 0.000000),
                (bond_length, 0.000000, 0.000000),
                (*pol2cart(bond_length, 60), 0.000000),
    ))
    atoms = ("C", "C", "C")
    masses = np.array((12.0107, 12.0107, 12.0107))
    return atoms, coords3d, masses


def para_dinitrobenzene_func():
    coords3d = np.array((
         (-1.140982, -0.693078, 0.186458),
         ( 0.162518, -1.200978, 0.186458),
         ( 1.254018, -0.326078, 0.186458),
         ( 1.042218,  1.056722, 0.186458),
         (-0.261282,  1.564522, 0.186458),
         (-1.352882,  0.689622, 0.186458),
         (-1.992682, -1.375778, 0.186458),
         ( 2.271218, -0.722278, 0.186458),
         ( 1.893918,  1.739422, 0.186458),
         (-2.369982,  1.086022, 0.186458),
         ( 0.385253, -2.654006, 0.186458),
         (-0.627788, -3.349330, 0.186458),
         ( 1.560057, -3.013955, 0.186458),
         (-0.483996,  3.017552, 0.186458),
         ( 0.529054,  3.712862, 0.186458),
         (-1.658795,  3.377518, 0.186458),
    )) / BOHR2ANG
    atoms = "C C C C C C H H H H N O O N O O".split()
    masses = np.array([MASS_DICT[atom.lower()] for atom in atoms])
    return atoms, coords3d, masses


def eclipsed_ferrocene_func():
    coords3d = np.array((
        ( 0.00000,      0.00000,      0.00000),
        ( 2.31483,      0.00000,     -1.88583),
        ( 1.22935,      0.00000,     -1.88702),
        ( 0.37989,      1.16918,     -1.88702),
        ( 0.71532,     -2.20153,     -1.88583),
        ( 0.71532,      2.20153,     -1.88583),
        (-0.99457,      0.72259,     -1.88702),
        (-1.87273,      1.36062,     -1.88583),
        (-0.99457,     -0.72259,     -1.88702),
        (-1.87273,     -1.36062,     -1.88583),
        ( 0.37989,     -1.16918,     -1.88702),
        ( 0.71532,      2.20153,      1.88583),
        ( 0.37989,      1.16918,      1.88702),
        ( 1.22935,      0.00000,      1.88702),
        (-1.87273,      1.36062,      1.88583),
        ( 2.31483,      0.00000,      1.88583),
        ( 0.37989,     -1.16918,      1.88702),
        ( 0.71532,     -2.20153,      1.88583),
        (-0.99457,     -0.72259,      1.88702),
        (-1.87273,     -1.36062,      1.88583),
        (-0.99457,      0.72259,      1.88702),
    )) / BOHR2ANG
    atoms = "Fe H C C H H C H C H C H C C H H C H C H C".split()
    masses = np.array([MASS_DICT[atom.lower()] for atom in atoms])
    return atoms, coords3d, masses


def twisted_ferrocene_func():
    coords3d = np.array((
            ( 0.00000,  0.00000,  0.00000),
            ( 2.31483,  0.00000,  1.88574),
            ( 1.22935,  0.00000,  1.88696),
            ( 0.37989, -1.16918,  1.88696),
            ( 0.71532,  2.20153,  1.88574),
            ( 0.71532, -2.20153,  1.88574),
            (-0.99457, -0.72260,  1.88696),
            (-1.87274, -1.36062,  1.88574),
            (-0.99457,  0.72260,  1.88696),
            (-1.87274,  1.36062,  1.88574),
            ( 0.37989,  1.16918,  1.88696),
            ( 1.87274, -1.36062, -1.88574),
            ( 0.99457, -0.72260, -1.88696),
            ( 0.99457,  0.72260, -1.88696),
            (-0.71532, -2.20153, -1.88574),
            ( 1.87274,  1.36062, -1.88574),
            (-0.37989,  1.16918, -1.88696),
            (-0.71532,  2.20153, -1.88574),
            (-1.22935,  0.00000, -1.88696),
            (-2.31483,  0.00000, -1.88574),
            (-0.37989, -1.16918, -1.88696),
    ))
    atoms = "Fe H C C H H C H C H C H C C H H C H C H C".split()
    masses = np.array([MASS_DICT[atom.lower()] for atom in atoms])
    return atoms, coords3d, masses


def fluoromethane():
    coords3d = np.array((
        (-0.280994,  -0.065815,   0.186478),
        ( 0.107225,   1.204907,   0.012837),
        (-0.735494,  -0.438815,  -0.739222),
        (-1.014594,  -0.116015,   0.999978),
        ( 0.589506,  -0.682515,   0.440678),
    )) / BOHR2ANG
    atoms = "C F H H H".split()
    masses = np.array([MASS_DICT[atom.lower()] for atom in atoms])
    return atoms, coords3d, masses


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


def test_proper_rotations(ammonia):
    atoms, coords3d, masses = ammonia
    coords3d = coords3d - center_of_mass(coords3d, masses)
    cdm = pdist(coords3d)
    dist_mat = squareform(cdm)
    se_atoms = find_se_atoms(atoms, dist_mat)
    h_se_atoms = se_atoms["h0"]
    find_proper_rotations(coords3d, masses, h_se_atoms)


@pytest.mark.parametrize(
    "func, ref_arr, kwargs", [
        (hydrogen_atom_func, Arrangement.SINGLE_ATOM, {}),
        (hydrogen_molecule_func, Arrangement.LINEAR, {}),
        (c3_carbons_func, Arrangement.REGULAR_POLYGON, {}),
        (para_dinitrobenzene_func, Arrangement.IRREGULAR_POLYGON, {}),
        (eclipsed_ferrocene_func, Arrangement.PROLATE_SYMMETRIC_TOP, {"prec": 1e-2,}),
        (fluoromethane, Arrangement.PROLATE_SYMMETRIC_TOP, {"prec": 1e-3}),
        (twisted_ferrocene_dummy, Arrangement.PROLATE_SYMMETRIC_TOP, {}),
        (ferrocene_dummy, Arrangement.PROLATE_SYMMETRIC_TOP, {}),
    ]
)
def test_arrangement(func, ref_arr, kwargs):
    atoms, coords3d, masses = func()
    coords3d -= center_of_mass(coords3d, masses)

    pms, pas = diagonalized_inertia_tensor(coords3d, masses)
    print("pms", pms)
    arr = get_arrangement(pms, **kwargs)
    assert arr == ref_arr


# fn = "lib:h2o.xyz"
# bla(fn)

# fn = "lib:benzene.xyz"
# fn = "xtbopt.xyz"
# bla(fn)
