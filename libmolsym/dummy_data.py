import numpy as np

from libmolsym import SEA


def pol2cart(r, phi):
    """Cartesian coordinates for given polar coordinates."""

    rad = np.deg2rad(phi)
    return (r*np.cos(rad), r*np.sin(rad))


def get_ring(r=2, atom_num=6, atom="C", rot=0.):
    """Get ring with symmetrically equivalent atoms."""

    assert atom_num >= 3, "atom_num must be >= 3!"
    step = 360. / atom_num
    phi = (np.arange(1, atom_num+1) * step) + rot
    x, y = pol2cart(r, phi)
    z = np.zeros_like(x)

    coords3d = np.stack((x, y, z), axis=1)
    sea = SEA(atom, coords3d)
    return sea
