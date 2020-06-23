import numpy as np

from libmolsym.main import center_of_mass, centroid, get_mass, \
                           get_arrangement, inertia_tensor


class SEA:

    def __init__(self, atom, coords3d):
        self.atom = atom
        self.coords3d = np.array(coords3d).reshape(-1, 3)

        self.atom_mass = get_mass(self.atom)

        self._I = inertia_tensor(self.coords3d, self.masses)
        princ_moms, princ_axs = np.linalg.eigh(self.I)
        self.princ_moms = princ_moms
        self.princ_axs = princ_axs
        self.arrangement = get_arrangement(self.princ_moms)

    def __len__(self):
        return len(self.coords3d)

    @property
    def atoms(self):
        return [self.atom] * len(self)

    @property
    def masses(self):
        return np.full(len(self), self.atom_mass)

    @property
    def centroid(self):
        return centroid(self.coords3d)

    @property
    def center_of_mass(self):
        return center_of_mass(self.coords3d, self.masses)

    @property
    def I(self):
        return self._I

    def __str__(self):
        return f'SEA({len(self)} {self.atom} atoms)"'
