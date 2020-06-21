from math import log10


def get_AtomDist(prec=1e-4):
    assert prec > 0.
    prec_fmt = abs(int(log10(prec)))

    class AtomDist(tuple):
        """Tuple, allowing equivalence with a certain precision."""

        def __eq__(self, other):
            this_atom, this_dist = self
            other_atom, other_dist = other
            return (this_atom == other_atom) \
                   and (abs(this_dist - other_dist) < prec)

        def __hash__(self):
            nonlocal prec_fmt
            this_atom, this_dist = self
            return hash(f"{this_atom}{this_dist:.{prec_fmt}f}")
    return AtomDist


def handle_rows(atoms, dist_mat, row_inds):
    """Find symmetrically equivalent atoms for given subset
    of the distance matrix.

    Parameters
    ----------
    atoms : iterable of strings
        Iterable containing the atom types.
    dist_mat : np.array, (atoms.size, atoms.size)
        Distance matrix.
    row_inds : list
        Row indices, with all rows belonging to the same atom type.

    Returns
    -------
    sea_rows : list of lists
        List of lists containing symmetically equivalent atoms.
    """
    AtomDist = get_AtomDist()

    atom_dist_sets = list()
    row_inds = row_inds.copy()
    for row_ind in row_inds:
        dists = dist_mat[:, row_ind]
        atom_dist_sets.append(
            set([AtomDist((atom, dist)) for atom, dist in zip(atoms, dists)])
        )

    sea_rows = list()
    while atom_dist_sets:
        ads_i = atom_dist_sets.pop(0)
        row_i = row_inds.pop(0)

        # same_rows = [row_inds[i] for i, ads in enumerate(atom_dist_sets)
                     # if ads == ads_i]
        same_rows = [row_i]
        j = 0
        while atom_dist_sets:
            if atom_dist_sets[j] == ads_i:
                row_ind = row_inds[j]
                same_rows.append(row_ind)
                index = row_inds.index(row_ind)
                atom_dist_sets.pop(index)
                row_inds.pop(index)
            else:
                j += 1

            if j == len(atom_dist_sets):
                break

        sea_rows.append(same_rows)
    return sea_rows


def find_se_atoms(atoms, dist_mat):
    """Find symmetrically equivalent atoms for the given distance matrix.

    Parameters
    ----------
    atoms : iterable of strings
        Iterable containing the atom types.
    dist_mat : np.array, (atoms.size, atoms.size)
        Distance matrix.

    Returns
    -------
    se_atoms : dict
        Dictionary with one key-value pair for every set of symmetrically
        equivalent atoms.
    """
    atoms = [atom.lower() for atom in atoms]

    # Group by atom type
    atom_rows = dict()
    for i, atom in enumerate(atoms):
        atom_rows.setdefault(atom, list()).append(i)

    se_atoms = dict()
    for atom, row_inds in atom_rows.items():
        per_atom = handle_rows(atoms, dist_mat, row_inds)
        se_atoms.update({
            f"{atom}{i}": inds for i, inds in enumerate(per_atom)
        })
    return se_atoms
