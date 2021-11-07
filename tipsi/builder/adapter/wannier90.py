"""
Functions and classes for converting tight-binding models from other packages.

Functions
--------
    wan2pc: user function
        create primitive cell from output of Wannier90
"""

import numpy as np

from ..constants import BOHR2ANG
from ..lattice import cart2frac
from .. import core
from ..primitive import PrimitiveCell


def wan2pc(seed_name="wannier90", eng_cutoff=1.0e-5):
    """
    Create primitive cell from output of Wannier90.

    :param seed_name: string
        seed_name of Wannier90 output files
    :param eng_cutoff: float
        energy cutoff for hopping terms in eV
        Hopping terms with energy below this threshold will be dropped.
    :return: prim_cell: instance of 'PrimitiveCell' class
        primitive cell created from Wannier90 output files
    :raise ValueError: if units of lattice vectors is not "Ang" or "Bohr"
    """
    # Parse lattice vectors
    with open(f"{seed_name}.win", "r") as win_file:
        win_content = win_file.readlines()

    # Locate lattice vectors
    nl_start, nl_end = 0, 0
    for nl, line in win_content:
        if line.lstrip().startswith("begin unit_cell_cart"):
            nl_start = nl
        elif line.lstrip().startswith("end unit_cell_cart"):
            nl_end = nl
    lat_block = win_content[nl_start:nl_end+1]

    # Check if there are units contained in lattice block
    if len(lat_block) == 6:
        units = lat_block[1].lstrip().rstrip()
        lat_vec_raw = lat_block[2:-1]
    else:
        units = "Ang"
        lat_vec_raw = lat_block[0:-1]
    lat_vec = np.array([[float(x) for x in line.split()]
                        for line in lat_vec_raw], dtype=np.float64)
    if units in ("Ang", "ang"):
        pass
    elif units in ("Bohr", "bohr"):
        lat_vec *= BOHR2ANG
    else:
        raise ValueError(f"Illegal lattice vector unit '{units}'")

    # Parse wannier centers
    # Reference: https://en.wikipedia.org/wiki/XYZ_file_format
    with open(f"{seed_name}_centres.xyz", "r") as centre_file:
        centre_content = centre_file.readlines()
    orb_coord = []
    for line in centre_content[2:]:
        data = line.split()
        if data[0] == 'X':
            orb_coord.append([float(x) for x in data[1:]])
    orb_coord = np.array(orb_coord, dtype=np.float64)
    orb_coord = cart2frac(lat_vec, orb_coord)

    # Parse hopping terms
    with open(f"{seed_name}_hr.dat", "r") as hr_file:
        hr_content = hr_file.readlines()
    num_wan = int(hr_content[1])
    nl_hopping = int(hr_content[2])
    nl_skip = 3 + int(np.ceil(nl_hopping / 15))
    hop_ind, hop_eng = [], []
    for line in hr_content[nl_skip:]:
        data = line.split()
        ind = (int(data[0]), int(data[1]), int(data[2]),
               int(data[3])-1, int(data[4])-1)
        eng = float(data[5]) + 1j * float(data[6])
        hop_ind.append(ind)
        hop_eng.append(eng)
    hop_ind = np.array(hop_ind, dtype=np.int32)
    hop_eng = np.array(hop_eng, dtype=np.complex128)

    # Extract on-site energies and reduce hopping terms
    orb_eng, hop_ind, hop_eng = core.reduce_hop(hop_ind, hop_eng,
                                                eng_cutoff, num_wan)

    # Create the primitive cell
    prim_cell = PrimitiveCell(lat_vec)
    for i_o, coord in enumerate(orb_coord):
        prim_cell.add_orbital(tuple(coord), orb_eng.item(i_o))
    for i_h, hop in enumerate(hop_ind):
        ind, orb_i, orb_j = hop[:3], hop.item(3), hop.item(4)
        prim_cell.add_hopping(tuple(ind), orb_i, orb_j, hop_eng.item(i_h))
    return prim_cell
