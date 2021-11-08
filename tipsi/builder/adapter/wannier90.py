"""
Functions and classes for converting tight-binding models from other packages.

Functions
--------
    read_lat_vec: developer function
        read lattice vectors from seed_name.win
    read_orb_pos: developer function
        read orbital positions from seed_name_centres.xyz
    read_hop: developer function
        read hopping terms from seed_name_hr.dat
    read_wsvec: developer function
        read correction terms from seed_name_wsvec.dat
    apply_wsvec: developer function
        correct hopping terms using data in from read_wsvec
    wan2pc: user function
        create primitive cell from output of Wannier90
"""

import os

import numpy as np

from ..constants import BOHR2ANG
from ..lattice import cart2frac
from .. import core
from ..primitive import Orbital, Hopping, PrimitiveCell


def read_lat_vec(seed_name="wannier90"):
    """
    Read lattice vectors from seed_name.win.

    :param seed_name: string
        seed_name of Wannier90 output files
    :return: lat_vec: (3, 3) float64 array
        lattice vectors in Angstrom
    :raises ValueError: if units are not "Ang" or "Bohr"
    """
    with open(f"{seed_name}.win", "r") as win_file:
        win_content = win_file.readlines()

    # Locate lattice vectors
    nl_start, nl_end = 0, 0
    for nl, line in enumerate(win_content):
        if line.lstrip().startswith("begin unit_cell_cart"):
            nl_start = nl
        elif line.lstrip().startswith("end unit_cell_cart"):
            nl_end = nl
    lat_block = win_content[nl_start:nl_end+1]

    # Parse lattice vectors
    if len(lat_block) == 6:
        units = lat_block[1].lstrip().rstrip()
        lat_vec_raw = lat_block[2:-1]
    else:
        units = "Ang"
        lat_vec_raw = lat_block[1:-1]
    lat_vec = np.array([[float(x) for x in line.split()]
                        for line in lat_vec_raw], dtype=np.float64)

    # Unit conversion
    if units in ("Ang", "ang"):
        pass
    elif units in ("Bohr", "bohr"):
        lat_vec *= BOHR2ANG
    else:
        raise ValueError(f"Illegal lattice vector unit '{units}'")
    return lat_vec


def read_orb_pos(seed_name="wannier90"):
    """
    Read orbital positions from seed_name_centres.xyz.

    :param seed_name: string
        seed_name of Wannier90 output files
    :return: orb_pos: (num_wan, 3) float64 array
        Cartesian coordinates of orbitals in Angstrom
    """
    # Reference: https://en.wikipedia.org/wiki/XYZ_file_format
    with open(f"{seed_name}_centres.xyz", "r") as centre_file:
        centre_content = centre_file.readlines()
    orb_pos = []
    for line in centre_content[2:]:
        data = line.split()
        if data[0] == 'X':
            orb_pos.append([float(x) for x in data[1:]])
    orb_coord = np.array(orb_pos, dtype=np.float64)
    return orb_coord


def read_hop(seed_name="wannier90"):
    """
    Read hopping terms from seed_name_hr.dat.

    :param seed_name: string
        seed_name of Wannier90 output files
    :return: hop_ind: list of (ra, rb, rc, orb_i, orb_j)
        hopping indices
    :return: hop_eng: list of complex numbers
        hopping energies in eV
    """
    with open(f"{seed_name}_hr.dat", "r") as hr_file:
        hr_content = hr_file.readlines()
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
    return hop_ind, hop_eng


def read_wsvec(seed_name="wannier90"):
    """
    Read correction terms from seed_name_wsvec.dat.

    :param seed_name: string
        seed_name of Wannier90 output files
    :return: wsvec: dictionary
        correction terms to vector R
        Keys should be (ra, rb, rc, orb_i, orb_j).
        Values should be (delta_ra, delta_rb, delta_rc).
    """
    wsvec = dict()
    with open(f"{seed_name}_wsvec.dat", "r") as iterator:
        next(iterator)  # skip comment line
        for first_line in iterator:
            data = first_line.split()
            ra = int(data[0])
            rb = int(data[1])
            rc = int(data[2])
            orb_i = int(data[3]) - 1
            orb_j = int(data[4]) - 1
            num_vec = int(next(iterator))
            sites_cor = []
            for i in range(num_vec):
                data = next(iterator).split()
                delta_ra = int(data[0])
                delta_rb = int(data[1])
                delta_rc = int(data[2])
                sites_cor.append((delta_ra, delta_rb, delta_rc))
            wsvec[(ra, rb, rc, orb_i, orb_j)] = sites_cor
    return wsvec


def apply_wsvec(wsvec, hop_ind, hop_eng):
    """
    Correct hopping terms using data from read_wsvec

    :param wsvec: dictionary
        correction terms to vector R
    :param hop_ind: list of (ra, rb, rc, orb_i, orb_j)
        hopping indices
    :param hop_eng: list of complex numbers
        hopping energies in eV
    :return: None
        hop_ind and hop_eng are modified.
    """
    for ind_0 in wsvec.keys():
        # Back up and reset original hopping term
        ih_0 = hop_ind.index(ind_0)
        eng_bak = hop_eng[ih_0]
        hop_eng[ih_0] = 0.0

        # Apply correction vector
        num_vec = len(wsvec[ind_0])
        for vec in wsvec[ind_0]:
            ind_1 = (ind_0[0] + vec[0], ind_0[1] + vec[1], ind_0[2] + vec[2],
                     ind_0[3], ind_0[4])
            if ind_1 in hop_ind:
                ih_1 = hop_ind.index(ind_1)
                hop_eng[ih_1] = eng_bak / num_vec
            else:
                hop_ind.append(ind_1)
                hop_eng.append(eng_bak / num_vec)


def wan2pc(seed_name="wannier90", correct_hop=False, eng_cutoff=1.0e-5):
    """
    Create primitive cell from output of Wannier90.

    :param seed_name: string
        seed_name of Wannier90 output files
    :param correct_hop: boolean
        whether to correct hopping terms using data in seed_name_wsvec.dat
    :param eng_cutoff: float
        energy cutoff for hopping terms in eV
        Hopping terms with energy below this threshold will be dropped.
    :return: prim_cell: instance of 'PrimitiveCell' class
        primitive cell created from Wannier90 output files
    :raise ValueError: if unit of lattice vectors is not "Ang" or "Bohr"
    :raise FileNotFoundError: if seed_name_wsvec.dat is not found
    """
    # Read Wannier90 output
    lat_vec = read_lat_vec(seed_name)
    orb_pos = read_orb_pos(seed_name)
    hop_ind, hop_eng = read_hop(seed_name)

    # Convert orbital positions from Cartesian to fractional
    orb_pos = cart2frac(lat_vec, orb_pos)

    # Read and apply correction to hopping terms
    if correct_hop:
        if os.path.exists(f"{seed_name}_wsvec.dat"):
            wsvec = read_wsvec(seed_name)
            apply_wsvec(wsvec, hop_ind, hop_eng)
        else:
            raise FileNotFoundError(f"{seed_name}_wsvec.dat not found")

    # Reduce hopping terms
    hop_ind = np.array(hop_ind, dtype=np.int32)
    hop_eng = np.array(hop_eng, dtype=np.complex128)
    orb_eng, hop_ind, hop_eng = core.reduce_hop(hop_ind, hop_eng, eng_cutoff,
                                                orb_pos.shape[0])

    # Create the primitive cell
    # Here we manipulate the attributes of prim_cell directly
    # since it is much faster.
    prim_cell = PrimitiveCell(lat_vec)
    for i_o, pos in enumerate(orb_pos):
        orbital = Orbital(tuple(pos), orb_eng.item(i_o))
        prim_cell.orbital_list.append(orbital)
    for i_h, hop in enumerate(hop_ind):
        ind, orb_i, orb_j = hop[:3], hop.item(3), hop.item(4)
        hopping = Hopping(tuple(ind), orb_i, orb_j, hop_eng.item(i_h))
        prim_cell.hopping_list.append(hopping)
    prim_cell.sync_array(force_sync=True)
    return prim_cell
