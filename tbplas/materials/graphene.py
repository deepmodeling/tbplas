"""Utilities for constructing graphene primitive cells."""

import math

import numpy as np

from ..builder import (gen_lattice_vectors, PrimitiveCell, reshape_prim_cell,
                       find_neighbors, SK, NM, HopDict)


__all__ = ["make_graphene_rect", "make_graphene_diamond", "make_graphene_sp",
           "make_graphene_soc"]


def make_graphene_diamond(c: float = 1.0, t: complex = -2.7) -> PrimitiveCell:
    """
    Make graphene primitive cell in diamond shape.

    :param c: length of c-axis in NANOMETER
    :param t: hopping integral in eV
    :return: graphene primitive cell
    """
    vectors = gen_lattice_vectors(a=0.246, b=0.246, c=c, gamma=60)
    cell = PrimitiveCell(vectors, unit=NM)
    cell.add_orbital((0.0, 0.0), label="C_pz")
    cell.add_orbital((1/3., 1/3.), label="C_pz")
    cell.add_hopping((0, 0), 0, 1, t)
    cell.add_hopping((1, 0), 1, 0, t)
    cell.add_hopping((0, 1), 1, 0, t)
    return cell


def make_graphene_rect(from_scratch: bool = True,
                       c: float = 1.0,
                       t: complex = -2.7) -> PrimitiveCell:
    """
    Make graphene primitive cell in rectangular shape.

    :param c: length of c-axis in NANOMETER
    :param from_scratch: method to build the primitive cell
        If true, build the cell from scratch.
        Otherwise, build it by reshaping a primitive cell.
    :param t: hopping integral in eV
    :return: graphene primitive cell
    """
    if from_scratch:
        # Calculate lattice vectors
        sqrt3 = math.sqrt(3)
        a = 0.246
        cc_bond = sqrt3 / 3 * a
        vectors = gen_lattice_vectors(sqrt3 * cc_bond, 3 * cc_bond, c)

        # Create cell and add orbitals
        cell = PrimitiveCell(vectors, unit=NM)
        cell.add_orbital((0, 0), label="C_pz")
        cell.add_orbital((0, 2/3.), label="C_pz")
        cell.add_orbital((1/2., 1/6.), label="C_pz")
        cell.add_orbital((1/2., 1/2.), label="C_pz")

        # Add hopping terms
        cell.add_hopping((0, 0), 0, 2, t)
        cell.add_hopping((0, 0), 2, 3, t)
        cell.add_hopping((0, 0), 3, 1, t)
        cell.add_hopping((0, 1), 1, 0, t)
        cell.add_hopping((1, 0), 3, 1, t)
        cell.add_hopping((1, 0), 2, 0, t)
    else:
        cell = make_graphene_diamond(t=t)
        lat_frac = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]])
        cell = reshape_prim_cell(cell, lat_frac)
    return cell


def make_graphene_sp(c: float = 1.0) -> PrimitiveCell:
    """
    Make graphene primitive cell in rectangular shape.

    Reference:
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.82.245412

    :param c: length of c-axis in NANOMETER
    :return: graphene primitive cell
    """
    # Lattice vectors and orbital info.
    vectors = gen_lattice_vectors(a=0.246, b=0.246, c=c, gamma=60)
    orb_pos = np.array([[0.0, 0.0, 0.0], [1/3., 1/3., 0.0]])
    orb_label = ("s", "px", "py", "pz")
    orb_eng = {"s": -8.370, "px": 0.0, "py": 0.0, "pz": 0.0}

    # Create cell and add orbitals
    cell = PrimitiveCell(vectors, unit=NM)
    for pos in orb_pos:
        for label in orb_label:
            cell.add_orbital(pos, energy=orb_eng[label], label=label)

    # Add hopping terms
    neighbors = find_neighbors(cell, a_max=5, b_max=5, max_distance=0.25)
    d1 = 0.14202816622064793  # 1st nearest hopping distance in NM
    d2 = 0.24599999999999997  # 2nd neatest hopping distance in NM
    sk = SK()
    for term in neighbors:
        # Extract info.
        i, j = term.pair
        label_i = cell.get_orbital(i).label
        label_j = cell.get_orbital(j).label
        distance = term.distance
        rij = term.rij

        # Evaluate hopping integral using Slater-Koster formulation
        if abs(distance - d1) < 1.0e-5:
            v_sss, v_sps, v_pps, v_ppp = -5.729, 5.618, 6.050, -3.070
        elif abs(distance - d2) < 1.0e-5:
            v_sss, v_sps, v_pps, v_ppp = 0.102, -0.171, -0.377, 0.070
        else:
            raise ValueError(f"Too large distance {distance}")
        hop = sk.eval(rij, label_i=label_i, label_j=label_j,
                      v_sss=v_sss, v_sps=v_sps, v_pps=v_pps, v_ppp=v_ppp)

        # Add hopping term
        cell.add_hopping(term.rn, i, j, hop)
    return cell


def make_graphene_soc(is_qsh: bool = True) -> PrimitiveCell:
    """
    Set up the primitive cell of graphene with Rashba and Kane-Mele SOC.

    NOTE: This piece of code is adapted from a legacy program and uses the
    'HopDict' class for holding the hopping terms for compatibility reasons.
    It is not recommended to use this class any longer.

    :param is_qsh: whether is the model is in quantum spin Hall phase
    :return: the primitive cell of graphene
    """
    # Parameters
    lat = 1.
    t = -1.
    lamb_so = 0.06 * t
    lamb_r = 0.05 * t
    if is_qsh:
        lamb_nu = 0.1 * t  # QSH phase
    else:
        lamb_nu = 0.4 * t  # normal insulating phase

    # Lattice
    vectors = np.array([[0.5 * lat * math.sqrt(3), -0.5 * lat, 0.],
                        [0.5 * lat * math.sqrt(3), 0.5 * lat, 0.],
                        [0, 0, 1]])
    prim_cell = PrimitiveCell(vectors)

    # Add orbitals
    prim_cell.add_orbital((0, 0), lamb_nu)
    prim_cell.add_orbital((0, 0), lamb_nu)
    prim_cell.add_orbital((1./3, 1./3), -lamb_nu)
    prim_cell.add_orbital((1./3, 1./3), -lamb_nu)

    # Add hopping terms
    def _hop_1st(vec):
        a = vec[1] + 1j * vec[0]
        ac = vec[1] - 1j * vec[0]
        return np.array([[t, 1j * a * lamb_r],
                         [1j * ac * lamb_r, t]])

    def _hop_2nd(vec0, vec1):
        b = 2. / math.sqrt(3.) * np.cross(vec0, vec1)
        return np.array([[1j * b[2] * lamb_so, 0.],
                         [0., -1j * b[2] * lamb_so]])

    # Carbon-carbon vectors
    ac_vec = np.array([[1., 0., 0.],
                       [-0.5, np.sqrt(3.) / 2., 0.],
                       [-0.5, -np.sqrt(3.) / 2., 0.]])
    bc_vec = np.array([[-1., 0., 0.],
                       [0.5, -np.sqrt(3.) / 2., 0.],
                       [0.5, np.sqrt(3.) / 2., 0.]])

    # Initialize hop_dict
    hop_dict = HopDict(4)

    # 1st nearest neighbours
    # (0, 0, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 2:4] = _hop_1st(ac_vec[0])
    hop_dict.set_mat((0, 0, 0), hop_mat)

    # (-1, 0, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 2:4] = _hop_1st(ac_vec[1])
    hop_dict.set_mat((-1, 0, 0), hop_mat)

    # (0, -1, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 2:4] = _hop_1st(ac_vec[2])
    hop_dict.set_mat((0, -1, 0), hop_mat)

    # 2nd nearest neighbours
    # (0, 1, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 0:2] = _hop_2nd(ac_vec[0], bc_vec[2])
    hop_mat[2:4, 2:4] = _hop_2nd(bc_vec[2], ac_vec[0])
    hop_dict.set_mat((0, 1, 0), hop_mat)

    # (1, 0, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 0:2] = _hop_2nd(ac_vec[0], bc_vec[1])
    hop_mat[2:4, 2:4] = _hop_2nd(bc_vec[1], ac_vec[0])
    hop_dict.set_mat((1, 0, 0), hop_mat)

    # (1, -1, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 0:2] = _hop_2nd(ac_vec[2], bc_vec[1])
    hop_mat[2:4, 2:4] = _hop_2nd(bc_vec[1], ac_vec[2])
    hop_dict.set_mat((1, -1, 0), hop_mat)

    # Apply hopping terms
    prim_cell.add_hopping_dict(hop_dict)
    return prim_cell
