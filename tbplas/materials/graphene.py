"""
Utilities for constructing graphene samples.

Functions
---------
    make_graphene_diamond: user function
        make graphene primitive cell in diamond shape
    make_graphene_rect: user function
        make graphene primitive cell in rectangular shape
    make_graphene_sp: user function
        make graphene primitive cell with 8-bands
"""

import math

import numpy as np

from ..builder import (gen_lattice_vectors, PrimitiveCell, reshape_prim_cell,
                       find_neighbors, SK, NM)


def make_graphene_diamond(c=1.0, t=-2.7):
    """
    Make graphene primitive cell in diamond shape.

    :param c: float
        length of c-axis in NANOMETER
    :param t: float
        hopping integral in eV
    :return: cell: instance of 'PrimitiveCell' class
        graphene primitive cell
    """
    vectors = gen_lattice_vectors(a=0.246, b=0.246, c=c, gamma=60)
    cell = PrimitiveCell(vectors, unit=NM)
    cell.add_orbital([0.0, 0.0], label="C_pz")
    cell.add_orbital([1/3., 1/3.], label="C_pz")
    cell.add_hopping([0, 0], 0, 1, t)
    cell.add_hopping([1, 0], 1, 0, t)
    cell.add_hopping([0, 1], 1, 0, t)
    return cell


def make_graphene_rect(from_scratch=True, c=1.0, t=-2.7):
    """
    Make graphene primitive cell in rectangular shape.

    :param c: float
        length of c-axis in NANOMETER
    :param from_scratch: boolean
        method to build the primitive cell
        If true, build the cell from scratch.
        Otherwise, build it by reshaping a primitive cell.
    :param t: float
        hopping integral in eV
    :return: cell: instance of 'PrimitiveCell' class
        graphene primitive cell
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
        cell.add_hopping([0, 0], 0, 2, t)
        cell.add_hopping([0, 0], 2, 3, t)
        cell.add_hopping([0, 0], 3, 1, t)
        cell.add_hopping([0, 1], 1, 0, t)
        cell.add_hopping([1, 0], 3, 1, t)
        cell.add_hopping([1, 0], 2, 0, t)
    else:
        cell = make_graphene_diamond(t=t)
        lat_frac = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]])
        cell = reshape_prim_cell(cell, lat_frac)
    return cell


def make_graphene_sp(c=1.0):
    """
    Make graphene primitive cell in rectangular shape.

    Reference:
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.82.245412

    :param c: float
        length of c-axis in NANOMETER
    :return: cell: instance of 'PrimitiveCell' class
        graphene primitive cell
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
