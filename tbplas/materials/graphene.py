"""
Utilities for constructing graphene samples.

Functions
---------
    make_graphene_diamond: user function
        make graphene primitive cell in diamond shape
    make_graphene_rect: user function
        make graphene primitive cell in rectangular shape
"""

import math

import numpy as np

from ..builder import gen_lattice_vectors, PrimitiveCell, reshape_prim_cell, NM


def make_graphene_diamond(c=1.0):
    """
    Make graphene primitive cell in diamond shape.

    :param c: float
        length of c-axis in NANOMETER
    :return: cell: instance of 'PrimitiveCell' class
        graphene primitive cell
    """
    vectors = gen_lattice_vectors(a=0.246, b=0.246, c=c, gamma=60)
    cell = PrimitiveCell(vectors, unit=NM)
    cell.add_orbital([0.0, 0.0], label="C_pz")
    cell.add_orbital([1/3., 1/3.], label="C_pz")
    cell.add_hopping([0, 0], 0, 1, -2.7)
    cell.add_hopping([1, 0], 1, 0, -2.7)
    cell.add_hopping([0, 1], 1, 0, -2.7)
    return cell


def make_graphene_rect(from_scratch=True, c=1.0):
    """
    Make graphene primitive cell in rectangular shape.

    :param c: float
        length of c-axis in NANOMETER
    :param from_scratch: boolean
        method to build the primitive cell
        If true, build the cell from scratch.
        Otherwise, build it by reshaping a primitive cell.
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
        cell.add_hopping([0, 0], 0, 2, -2.7)
        cell.add_hopping([0, 0], 2, 3, -2.7)
        cell.add_hopping([0, 0], 3, 1, -2.7)
        cell.add_hopping([0, 1], 1, 0, -2.7)
        cell.add_hopping([1, 0], 3, 1, -2.7)
        cell.add_hopping([1, 0], 2, 0, -2.7)
    else:
        cell = make_graphene_diamond()
        lat_frac = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]])
        cell = reshape_prim_cell(cell, lat_frac)
    return cell
