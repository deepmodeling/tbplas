#! /usr/bin/env python
"""
Example for constructing twisted bilayer graphene.

References:
[1] https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.256802
[2] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.86.125413
"""

import math
from typing import Tuple

import numpy as np
from numpy.linalg import norm

import tbplas as tb


def calc_twist_angle(i: int) -> float:
    """
    Calculate twisting angle according to ref. [1].

    :param i: parameter controlling the twisting angle
    :return: twisting angle in RADIANs, NOT degrees
    """
    cos_ang = (3 * i ** 2 + 3 * i + 0.5) / (3 * i ** 2 + 3 * i + 1)
    return math.acos(cos_ang)


def calc_twist_angle2(n: int, m: int) -> float:
    """
    Calculate twisting angle according to ref. [2].

    :param n: parameter controlling the twisting angle
    :param m: parameter controlling the twisting angle
    :return: twisting angle in RADIANs, NOT degrees
    """
    cos_ang = (n**2 + 4 * n * m + m**2) / (2 * (n**2 + n * m + m**2))
    return math.acos(cos_ang)


def calc_hetero_lattice(i: int, prim_cell_fixed: tb.PrimitiveCell) -> np.ndarray:
    """
    Calculate Cartesian coordinates of lattice vectors of hetero-structure
    according to ref. [1].

    :param i: parameter controlling the twisting angle
    :param prim_cell_fixed: primitive cell of fixed layer
    :return: (3, 3) float64 array, Cartesian coordinates of hetero-structure
        lattice vectors in NANOMETER
    """
    hetero_lattice = np.array([[i, i + 1, 0],
                               [-(i + 1), 2 * i + 1, 0],
                               [0, 0, 1]])
    hetero_lattice = tb.frac2cart(prim_cell_fixed.lat_vec, hetero_lattice)
    return hetero_lattice


def calc_hetero_lattice2(n: int, m: int,
                         prim_cell_fixed: tb.PrimitiveCell) -> np.ndarray:
    """
    Calculate Cartesian coordinates of lattice vectors of hetero-structure
    according to ref. [2].

    :param n: parameter controlling the twisting angle
    :param m: parameter controlling the twisting angle
    :param prim_cell_fixed: primitive cell of fixed layer
    :return: (3, 3) float64 array, Cartesian coordinates of hetero-structure
        lattice vectors in NANOMETER
    """
    hetero_lattice = np.array([[n, m, 0],
                               [-m, n + m, 0],
                               [0, 0, 1]])
    hetero_lattice = tb.frac2cart(prim_cell_fixed.lat_vec, hetero_lattice)
    return hetero_lattice


def calc_hop(rij: np.ndarray) -> float:
    """
    Calculate hopping parameter according to Slater-Koster relation.
    See ref. [2] for the formulae.

    :param rij: (3,) array, displacement vector between two orbitals in NM
    :return: hopping parameter in eV
    """
    a0 = 0.1418
    a1 = 0.3349
    r_c = 0.6140
    l_c = 0.0265
    gamma0 = 2.7
    gamma1 = 0.48
    decay = 22.18
    q_pi = decay * a0
    q_sigma = decay * a1
    dr = norm(rij).item()
    n = rij.item(2) / dr
    v_pp_pi = - gamma0 * math.exp(q_pi * (1 - dr / a0))
    v_pp_sigma = gamma1 * math.exp(q_sigma * (1 - dr / a1))
    fc = 1 / (1 + math.exp((dr - r_c) / l_c))
    hop = (n**2 * v_pp_sigma + (1 - n**2) * v_pp_pi) * fc
    return hop


def extend_hop(prim_cell: tb.PrimitiveCell, max_distance: float = 0.75) -> None:
    """
    Extend the hopping terms in primitive cell up to cutoff distance.

    :param prim_cell: primitive cell to extend
    :param max_distance: cutoff distance in NM
    :return: None. Incoming primitive cell is modified
    """
    neighbors = tb.find_neighbors(prim_cell, a_max=1, b_max=1,
                                  max_distance=max_distance)
    for term in neighbors:
        i, j = term.pair
        prim_cell.add_hopping(term.rn, i, j, calc_hop(term.rij))


def make_layers(i: int = 1, shift: float = 0.3349) -> Tuple[tb.PrimitiveCell, tb.PrimitiveCell]:
    """
    Prepare the layers for assembling the hetero-structure.

    :param i: integer controlling the twisting angle
    :param shift: interlayer distance in nm
    :return: fixed and twisted layers of the hetero-structure
    """
    # Evaluate twisting angle.
    angle = calc_twist_angle(i)

    # To build a twist bilayer graphene we build the twisted primitive cells for
    # each layer first. The 'fixed' cell which is fixed at z=0 and not rotated
    # can be imported from the material repository directly.
    prim_cell_fixed = tb.make_graphene_diamond()

    # On the contrary, the 'twisted' cell must be rotated counter-clockwise by
    # the twisting angle and shifted towards +z by 0.3349 nanometers, which is
    # done by calling the function 'spiral_prim_cell'.
    prim_cell_twisted = tb.make_graphene_diamond()
    tb.spiral_prim_cell(prim_cell_twisted, angle=angle, shift=shift)

    # Evaluate coordinates of lattice vectors of hetero-structure.
    # The reference papers give the fractional coordinates in basis of 'fixed'
    # primitive cell. However, we want the Cartesian coordinates. This is done
    # by calling the 'calc_hetero_lattice' function.
    hetero_lattice = calc_hetero_lattice(i, prim_cell_fixed)

    # With all the primitive cells ready, we build the 'fixed' and 'twisted'
    # layers by reshaping corresponding cells to the lattice vectors of
    # hetero-structure. This is done by calling 'make_hetero_layer'.
    layer_fixed = tb.make_hetero_layer(prim_cell_fixed, hetero_lattice)
    layer_twisted = tb.make_hetero_layer(prim_cell_twisted, hetero_lattice)

    # Align the layers for better appearance though it is not essential.
    angle = -math.atan(hetero_lattice[0, 1] / hetero_lattice[0, 0])
    tb.spiral_prim_cell(layer_fixed, angle=angle)
    tb.spiral_prim_cell(layer_twisted, angle=angle)
    return layer_fixed, layer_twisted


def make_tbg_pc():
    """Set up tbg at PrimitiveCell level."""

    # Prepare the layers
    layer_fixed, layer_twisted = make_layers(i=5)

    # From now, we have two approaches to build the hetero-structure.
    # The first one is to merge the layers and then extend the hopping terms of
    # the whole cell.
    algo = 0
    if algo == 0:
        merged_cell = tb.merge_prim_cell(layer_fixed, layer_twisted)
        extend_hop(merged_cell, max_distance=0.75)

    # The second approach is complex, but more general. We build the inter-cell
    # hopping terms first, then extend the layers. Finally, we merge them to
    # yield the hetero-structure.
    else:
        # Find the hopping neighbors between 'fixed' and 'twisted' layers up to
        # cutoff distance. We only need to take the hopping terms from (0, 0, 0)
        # cell of 'fixed' layer to any cell of 'twisted' layer. The conjugate
        # terms are handled automatically.
        inter_hop = tb.PCInterHopping(layer_fixed, layer_twisted)
        neighbors = tb.find_neighbors(layer_fixed, layer_twisted,
                                      a_max=1, b_max=1, max_distance=0.75)
        for term in neighbors:
            i, j = term.pair
            inter_hop.add_hopping(term.rn, i, j, calc_hop(term.rij))

        # Then we need to extend the hopping terms in each layer up to cutoff
        # distance by calling 'extend_intra_hop'.
        extend_hop(layer_fixed, max_distance=0.75)
        extend_hop(layer_twisted, max_distance=0.75)

        # Finally, we merge layers and inter_hop to yield a hetero-structure
        merged_cell = tb.merge_prim_cell(layer_fixed, layer_twisted, inter_hop)

    # Visualize Moire's pattern
    sample = tb.Sample(tb.SuperCell(merged_cell, dim=(4, 4, 1),
                                    pbc=(True, True, False)))
    sample.plot(with_orbitals=False, hop_as_arrows=False,
                hop_eng_cutoff=0.3)


def make_tbg_sample():
    """Set up tbg at Sample level."""

    # Prepare the layers
    layer_fixed, layer_twisted = make_layers(i=5)

    # Make supercells
    sc_fixed = tb.SuperCell(layer_fixed, dim=(4, 4, 1),
                            pbc=(True, True, False))
    sc_twisted = tb.SuperCell(layer_twisted, dim=(4, 4, 1),
                              pbc=(True, True, False))

    # Build interlayer hopping terms
    inter_hop = tb.SCInterHopping(sc_fixed, sc_twisted)
    neighbors = tb.find_neighbors(sc_fixed, sc_twisted, a_max=1, b_max=1,
                                  max_distance=0.75)
    for term in neighbors:
        i, j = term.pair
        inter_hop.add_hopping(term.rn, i, j, calc_hop(term.rij))

    # Create the sample
    sample = tb.Sample(sc_fixed, sc_twisted, inter_hop)
    sample.plot(with_orbitals=False, hop_as_arrows=False,
                hop_eng_cutoff=0.3, sc_hop_colors=['r', 'b'], inter_hop_colors=['g'])


if __name__ == "__main__":
    make_tbg_pc()
    make_tbg_sample()
