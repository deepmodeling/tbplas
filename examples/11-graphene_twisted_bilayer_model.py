#! /usr/bin/env python

import math

import numpy as np
from scipy.spatial import cKDTree

import tbplas as tb


def calc_twist_angle(i):
    """
    Calculate twisting angle.

    :param i: integer
        parameter controlling the twisting angle
    :return: float
        twisting angle in RADIANs, NOT degrees
    """
    cos_ang = (3 * i ** 2 + 3 * i + 0.5) / (3 * i ** 2 + 3 * i + 1)
    return math.acos(cos_ang)


def calc_hetero_lattice(i, prim_cell_fixed: tb.PrimitiveCell):
    """
    Calculate Cartesian coordinates of lattice vectors of hetero-structure.

    :param i: integer
        parameter controlling the twisting angle
    :param prim_cell_fixed: instance of 'PrimitiveCell' class
        primitive cell of fixed layer
    :return: hetero_lattice: (3, 3) float64 array
        Cartesian coordinates of hetero-structure lattice vectors in NANOMETER
    """
    hetero_lattice = np.array([[i, i + 1, 0],
                               [-(i + 1), 2 * i + 1, 0],
                               [0, 0, 1]])
    hetero_lattice = tb.frac2cart(prim_cell_fixed.lat_vec, hetero_lattice)
    return hetero_lattice


def main():
    # In this tutorial we show how to build twisted bilayer graphene. Firstly, we need
    # to define the functions for evaluating twisting angle and coordinates of lattice
    # vectors. See the following papers for the formulae:
    #
    # [1] https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.256802
    # [2] https://journals.aps.org/prb/pdf/10.1103/PhysRevB.81.161405
    #
    # We employ the formulae in ref [1] and implement them in functions
    # 'calc_twist_angle' and 'calc_hetero_lattice'.

    # Evaluate twisting angle.
    i = 5
    angle = calc_twist_angle(i)

    # To build a twist bilayer graphene we build the twisted primitive cells for
    # each layer first. The 'fixed' cell which is fixed at z=0 and not rotated
    # can be imported from the material repository directly.
    prim_cell_fixed = tb.make_graphene_diamond()

    # On the contrary, the 'twisted' cell must be rotated counter-clockwise by
    # the twisting angle and shifted towards +z by 0.5 nanometers, which is
    # done by calling the function 'spiral_prim_cell'.
    prim_cell_twisted = tb.make_graphene_diamond()
    tb.spiral_prim_cell(prim_cell_twisted, angle=angle, shift=0.5)

    # Calculate coordinates of lattice vectors.
    # The reference paper gives the fractional lattice vectors in basis of
    # 'fix' primitive cell. However, we want the Cartesian coordinates. This
    # is done by calling the 'calc_hetero_lattice' function.
    hetero_lattice = calc_hetero_lattice(i, prim_cell_fixed)

    # With all the primitive cells ready, we build the 'fixed' and 'twisted'
    # layers by reshaping corresponding cells to the lattice vectors of
    # hetero-structure. This is done by calling 'make_hetero_layer'.
    layer_fixed = tb.make_hetero_layer(prim_cell_fixed, hetero_lattice)
    layer_twisted = tb.make_hetero_layer(prim_cell_twisted, hetero_lattice)

    # Now we come to the most difficult part: construction of interlayer
    # hopping terms. Note that this procedure is strongly system dependent,
    # and the example shown here is just for demonstration purpose without
    # any actual physics.

    # Interlayer hopping terms are position-dependent. So we need to get
    # the orbital positions of 'fixed' and 'twisted' layers first.
    layer_fixed.sync_array()
    pos_fixed = layer_fixed.orb_pos_nm
    layer_twisted.sync_array()
    pos_twisted = layer_twisted.orb_pos_nm

    # Loop over neighbouring cells to build inter-hopping dictionary.
    # We only need to take the hopping terms from (0, 0, 0) cell of 'fixed'
    # layer to any cell of 'twisted' layer. The conjugate terms are handled
    # automatically.
    # We utilize KDTree from scipy to detect interlayer neighbours up to cutoff
    # distance.
    inter_hop = tb.InterHopDict(layer_fixed, layer_twisted)
    tree_fixed = cKDTree(pos_fixed)
    for ia in range(-1, 2):
        for ib in range(-1, 2):
            rn = (ia, ib, 0)
            # Get Cartesian coordinates of orbitals Rn cell of twisted layer
            pos_rn = pos_twisted + np.matmul(rn, layer_twisted.lat_vec)

            # Get the distance matrix between fixed and twisted layers
            tree_rn = cKDTree(pos_rn)
            dist_matrix = tree_fixed.sparse_distance_matrix(tree_rn,
                                                            max_distance=0.55)

            # Add terms to inter_hop
            # We assume a simple Gaussian decay of hopping energies w.r.t
            # 0.246 NANOMETER.
            for index, distance in dist_matrix.items():
                inter_hop.add_hopping(rn, index[0], index[1],
                                      -2.7 * math.exp(-(distance - 0.246) ** 2))

    # Finally, we merge layers and inter_hop to yield a hetero-structure
    merged_cell = tb.merge_prim_cell(layer_fixed, layer_twisted, inter_hop)

    sample = tb.Sample(tb.SuperCell(merged_cell, dim=(3, 3, 1),
                                    pbc=(False, False, False)))
    sample.plot(with_orbitals=False, hop_as_arrows=False)


if __name__ == "__main__":
    main()
