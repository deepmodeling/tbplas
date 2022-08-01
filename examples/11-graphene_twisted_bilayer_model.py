#! /usr/bin/env python

import math

import numpy as np
from numpy.linalg import norm
from scipy.spatial import KDTree

import tbplas as tb


def calc_twist_angle(i):
    """
    Calculate twisting angle according to ref. [1].

    :param i: integer
        parameter controlling the twisting angle
    :return: float
        twisting angle in RADIANs, NOT degrees
    """
    cos_ang = (3 * i ** 2 + 3 * i + 0.5) / (3 * i ** 2 + 3 * i + 1)
    return math.acos(cos_ang)


def calc_twist_angle2(n, m):
    """
    Calculate twisting angle according to ref. [2].

    :param n: integer
        parameter controlling the twisting angle
    :param m: integer
        parameter controlling the twisting angle
    :return: float
        twisting angle in RADIANs, NOT degrees
    """
    cos_ang = (n**2 + 4 * n * m + m**2) / (2 * (n**2 + n * m + m**2))
    return math.acos(cos_ang)


def calc_hetero_lattice(i, prim_cell_fixed: tb.PrimitiveCell):
    """
    Calculate Cartesian coordinates of lattice vectors of hetero-structure
    according to ref. [1].

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


def calc_hetero_lattice2(n, m, prim_cell_fixed: tb.PrimitiveCell):
    """
    Calculate Cartesian coordinates of lattice vectors of hetero-structure
    according to ref. [2].

    :param n: integer
        parameter controlling the twisting angle
    :param m: integer
        parameter controlling the twisting angle
    :param prim_cell_fixed: instance of 'PrimitiveCell' class
        primitive cell of fixed layer
    :return: hetero_lattice: (3, 3) float64 array
        Cartesian coordinates of hetero-structure lattice vectors in NANOMETER
    """
    hetero_lattice = np.array([[n, m, 0],
                               [-m, n + m, 0],
                               [0, 0, 1]])
    hetero_lattice = tb.frac2cart(prim_cell_fixed.lat_vec, hetero_lattice)
    return hetero_lattice


def calc_hop(rij: np.ndarray):
    """
    Calculate hopping parameter according to Slater-Koster relation.
    See ref. [2] for the formulae.

    :param rij: (3,) array
        displacement vector between two orbitals in NM
    :return: hop: float
        hopping parameter
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


def extend_hop(prim_cell: tb.PrimitiveCell, max_distance=0.75):
    """
    Extend the hopping terms in primitive cell up to cutoff distance.

    :param prim_cell: tb.PrimitiveCell
        primitive cell to extend
    :param max_distance: cutoff distance in NM
    :return: None. Incoming primitive cell is modified
    """
    prim_cell.sync_array()
    pos_r0 = prim_cell.orb_pos_nm
    tree_r0 = KDTree(pos_r0)
    neighbors = [(ia, ib, 0) for ia in range(-1, 2) for ib in range(-1, 2)]
    for rn in neighbors:
        pos_rn = pos_r0 + np.matmul(rn, prim_cell.lat_vec)
        tree_rn = KDTree(pos_rn)
        dist_matrix = tree_r0.sparse_distance_matrix(tree_rn,
                                                     max_distance=max_distance)
        for index, distance in dist_matrix.items():
            if distance > 0.0:
                rij = pos_rn[index[1]] - pos_r0[index[0]]
                prim_cell.add_hopping(rn, index[0], index[1], calc_hop(rij))


def main():
    # In this tutorial we show how to build twisted bilayer graphene. Firstly, we need
    # to define the functions for evaluating twisting angle and coordinates of lattice
    # vectors. See the following papers for the formulae:
    #
    # [1] https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.256802
    # [2] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.86.125413
    #
    # See function 'calc_twist_angle', 'calc_twist_angle2', 'calc_hetero_lattice' and
    # 'calc_hetero_lattice2' for the implementation.

    # Evaluate twisting angle.
    i = 5
    angle = calc_twist_angle(i)

    # To build a twist bilayer graphene we build the twisted primitive cells for
    # each layer first. The 'fixed' cell which is fixed at z=0 and not rotated
    # can be imported from the material repository directly.
    prim_cell_fixed = tb.make_graphene_diamond()

    # On the contrary, the 'twisted' cell must be rotated counter-clockwise by
    # the twisting angle and shifted towards +z by 0.3349 nanometers, which is
    # done by calling the function 'spiral_prim_cell'.
    prim_cell_twisted = tb.make_graphene_diamond()
    tb.spiral_prim_cell(prim_cell_twisted, angle=angle, shift=0.3349)

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
        # We utilize KDTree from scipy to detect interlayer neighbours up to
        # cutoff distance.
        inter_hop = tb.PCInterHopping(layer_fixed, layer_twisted)
        tree_fixed = KDTree(pos_fixed)
        neighbors = [(ia, ib, 0) for ia in range(-1, 2) for ib in range(-1, 2)]
        for rn in neighbors:
            # Get Cartesian coordinates of orbitals Rn cell of twisted layer
            pos_rn = pos_twisted + np.matmul(rn, layer_twisted.lat_vec)

            # Get the distance matrix between fixed and twisted layers
            tree_rn = KDTree(pos_rn)
            dist_matrix = tree_fixed.sparse_distance_matrix(tree_rn,
                                                            max_distance=0.75)

            # Add terms to inter_hop
            for index in dist_matrix.keys():
                rij = pos_rn[index[1]] - pos_fixed[index[0]]
                inter_hop.add_hopping(rn, index[0], index[1], calc_hop(rij))

        # Then we need to extend the hopping terms in each layer up to cutoff
        # distance by calling 'extend_intra_hop'.
        extend_hop(layer_fixed, max_distance=0.75)
        extend_hop(layer_twisted, max_distance=0.75)

        # Finally, we merge layers and inter_hop to yield a hetero-structure
        merged_cell = tb.merge_prim_cell(layer_fixed, layer_twisted, inter_hop)

    # Evaluate band structure of hetero-structure
    k_points = np.array([
       [0.0, 0.0, 0.0],
       [2./3, 1./3, 0.0],
       [1./2, 0.0, 0.0],
       [0.0, 0.0, 0.0],
    ])
    k_label = ["G", "K", "M", "G"]
    k_path, k_idx = tb.gen_kpath(k_points, [10, 10, 10])
    k_len, bands = merged_cell.calc_bands(k_path)
    vis = tb.Visualizer()
    vis.plot_bands(k_len, bands, k_idx, k_label)

    # Visualize Moire's pattern
    angle = -math.atan(hetero_lattice[0, 1] / hetero_lattice[0, 0])
    tb.spiral_prim_cell(merged_cell, angle=angle)
    sample = tb.Sample(tb.SuperCell(merged_cell, dim=(4, 4, 1),
                                    pbc=(True, True, False)))
    sample.plot(with_orbitals=False, hop_as_arrows=False,
                hop_eng_cutoff=0.3)


if __name__ == "__main__":
    main()
