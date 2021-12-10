#! /usr/bin/env python

import unittest
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


class TestHetero(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_inter_hop_dict(self):
        """
        Test InterHopDict class.

        :return: None.
        """
        pc_bra = tb.make_graphene_diamond()
        pc_ket = tb.make_graphene_diamond()

        # Check initial state of inter_hop
        inter_hop = tb.InterHopDict(pc_bra, pc_ket)
        self.assertDictEqual(inter_hop.dict, {})

        # Add the 1st hopping term
        rn = (0, 0, 0)
        inter_hop.add_hopping(rn, 2, 3, 1.0)
        self.assertTrue(rn in inter_hop.dict.keys())
        self.assertTrue((2, 3) in inter_hop.dict[rn].keys())

        # Add a new term to existing (0, 0, 0) cell
        inter_hop.add_hopping(rn, 0, 1, 2.0)
        self.assertTrue((0, 1) in inter_hop.dict[rn].keys())

        # Add a new term with new cell index
        rn = (2, 1, 0)
        inter_hop.add_hopping(rn, 2, 3, 1.5)
        self.assertTrue(rn in inter_hop.dict.keys())
        self.assertTrue((2, 3) in inter_hop.dict[rn].keys())

        # Overwrite existing hopping term
        rn = (0, 0, 0)
        inter_hop.add_hopping(rn, 0, 1, 2.5)
        self.assertAlmostEqual(inter_hop.dict[rn][(0, 1)], 2.5)

    def test_merge_prim_cell(self):
        """
        Test function 'merge_prim_cell'.

        :return: None.
        """
        pc_bra = tb.make_graphene_diamond()
        pc_ket = tb.make_graphene_diamond()
        tb.spiral_prim_cell(pc_ket, shift=0.5)

        # First we glue them using the 'Sample' class
        sample = tb.Sample(tb.SuperCell(pc_bra, dim=(3, 3, 1),
                                        pbc=(False, False, False)),
                           tb.SuperCell(pc_ket, dim=(3, 3, 1),
                                        pbc=(False, False, False)))
        for view in ("ab", "bc", "ca"):
            sample.plot(view=view)

        # Then we glue them with 'merge_prim_cell'.
        # It should yield the same result.
        merged_cell = tb.merge_prim_cell(pc_bra, pc_ket)
        sample = tb.Sample(tb.SuperCell(merged_cell, dim=(3, 3, 1),
                                        pbc=(False, False, False)))
        for view in ("ab", "bc", "ca"):
            sample.plot(view=view)

        # Then we bind the two primitive cells with an inter_hop_dict.
        inter_hop = tb.InterHopDict(pc_bra, pc_ket)
        inter_hop.add_hopping((0, 0, 0), 0, 0, 1.0)
        inter_hop.add_hopping((1, 1, 0), 0, 1, 1.0)
        merged_cell = tb.merge_prim_cell(pc_bra, pc_ket, inter_hop)
        sample = tb.Sample(tb.SuperCell(merged_cell, dim=(3, 3, 1),
                                        pbc=(False, False, False)))
        for view in ("ab", "bc", "ca"):
            sample.plot(view=view)

    def test_twist_hetero(self):
        """
        Test the procedure of constructing a twisted hetero-structure without
        inter-hopping between different layers.

        :return: None.
        """
        i = 5
        angle = calc_twist_angle(i)

        # Import primitive cell from database
        prim_cell_fixed = tb.make_graphene_diamond()

        # Build a twisted primitive cell
        prim_cell_twisted = tb.make_graphene_diamond()
        tb.spiral_prim_cell(prim_cell_twisted, angle=angle, shift=0.5)

        # Reshape primitive cells to yield layers
        hetero_lattice = calc_hetero_lattice(i, prim_cell_fixed)
        layer_fixed = tb.make_hetero_layer(prim_cell_fixed, hetero_lattice)
        layer_twisted = tb.make_hetero_layer(prim_cell_twisted, hetero_lattice)

        # Merge primitive cells
        merged_cell = tb.merge_prim_cell(layer_fixed, layer_twisted)

        # View the structure
        sample = tb.Sample(tb.SuperCell(merged_cell, dim=(3, 3, 1),
                                        pbc=(False, False, False)))
        sample.plot(with_orbitals=False, hop_as_arrows=False)

    def test_inter_hop_hetero(self):
        """
        Test the procedure of building inter_hop_dict for twisted
        hetero-structure using cKDTree.

        :return: None
        """
        i = 5
        angle = calc_twist_angle(i)

        # Import primitive cell from database
        prim_cell_fixed = tb.make_graphene_diamond()

        # Build a twisted primitive cell
        prim_cell_twisted = tb.make_graphene_diamond()
        tb.spiral_prim_cell(prim_cell_twisted, angle=angle, shift=0.5)

        # Reshape primitive cells to yield layers
        hetero_lattice = calc_hetero_lattice(i, prim_cell_fixed)
        layer_fixed = tb.make_hetero_layer(prim_cell_fixed, hetero_lattice)
        layer_twisted = tb.make_hetero_layer(prim_cell_twisted, hetero_lattice)

        # Get the Cartesian coordinates of orbitals of fixed layer
        layer_fixed.sync_array()
        pos_fixed = tb.frac2cart(layer_fixed.lat_vec, layer_fixed.orb_pos)

        # Same for twisted layer
        layer_twisted.sync_array()
        pos_twisted = tb.frac2cart(layer_twisted.lat_vec, layer_twisted.orb_pos)

        # Loop over neighbouring cells to build inter-hopping dictionary
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
                                          -2.7*math.exp(-(distance-0.246)**2))

        # Finally, we merge layers and inter_hop to yield a hetero-structure
        merged_cell = tb.merge_prim_cell(layer_fixed, layer_twisted, inter_hop)

        sample = tb.Sample(tb.SuperCell(merged_cell, dim=(3, 3, 1),
                                        pbc=(False, False, False)))
        sample.plot(with_orbitals=False, hop_as_arrows=False)


if __name__ == "__main__":
    unittest.main()
