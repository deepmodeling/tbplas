#! /usr/bin/env python

import unittest
from math import cos, sin, sqrt

import numpy as np
import matplotlib.pyplot as plt

from tbplas import (gen_lattice_vectors, gen_kpath, gen_kmesh, DiagSolver,
                    PrimitiveCell, extend_prim_cell, reshape_prim_cell,
                    ANG, NM, Visualizer, frac2cart, TestHelper)
import tbplas.builder.exceptions as exc


def make_cell():
    vectors = gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
    cell = PrimitiveCell(vectors)
    cell.add_orbital((0.0, 0.0), 0.0, label="C_pz")
    cell.add_orbital((1. / 3, 1. / 3), 0.0, label="C_pz")
    cell.add_hopping((0, 0), 0, 1, -2.7)
    cell.add_hopping((1, 0), 1, 0, -2.7)
    cell.add_hopping((0, 1), 1, 0, -2.7)
    return cell


class TestPrimitive(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test00_pc_init(self):
        """
        Test initialization of 'PrimitiveCell' class.

        :return: None
        """
        # Check if feeding an illegal lattice vector raises the right exception
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        with self.assertRaises(exc.LatVecError) as cm:
            PrimitiveCell(vectors)
        self.assertRegex(str(cm.exception), r"^illegal lattice vectors$")

        # Check unit conversion
        vectors = gen_lattice_vectors(a=1.5, b=2.0, c=2.5)
        cell = PrimitiveCell(vectors, unit=ANG)
        self.assertAlmostEqual(cell.lat_vec.item(0, 0), 0.15)
        self.assertAlmostEqual(cell.lat_vec.item(1, 1), 0.20)
        self.assertAlmostEqual(cell.lat_vec.item(2, 2), 0.25)
        cell = PrimitiveCell(vectors, unit=NM)
        self.assertAlmostEqual(cell.lat_vec.item(0, 0), 1.5)
        self.assertAlmostEqual(cell.lat_vec.item(1, 1), 2.0)
        self.assertAlmostEqual(cell.lat_vec.item(2, 2), 2.5)

    def test01_lock(self):
        """
        Test if calling the functions that may change the structure of
        model raises errors when the model is locked.

        :return: None
        """
        # add_orbital
        cell = make_cell()
        cell.lock("test")
        with self.assertRaises(exc.LockError) as cm:
            cell.add_orbital((1.2, 0.5), 0.1)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked object")

        # set_orbital
        cell = make_cell()
        cell.lock("test")
        with self.assertRaises(exc.LockError) as cm:
            cell.set_orbital(1, energy=0.25)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked object")

        # remove_orbital
        cell = make_cell()
        cell.lock("test")
        with self.assertRaises(exc.LockError) as cm:
            cell.remove_orbital(0)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked object")

        # add_hopping
        cell = make_cell()
        cell.lock("test")
        with self.assertRaises(exc.LockError) as cm:
            cell.add_hopping((-1, 0), 0, 0, 2.5)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked object")

        # remove_hopping
        cell = make_cell()
        cell.lock("test")
        with self.assertRaises(exc.LockError) as cm:
            cell.remove_hopping((0, 0), 0, 1)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked object")

    def test02_unlock(self):
        """
        Test if calling the functions that may change the structure of
        model raises errors when the model is unlocked.

        :return:
        """
        th = TestHelper(self)

        # add_orbital
        cell = make_cell()
        cell.lock("test")
        cell.unlock()

        def _test():
            cell.add_orbital((1.2, 0.5), 0.1)
        th.test_no_raise(_test, exc.LockError)

        # set_orbital
        cell = make_cell()
        cell.lock("test")
        cell.unlock()

        def _test():
            cell.set_orbital(1, energy=0.25)
        th.test_no_raise(_test, exc.LockError)

        # remove_orbital
        cell = make_cell()
        cell.lock("test")
        cell.unlock()

        def _test():
            cell.remove_orbital(0)
        th.test_no_raise(_test, exc.LockError)

        # add_hopping
        cell = make_cell()
        cell.lock("test")
        cell.unlock()

        def _test():
            cell.add_hopping((-1, 0), 0, 0, 2.5)
        th.test_no_raise(_test, exc.LockError)

        # remove_hopping
        cell = make_cell()
        cell.lock("test")
        cell.unlock()

        def _test():
            cell.remove_hopping((0, 0), 0, 1)
        th.test_no_raise(_test, exc.LockError)

    def test03_add_orbital(self):
        """
        Test if add_orbital works as expected.

        :return: None
        """
        # Test if feeding an illegal position raises the right exception
        cell = make_cell()
        with self.assertRaises(exc.OrbPositionLenError) as cm:
            cell.add_orbital((1.0,), energy=1.0)
        self.assertRegex(str(cm.exception),
                         r"length of orbital position .+ not in \(2, 3\)")

        # Test the normal case
        cell = make_cell()
        orbitals = cell.orbitals
        self.assertEqual(cell.num_orb, 2)
        self.assertAlmostEqual(np.linalg.norm(orbitals[0].position), 0.0)
        self.assertAlmostEqual(np.linalg.norm(orbitals[1].position),
                               4.714045207910317e-1)
        self.assertAlmostEqual(orbitals[0].energy, 0.0)
        self.assertAlmostEqual(orbitals[1].energy, 0.0)
        self.assertEqual(orbitals[0].label, "C_pz")
        self.assertEqual(orbitals[1].label, "C_pz")

        # Test adding orbital with Cartesian coordinates
        th = TestHelper(self)
        vectors = gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
        cell_ref = PrimitiveCell(vectors)
        cell_ref.add_orbital((1. / 3, 1. / 3), 0.0, label="C_pz")
        cell_ref.add_orbital((2. / 3, 2. / 3), 0.0, label="C_pz")
        orb_pos_ref = frac2cart(cell_ref.lat_vec, cell_ref.orb_pos)

        cell_test = PrimitiveCell(vectors)
        cell_test.add_orbital_cart(orb_pos_ref[0] * 10, unit=ANG, label="C_pz")
        cell_test.add_orbital_cart(orb_pos_ref[1], unit=NM, label="C_pz")

        th.test_equal_array(cell_ref.orb_pos, cell_test.orb_pos, almost=True)
        orb_pos_test = frac2cart(cell_test.lat_vec, cell_test.orb_pos)
        th.test_equal_array(orb_pos_ref, orb_pos_test, almost=True)

    def test04_set_orbital(self):
        """
        Test if set_orbital works as expected.

        :return: None
        """
        # Test error handling for illegal orbital index
        cell = make_cell()
        with self.assertRaises(exc.PCOrbIndexError) as cm:
            cell.set_orbital(3, energy=4.2)
        self.assertRegex(str(cm.exception), r"orbital index .+ out of range")

        # Test error handling for illegal position
        cell = make_cell()
        with self.assertRaises(exc.OrbPositionLenError) as cm:
            cell.set_orbital(-1, position=(1.5,))
        self.assertRegex(str(cm.exception),
                         r"length of orbital position .+ not in \(2, 3\)")

        # Test setting both position and energy
        cell = make_cell()
        cell.set_orbital(-1, position=(1.5, 0.0), energy=4.2)
        self.assertEqual(cell.num_orb, 2)
        self.assertAlmostEqual(cell.orbitals[-1].energy, 4.2)
        self.assertAlmostEqual(np.linalg.norm(cell.orbitals[-1].position),
                               1.5)

        # Test setting position
        cell = make_cell()
        cell.set_orbital(-1, position=(1.5, 0.0))
        self.assertEqual(cell.num_orb, 2)
        self.assertAlmostEqual(cell.orbitals[-1].energy, 0.0)
        self.assertAlmostEqual(np.linalg.norm(cell.orbitals[-1].position),
                               1.5)

        # Test setting energy
        cell = make_cell()
        cell.set_orbital(-1, energy=4.2)
        self.assertEqual(cell.num_orb, 2)
        self.assertAlmostEqual(cell.orbitals[-1].energy, 4.2)
        self.assertAlmostEqual(np.linalg.norm(cell.orbitals[-1].position),
                               4.714045207910317e-1)

        # Test setting label
        cell = make_cell()
        cell.set_orbital(-1, label="C_pz_2")
        self.assertEqual(cell.orbitals[-1].label, "C_pz_2")

        # Test setting orbital with Cartesian coordinates
        th = TestHelper(self)
        cell_ref = make_cell()
        cell_ref.set_orbital(0, position=(1. / 3, 1. / 3))
        cell_ref.set_orbital(1, position=(2. / 3, 2. / 3))
        orb_pos_ref = frac2cart(cell_ref.lat_vec, cell_ref.orb_pos)

        cell_test = make_cell()
        cell_test.set_orbital_cart(0, position=orb_pos_ref[0]*10, unit=ANG)
        cell_test.set_orbital_cart(1, position=orb_pos_ref[1], unit=NM)

        th.test_equal_array(cell_ref.orb_pos, cell_test.orb_pos, almost=True)
        orb_pos_test = frac2cart(cell_test.lat_vec, cell_test.orb_pos)
        th.test_equal_array(orb_pos_ref, orb_pos_test, almost=True)

    def test04_get_orbital(self):
        """
        Test get_orbital.

        :return: None
        """
        # Error handling
        cell = make_cell()
        with self.assertRaises(exc.PCOrbIndexError) as cm:
            cell.get_orbital(3)
        self.assertRegex(str(cm.exception), r"orbital index .+ out of range")

        # Normal case
        cell = make_cell()
        orbital = cell.get_orbital(-1)
        self.assertAlmostEqual(orbital.energy, 0.0)
        self.assertAlmostEqual(np.linalg.norm(cell.orbitals[-1].position),
                               4.714045207910317e-1)

    def test05_remove_orbital(self):
        """
        Test if remove_orbital works as expected.

        :return: None
        """
        # Test if feeding an illegal orbital index raises the right error
        cell = make_cell()
        with self.assertRaises(exc.PCOrbIndexError) as cm:
            cell.remove_orbital(3)
        self.assertRegex(str(cm.exception), r"orbital index .+ out of range")

        # removing orbital #0
        cell = make_cell()
        cell.remove_orbital(0)
        self.assertEqual(cell.num_orb, 1)
        self.assertEqual(cell.num_hop, 0)

        # removing orbital #1
        cell = make_cell()
        cell.remove_orbital(1)
        self.assertEqual(cell.num_orb, 1)
        self.assertEqual(cell.num_hop, 0)

        # adding orbital #2
        cell = make_cell()
        cell.add_orbital((0.5, 0.5), energy=-0.1)
        cell.add_hopping((0, 0), 0, 2, -1.5)
        cell.add_hopping((0, 0), 1, 2, -1.9)

        # removing orbital #0
        cell.remove_orbital(0)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(cell.num_hop, 1)
        self.assertAlmostEqual(cell.get_hopping((0, 0, 0), 0, 1), -1.9)

        # removing orbital #1
        cell = make_cell()
        cell.add_orbital((0.5, 0.5), energy=-0.1)
        cell.add_hopping((0, 0), 0, 2, -1.5)
        cell.add_hopping((0, 0), 1, 2, -1.9)
        cell.remove_orbital(1)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(cell.num_hop, 1)
        self.assertAlmostEqual(cell.get_hopping((0, 0, 0), 0, 1), -1.5)

        # removing orbital #2
        cell = make_cell()
        cell.add_orbital((0.5, 0.5), energy=-0.1)
        cell.add_hopping((0, 0), 0, 2, -1.5)
        cell.add_hopping((0, 0), 1, 2, -1.9)
        cell.remove_orbital(2)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(cell.num_hop, 3)
        self.assertAlmostEqual(cell.get_hopping((0, 0, 0), 0, 1), -2.7)

    def test06_add_hopping(self):
        """
        Test if add_hopping works as expected.

        :return: None
        """
        # Error handling
        cell = make_cell()
        with self.assertRaises(exc.PCOrbIndexError) as cm:
            cell.add_hopping(rn=(2, 0), orb_i=2, orb_j=0, energy=-2.8)
        self.assertRegex(str(cm.exception), r"orbital index .+ out of range")
        with self.assertRaises(exc.PCOrbIndexError) as cm:
            cell.add_hopping(rn=(2, 0), orb_i=0, orb_j=2, energy=-2.8)
        self.assertRegex(str(cm.exception), r"orbital index .+ out of range")
        with self.assertRaises(exc.PCHopDiagonalError) as cm:
            cell.add_hopping(rn=(0, 0), orb_i=0, orb_j=0, energy=-2.8)
        self.assertRegex(str(cm.exception), r"hopping term .+ is diagonal")

        # The normal case
        th = TestHelper(self)
        cell = make_cell()
        self.assertEqual(cell.num_hop, 3)
        th.test_equal_array(cell.hop_ind[0], np.array([0, 0, 0, 0, 1]))
        th.test_equal_array(cell.hop_ind[1], np.array([1, 0, 0, 1, 0]))
        th.test_equal_array(cell.hop_ind[2], np.array([0, 1, 0, 1, 0]))
        self.assertAlmostEqual(cell.hop_eng[0], -2.7)
        self.assertAlmostEqual(cell.hop_eng[1], -2.7)
        self.assertAlmostEqual(cell.hop_eng[2], -2.7)

        # Updating an existing hopping term
        cell = make_cell()
        cell.add_hopping(rn=(0, 0), orb_i=0, orb_j=1, energy=-2.8)
        self.assertEqual(cell.num_hop, 3)
        self.assertAlmostEqual(cell.get_hopping((0, 0, 0), 0, 1), -2.8)

        # Updating an existing conjugate hopping term
        cell = make_cell()
        cell.add_hopping(rn=(-1, 0), orb_i=0, orb_j=1, energy=-2.7)
        cell.add_hopping(rn=(0, 0), orb_i=1, orb_j=0, energy=-2.8)
        self.assertEqual(cell.num_hop, 3)
        self.assertAlmostEqual(cell.get_hopping((-1, 0, 0), 0, 1), -2.7)
        self.assertAlmostEqual(cell.get_hopping((0, 0, 0), 0, 1), -2.8)

    def test06_get_hopping(self):
        """
        Test get_hopping.

        :return:
        """
        # Error handling
        cell = make_cell()
        with self.assertRaises(exc.PCHopNotFoundError) as cm:
            cell.get_hopping((-2, 0), 0, 1)
        self.assertRegex(str(cm.exception), r"hopping term .+ not found")

        # The normal case
        energy = cell.get_hopping((0, 0), 0, 1)
        self.assertAlmostEqual(energy, -2.7)

    def test07_remove_hopping(self):
        """
        Test if remove_hopping works as expected.

        :return: None.
        """
        # Error handling
        cell = make_cell()
        with self.assertRaises(exc.PCHopNotFoundError) as cm:
            cell.remove_hopping((-2, 0), 0, 1)
        self.assertRegex(str(cm.exception), r"hopping term .+ not found")
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(cell.num_hop, 3)

        # The normal case
        cell = make_cell()
        cell.remove_hopping((0, 0), 0, 1)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(cell.num_hop, 2)

        # Remove conjugate part
        cell = make_cell()
        cell.remove_hopping((-1, 0), 0, 1)
        cell.remove_hopping((0, -1), 0, 1)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(cell.num_hop, 1)

    def test08_sync_array(self):
        """
        Test if sync_array works as expected.

        :return: None
        """
        th = TestHelper(self)

        orb = r"INFO: updating pc orbital arrays"
        hop = r"INFO: updating pc hopping arrays"
        no_orb = r"INFO: no need to update pc orbital arrays"
        no_hop = r"INFO: no need to update pc hopping arrays"
        update_both = [orb, hop]
        update_orb = [orb, no_hop]
        update_hop = [no_orb, hop]
        update_none = [no_orb, no_hop]

        def _test():
            cell.sync_array(verbose=True)

        def _test_force_sync():
            cell.sync_array(verbose=True, force_sync=True)

        # DO NOT add or remove anything
        cell = make_cell()
        # 1st call, expected: updating both
        th.test_stdout(_test, update_both)
        # 2nd call, expected: updating none
        th.test_stdout(_test, update_none)
        # 3rd call, expected: updating both
        th.test_stdout(_test_force_sync, update_both)

        # Add one orbital
        cell = make_cell()
        # 1st call, expected: updating both
        th.test_stdout(_test, update_both)
        # 2nd call, expected: updating orbital
        cell.add_orbital((0.0, 0.5), 0.15)
        th.test_stdout(_test, update_orb)
        # 3rd call, expected: updating none
        th.test_stdout(_test, update_none)
        # 4th call, expected: updating both
        th.test_stdout(_test_force_sync, update_both)

        # Add one hopping term
        cell = make_cell()
        # 1st call, expected: updating both
        th.test_stdout(_test, update_both)
        # 2nd call, expected: updating hopping
        cell.add_hopping((0, 2), 0, 1, -1.5)
        th.test_stdout(_test, update_hop)
        # 3rd call, expected: update none
        th.test_stdout(_test, update_none)
        # 4th call, expected: updating both
        th.test_stdout(_test_force_sync, update_both)

        # Removing orbitals
        cell = make_cell()
        # 1st call, expected: updating both
        th.test_stdout(_test, update_both)
        # 2nd call, expected: updating both
        cell.remove_orbital(0)
        th.test_stdout(_test, update_both)
        self.assertEqual(cell.hop_ind, None)
        self.assertEqual(cell.hop_eng, None)
        # 3rd call, expected: updating nothing
        th.test_stdout(_test, update_none)
        # 4th call, expected: updating orb
        cell.remove_orbital(0)
        th.test_stdout(_test, update_orb)
        self.assertEqual(cell.hop_ind, None)
        self.assertEqual(cell.hop_eng, None)
        self.assertEqual(cell.orb_pos, None)
        self.assertEqual(cell.orb_eng, None)
        # 5th call, expected: updating nothing
        th.test_stdout(_test, update_none)
        # 6th call, expected: updating both
        th.test_stdout(_test_force_sync, update_both)

        # Removing hopping terms
        cell = make_cell()
        # 1st call, expected: updating both
        th.test_stdout(_test, update_both)
        # 2nd call, expected: updating hopping
        cell.remove_hopping((0, 0), 0, 1)
        th.test_stdout(_test, update_hop)
        # 3rd call, expected: updating none
        th.test_stdout(_test, update_none)
        # 4th call, expected: updating hopping
        cell.remove_hopping((-1, 0), 0, 1)
        th.test_stdout(_test, update_hop)
        # 5th call, expected: updating none
        th.test_stdout(_test, update_none)
        # 6th call, expected: updating hopping
        cell.remove_hopping((0, 1), 1, 0)
        th.test_stdout(_test, update_hop)
        self.assertEqual(cell.hop_ind, None)
        self.assertEqual(cell.hop_eng, None)
        # 7th call, expected: updating none
        th.test_stdout(_test, update_none)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(cell.num_hop, 0)
        self.assertEqual(cell.hop_ind, None)
        self.assertEqual(cell.hop_eng, None)
        # 8th call, expected: updating both
        th.test_stdout(_test_force_sync, update_both)

    def test09_plot(self):
        """
        Test plotting of orbitals and hopping terms.

        :return: None
        """
        cell = make_cell()
        cell.plot()

    def test10_print(self):
        """
        Test printing attributes.

        :return: None
        """
        cell = make_cell()
        cell.print()

    def test11_calc_bands_dos(self):
        """
        Test band structure and dos calculation.

        :return: None
        """
        # Calculate band structure.
        cell = make_cell()
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_path, k_idx = gen_kpath(k_points, [40, 40, 40])
        k_len, bands = cell.calc_bands(k_path)
        num_bands = bands.shape[1]
        for i in range(num_bands):
            plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
        plt.show()

        # Calculate DOS.
        k_points = gen_kmesh((120, 120, 1))
        energies, dos = cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

        # Calculate band structure from user-defined analytical Hamiltonian
        a, t, sqrt3 = 0.246, 2.7, sqrt(3.0)
        recip_lat = cell.get_reciprocal_vectors()

        def _exp(x):
            return cos(x) + 1j * sin(x)

        def _hk(kpt, ham):
            ka = np.matmul(kpt, recip_lat) * a
            kxa, kya = ka.item(0), ka.item(1)
            fk = _exp(kya / sqrt3) + 2 * _exp(-kya / 2 / sqrt3) * cos(kxa / 2)
            ham[0, 1] = t * fk
            ham[1, 0] = t * fk.conjugate()

        solver = DiagSolver(cell, hk_dense=_hk)
        k_len, bands = solver.calc_bands(k_path)[:2]
        num_bands = bands.shape[1]
        for i in range(num_bands):
            plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
        plt.show()

    def test12_extend_prim_cell(self):
        """
        Test function 'extend_prim_cell'.

        :return: None.
        """
        th = TestHelper(self)
        cell = make_cell()

        # Exception handling
        def _test():
            extend_prim_cell(cell, dim=(0, 0, 1))
        th.test_raise(_test, ValueError, r"Dimension along direction [0-9]"
                                         r" should not be smaller than 1")

        # Plot
        extend_cell = extend_prim_cell(cell, dim=(3, 3, 1))
        self.assertEqual(extend_cell.extended, 9)
        extend_cell.plot()

        # Band structure
        extend_cell = extend_prim_cell(cell, dim=(5, 5, 1))
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_path, k_idx = gen_kpath(k_points, [40, 40, 40])
        k_len, bands = extend_cell.calc_bands(k_path)
        num_bands = bands.shape[1]
        for i in range(num_bands):
            plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
        plt.show()

        # DOS
        k_points = gen_kmesh((24, 24, 1))
        energies, dos = extend_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

        # Orbital labels
        self.assertSetEqual(set([orb.label for orb in extend_cell.orbitals]),
                            {"C_pz"})

    def test13_reshape_prim_cell(self):
        """
        Test function 'reshape_prim_cell'.

        :return: None
        """
        print("\nReference rectangular cell:")
        sqrt3 = 1.73205080757
        a = 2.46
        cc_bond = sqrt3 / 3 * a
        vectors = gen_lattice_vectors(sqrt3 * cc_bond, 3 * cc_bond)
        cell = PrimitiveCell(vectors)
        cell.add_orbital((0, 0))
        cell.add_orbital((0, 2./3))
        cell.add_orbital((1./2, 1./6))
        cell.add_orbital((1./2, 1./2))
        cell.add_hopping((0, 0), 0, 2, -2.7)
        cell.add_hopping((0, 0), 2, 3, -2.7)
        cell.add_hopping((0, 0), 3, 1, -2.7)
        cell.add_hopping((0, 1), 1, 0, -2.7)
        cell.add_hopping((1, 0), 3, 1, -2.7)
        cell.add_hopping((1, 0), 2, 0, -2.7)
        cell.plot()

        print("\nTest rectangular cell:")
        cell = make_cell()
        lat_frac = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]])
        cell = reshape_prim_cell(cell, lat_frac)
        self.assertEqual(cell.num_orb, 4)
        cell.plot()

        # Check orbital labels
        self.assertSetEqual(set([orb.label for orb in cell.orbitals]), {"C_pz"})

    def test16_apply_pbc(self):
        """
        Test function 'apply_pbc'.

        :return: None
        """
        lat_frac = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]])
        rect_cell = reshape_prim_cell(make_cell(), lat_frac)

        # GNR along AM direction
        gnr = extend_prim_cell(rect_cell, dim=(3, 3, 1))
        gnr.plot(with_conj=False)
        gnr.apply_pbc(pbc=(False, True, False))
        gnr.plot(with_conj=False)
        gnr.trim()
        gnr.plot(with_conj=False)

        k_points = np.array([
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ])
        k_label = ["X", "G", "X"]
        k_path, k_idx = gen_kpath(k_points, [40, 40])
        k_len, bands = gnr.calc_bands(k_path)
        Visualizer().plot_bands(k_len, bands, k_idx, k_label)

        # GNR along ZZ direction
        gnr = extend_prim_cell(rect_cell, dim=(3, 3, 1))
        gnr.plot(with_conj=False)
        gnr.apply_pbc(pbc=(True, False, False))
        gnr.plot(with_conj=False)
        gnr.trim()
        gnr.plot(with_conj=False)

        k_points = np.array([
            [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        k_label = ["X", "G", "X"]
        k_path, k_idx = gen_kpath(k_points, [40, 40])
        k_len, bands = gnr.calc_bands(k_path)
        Visualizer().plot_bands(k_len, bands, k_idx, k_label)

    def test17_get_orbital_positions_cart(self):
        """
        Test method 'get_orbital_positions_cart'.

        :return: None
        """
        th = TestHelper(self)
        cell = make_cell()
        orb_pos_ref = frac2cart(cell.lat_vec, cell.orb_pos)
        orb_pos_test = cell.orb_pos_nm
        th.test_equal_array(orb_pos_ref, orb_pos_test, almost=True)
        orb_pos_ref *= 10.0
        orb_pos_test = cell.orb_pos_ang
        th.test_equal_array(orb_pos_ref, orb_pos_test, almost=True)

    def test18_set_ham(self):
        """
        Test 'set_ham_dense' and 'set_ham_csr' methods.

        :return: None
        """
        th = TestHelper(self)
        cell = make_cell()
        cell.sync_array()

        # Check if convention I and convention II produce different results
        ham1 = np.zeros((cell.num_orb, cell.num_orb), dtype=np.complex128)
        ham2 = np.zeros((cell.num_orb, cell.num_orb), dtype=np.complex128)
        kpt = np.array([0.35, 0.50, 0.0])
        cell.set_ham_dense(kpt, ham1, convention=1)
        cell.set_ham_dense(kpt, ham2, convention=2)
        th.test_no_equal_array(ham1, ham2)

        # Check if set_ham_dense agrees with set_ham_csr
        ham1_csr = cell.set_ham_csr(kpt, convention=1)
        ham2_csr = cell.set_ham_csr(kpt, convention=2)
        th.test_equal_array(ham1, ham1_csr.todense(), almost=True)
        th.test_equal_array(ham2, ham2_csr.todense(), almost=True)


if __name__ == "__main__":
    unittest.main()
