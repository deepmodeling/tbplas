#! /usr/bin/env python

import unittest

import numpy as np
import matplotlib.pyplot as plt

import tbplas.builder.lattice as lat
import tbplas.builder.kpoints as kpt
import tbplas.builder.constants as consts
import tbplas.builder.exceptions as exc
from tbplas.builder import PrimitiveCell, extend_prim_cell, reshape_prim_cell
from test_utils import TestHelper


def make_cell():
    vectors = lat.gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
    cell = PrimitiveCell(vectors)
    cell.add_orbital([0.0, 0.0], 0.0)
    cell.add_orbital([1. / 3, 1. / 3], 0.0)
    cell.add_hopping([0, 0], 0, 1, -2.7)
    cell.add_hopping([1, 0], 1, 0, -2.7)
    cell.add_hopping([0, 1], 1, 0, -2.7)
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
        vectors = lat.gen_lattice_vectors(a=1.5, b=2.0, c=2.5)
        cell = PrimitiveCell(vectors, unit=consts.ANG)
        self.assertAlmostEqual(cell.lat_vec.item(0, 0), 0.15)
        self.assertAlmostEqual(cell.lat_vec.item(1, 1), 0.20)
        self.assertAlmostEqual(cell.lat_vec.item(2, 2), 0.25)
        cell = PrimitiveCell(vectors, unit=consts.NM)
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
        cell.lock()
        with self.assertRaises(exc.PCLockError) as cm:
            cell.add_orbital([1.2, 0.5], 0.1)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked primitive cell")

        # set_orbital
        cell = make_cell()
        cell.lock()
        with self.assertRaises(exc.PCLockError) as cm:
            cell.set_orbital(1, energy=0.25)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked primitive cell")

        # remove_orbital
        cell = make_cell()
        cell.lock()
        with self.assertRaises(exc.PCLockError) as cm:
            cell.remove_orbital(0)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked primitive cell")

        # add_hopping
        cell = make_cell()
        cell.lock()
        with self.assertRaises(exc.PCLockError) as cm:
            cell.add_hopping([-1, 0], 0, 0, 2.5)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked primitive cell")

        # remove_hopping
        cell = make_cell()
        cell.lock()
        with self.assertRaises(exc.PCLockError) as cm:
            cell.remove_hopping([0, 0], 0, 1)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked primitive cell")

    def test02_unlock(self):
        """
        Test if calling the functions that may change the structure of
        model raises errors when the model is unlocked.

        :return:
        """
        th = TestHelper(self)

        # add_orbital
        cell = make_cell()
        cell.lock()
        cell.unlock()

        def _test():
            cell.add_orbital([1.2, 0.5], 0.1)
        th.test_no_raise(_test, exc.PCLockError)

        # set_orbital
        cell = make_cell()
        cell.lock()
        cell.unlock()

        def _test():
            cell.set_orbital(1, energy=0.25)
        th.test_no_raise(_test, exc.PCLockError)

        # remove_orbital
        cell = make_cell()
        cell.lock()
        cell.unlock()

        def _test():
            cell.remove_orbital(0)
        th.test_no_raise(_test, exc.PCLockError)

        # add_hopping
        cell = make_cell()
        cell.lock()
        cell.unlock()

        def _test():
            cell.add_hopping([-1, 0], 0, 0, 2.5)
        th.test_no_raise(_test, exc.PCLockError)

        # remove_hopping
        cell = make_cell()
        cell.lock()
        cell.unlock()

        def _test():
            cell.remove_hopping([0, 0], 0, 1)
        th.test_no_raise(_test, exc.PCLockError)

    def test03_add_orbital(self):
        """
        Test if add_orbital works as expected.

        :return: None
        """
        # Test if feeding an illegal position raises the right exception
        cell = make_cell()
        with self.assertRaises(exc.OrbPositionLenError) as cm:
            cell.add_orbital([1.0], energy=1.0)
        self.assertRegex(str(cm.exception),
                         r"length of orbital position .+ not in \(2, 3\)")

        # Test the normal case
        cell = make_cell()
        orbitals = cell.orbital_list
        self.assertEqual(cell.num_orb, 2)
        self.assertAlmostEqual(np.linalg.norm(orbitals[0].position), 0.0)
        self.assertAlmostEqual(np.linalg.norm(orbitals[1].position),
                               4.714045207910317e-1)
        self.assertAlmostEqual(orbitals[0].energy, 0.0)
        self.assertAlmostEqual(orbitals[1].energy, 0.0)

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
            cell.set_orbital(-1, position=[1.5])
        self.assertRegex(str(cm.exception),
                         r"length of orbital position .+ not in \(2, 3\)")

        # Test setting both position and energy
        cell = make_cell()
        cell.set_orbital(-1, position=[1.5, 0.0], energy=4.2)
        self.assertEqual(cell.num_orb, 2)
        self.assertAlmostEqual(cell.orbital_list[-1].energy, 4.2)
        self.assertAlmostEqual(np.linalg.norm(cell.orbital_list[-1].position),
                               1.5)

        # Test setting position
        cell = make_cell()
        cell.set_orbital(-1, position=[1.5, 0.0])
        self.assertEqual(cell.num_orb, 2)
        self.assertAlmostEqual(cell.orbital_list[-1].energy, 0.0)
        self.assertAlmostEqual(np.linalg.norm(cell.orbital_list[-1].position),
                               1.5)

        # Test setting energy
        cell = make_cell()
        cell.set_orbital(-1, energy=4.2)
        self.assertEqual(cell.num_orb, 2)
        self.assertAlmostEqual(cell.orbital_list[-1].energy, 4.2)
        self.assertAlmostEqual(np.linalg.norm(cell.orbital_list[-1].position),
                               4.714045207910317e-1)

    def test04_get_orbital(self):
        """
        Test if get_orbital works as expected.

        :return:
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
        self.assertAlmostEqual(np.linalg.norm(cell.orbital_list[-1].position),
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
        self.assertEqual(len(cell.hopping_list), 0)

        # removing orbital #1
        cell = make_cell()
        cell.remove_orbital(1)
        self.assertEqual(cell.num_orb, 1)
        self.assertEqual(len(cell.hopping_list), 0)

        # adding orbital #2
        cell = make_cell()
        cell.add_orbital([0.5, 0.5], energy=-0.1)
        cell.add_hopping([0, 0], 0, 2, -1.5)
        cell.add_hopping([0, 0], 1, 2, -1.9)

        # removing orbital #0
        cell.remove_orbital(0)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(len(cell.hopping_list), 1)
        self.assertAlmostEqual(cell.hopping_list[-1].energy, -1.9)

        # removing orbital #1
        cell = make_cell()
        cell.add_orbital([0.5, 0.5], energy=-0.1)
        cell.add_hopping([0, 0], 0, 2, -1.5)
        cell.add_hopping([0, 0], 1, 2, -1.9)
        cell.remove_orbital(1)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(len(cell.hopping_list), 1)
        self.assertAlmostEqual(cell.hopping_list[-1].energy, -1.5)

        # removing orbital #2
        cell = make_cell()
        cell.add_orbital([0.5, 0.5], energy=-0.1)
        cell.add_hopping([0, 0], 0, 2, -1.5)
        cell.add_hopping([0, 0], 1, 2, -1.9)
        cell.remove_orbital(2)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(len(cell.hopping_list), 3)
        self.assertAlmostEqual(cell.hopping_list[-1].energy, -2.7)

    def test06_add_hopping(self):
        """
        Test if add_hopping and add_hopping_matrix work as expected.

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
        cell = make_cell()
        hopping_list = cell.hopping_list
        self.assertEqual(len(hopping_list), 3)
        self.assertEqual(hopping_list[0].index, (0, 0, 0, 0, 1))
        self.assertEqual(hopping_list[1].index, (1, 0, 0, 1, 0))
        self.assertEqual(hopping_list[2].index, (0, 1, 0, 1, 0))
        self.assertAlmostEqual(hopping_list[0].energy, -2.7)
        self.assertAlmostEqual(hopping_list[1].energy, -2.7)
        self.assertAlmostEqual(hopping_list[2].energy, -2.7)

        # Updating an existing hopping term
        cell = make_cell()
        cell.add_hopping(rn=(0, 0), orb_i=0, orb_j=1, energy=-2.8)
        self.assertEqual(len(cell.hopping_list), 3)
        self.assertAlmostEqual(cell.hopping_list[0].energy, -2.8)

        # Updating an existing conjugate hopping term
        cell = make_cell()
        cell.add_hopping(rn=(-1, 0), orb_i=0, orb_j=1, energy=-2.8)
        cell.add_hopping(rn=(0, 0), orb_i=1, orb_j=0, energy=-2.8)
        self.assertEqual(len(cell.hopping_list), 3)
        self.assertAlmostEqual(cell.hopping_list[1].energy, -2.8)

        # Add hopping as matrix
        cell = make_cell()

        # Error handling
        test_mat = np.zeros((3, 3))
        with self.assertRaises(ValueError) as cm:
            cell.add_hopping_matrix([0, 0], test_mat)
        self.assertRegex(str(cm.exception), r"Shape of hopping matrix \(3, 3\) "
                                            r"does not match num_orb 2")

        # Normal case
        hop_mat_00 = np.array([[0.7, -2.5], [-2.6, 3.6]])
        hop_mat_10 = np.array([[1.2, -2.6], [-2.3, 1.1]])
        hop_mat_01 = np.array([[1.6, -2.8], [-2.7, 1.2]])
        cell.add_hopping_matrix([0, 0], hop_mat_00)
        cell.add_hopping_matrix([1, 0], hop_mat_10)
        cell.add_hopping_matrix([0, 1], hop_mat_01)
        self.assertEqual(cell.get_hopping([0, 0], 0, 1).energy, -2.6)
        self.assertEqual(cell.get_hopping([1, 0], 0, 0).energy, 1.2)
        self.assertEqual(cell.get_hopping([1, 0], 1, 1).energy, 1.1)
        self.assertEqual(cell.get_hopping([1, 0], 0, 1).energy, -2.6)
        self.assertEqual(cell.get_hopping([1, 0], 1, 0).energy, -2.3)
        self.assertEqual(cell.get_hopping([0, 1], 0, 0).energy, 1.6)
        self.assertEqual(cell.get_hopping([0, 1], 1, 1).energy, 1.2)
        self.assertEqual(cell.get_hopping([0, 1], 0, 1).energy, -2.8)
        self.assertEqual(cell.get_hopping([0, 1], 1, 0).energy, -2.7)

    def test06_get_hopping(self):
        """
        Test if get_hopping works as expected.

        :return:
        """
        # Error handling
        cell = make_cell()
        with self.assertRaises(exc.PCHopNotFoundError) as cm:
            cell.get_hopping([-2, 0], 0, 1)
        self.assertRegex(str(cm.exception), r"hopping term .+ not found")

        # The normal case
        hopping = cell.get_hopping([0, 0], 0, 1)
        self.assertAlmostEqual(hopping.energy, -2.7)

        # Get the conjugate part
        def _test():
            hopping_2 = cell.get_hopping([0, 0], 1, 0)
            self.assertAlmostEqual(hopping_2.energy, -2.7)
        th = TestHelper(self)
        th.test_stdout(_test, ["INFO: given hopping term not found."
                               " Returning conjugate counterpart instead."])

    def test07_remove_hopping(self):
        """
        Test if remove_hopping works as expected.

        :return: None.
        """
        # Error handling
        cell = make_cell()
        with self.assertRaises(exc.PCHopNotFoundError) as cm:
            cell.remove_hopping([-2, 0], 0, 1)
        self.assertRegex(str(cm.exception), r"hopping term .+ not found")
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(len(cell.hopping_list), 3)

        # The normal case
        cell = make_cell()
        cell.remove_hopping([0, 0], 0, 1)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(len(cell.hopping_list), 2)

        # Remove conjugate part
        cell = make_cell()
        cell.remove_hopping([-1, 0], 0, 1)
        cell.remove_hopping([0, -1], 0, 1)
        self.assertEqual(cell.num_orb, 2)
        self.assertEqual(len(cell.hopping_list), 1)

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
        # 4th call, expected: updating orbitals
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
        self.assertEqual(len(cell.orbital_list), 2)
        self.assertEqual(len(cell.hopping_list), 0)
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
        k_path = kpt.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = cell.calc_bands(k_path)
        num_bands = bands.shape[1]
        for i in range(num_bands):
            plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
        plt.show()

        # Calculate DOS.
        k_points = kpt.gen_kmesh((120, 120, 1))
        energies, dos = cell.calc_dos(k_points)
        plt.plot(energies, dos)
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
        k_path = kpt.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = extend_cell.calc_bands(k_path)
        num_bands = bands.shape[1]
        for i in range(num_bands):
            plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
        plt.show()

        # DOS
        k_points = kpt.gen_kmesh((24, 24, 1))
        energies, dos = extend_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

    def test13_reshape_prim_cell(self):
        """
        Test function 'reshape_prim_cell'.

        :return: None
        """
        print("\nReference rectangular cell:")
        sqrt3 = 1.73205080757
        a = 2.46
        cc_bond = sqrt3 / 3 * a
        vectors = lat.gen_lattice_vectors(sqrt3 * cc_bond, 3 * cc_bond)
        cell = PrimitiveCell(vectors)
        cell.add_orbital((0, 0))
        cell.add_orbital((0, 2./3))
        cell.add_orbital((1./2, 1./6))
        cell.add_orbital((1./2, 1./2))
        cell.add_hopping([0, 0], 0, 2, -2.7)
        cell.add_hopping([0, 0], 2, 3, -2.7)
        cell.add_hopping([0, 0], 3, 1, -2.7)
        cell.add_hopping([0, 1], 1, 0, -2.7)
        cell.add_hopping([1, 0], 3, 1, -2.7)
        cell.add_hopping([1, 0], 2, 0, -2.7)
        cell.plot()

        print("\nTest rectangular cell:")
        cell = make_cell()
        lat_frac = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]])
        cell = reshape_prim_cell(cell, lat_frac)
        self.assertEqual(cell.num_orb, 4)
        cell.plot()


if __name__ == "__main__":
    unittest.main()
