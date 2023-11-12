#! /usr/bin/env python

import unittest

import numpy as np

import tbplas as tb
from tbplas.builder.base import (check_rn, check_pos, check_pbc, check_conj,
                                 IntraHopping, InterHopping)
import tbplas.builder.exceptions as exc


class MyTest(unittest.TestCase):

    def test_check_coord(self):
        """
        Test check_* functions.

        :return: None
        """
        # check_rn
        # Normal 3d case
        coord, status = check_rn((0, 0, 0))
        self.assertTupleEqual(coord, (0, 0, 0))
        self.assertTrue(status)

        # Normal 2d case with default complete_item
        coord, status = check_rn((0, 0))
        self.assertTupleEqual(coord, (0, 0, 0))
        self.assertTrue(status)

        # Normal 2d case with user-defined complete_item
        coord, status = check_rn((0, 0), complete_item=1)
        self.assertTupleEqual(coord, (0, 0, 1))
        self.assertTrue(status)

        # Abnormal 1d case
        coord, status = check_rn((0,),)
        self.assertTupleEqual(coord, (0,))
        self.assertFalse(status)

        # check_pos
        # Normal 3d case
        coord, status = check_pos((0.0, 0.0, 0.0))
        self.assertTupleEqual(coord, (0.0, 0.0, 0.0))
        self.assertTrue(status)

        # Normal 2d case with default complete_item
        coord, status = check_pos((0.0, 0.0))
        self.assertTupleEqual(coord, (0.0, 0.0, 0.0))
        self.assertTrue(status)

        # Normal 2d case with user-defined complete_item
        coord, status = check_pos((0.0, 0.0), complete_item=1.0)
        self.assertTupleEqual(coord, (0.0, 0.0, 1.0))
        self.assertTrue(status)

        # Abnormal 1d case
        coord, status = check_pos((0.0,),)
        self.assertTupleEqual(coord, (0.0,))
        self.assertFalse(status)

        # check_pbc
        # Normal 3d case
        coord, status = check_pbc((True, True, True))
        self.assertTupleEqual(coord, (True, True, True))
        self.assertTrue(status)

        # Normal 2d case with default complete_item
        coord, status = check_pbc((True, True))
        self.assertTupleEqual(coord, (True, True, False))
        self.assertTrue(status)

        # Normal 2d case with user-defined complete_item
        coord, status = check_pbc((True, True), complete_item=True)
        self.assertTupleEqual(coord, (True, True, True))
        self.assertTrue(status)

        # Abnormal 1d case
        coord, status = check_pbc((True,),)
        self.assertTupleEqual(coord, (True,))
        self.assertFalse(status)

    def test_invert_rn(self):
        """
        Test the accuracy and speed of invert_rn.

        :return:
        """
        # Accuracy
        test_dict = {
            (1, -1, 2, 0, 2): False, (-1, -1, 2, 0, 2): True,
            (0, 1, 2, 0, 1): False, (0, -1, 2, 0, 1): True,
            (0, 0, 1, 1, 0): False, (0, 0, -1, 1, 0): True,
            (0, 0, 0, 0, 1): False, (0, 0, 0, 1, 0): True,
        }
        for key, value in test_dict.items():
            self.assertEqual(check_conj(key), value)

        # Speed
        timer = tb.Timer()
        timer.tic("invert_rn")
        for i in range(-100, 100):
            for j in range(-100, 100):
                for k in range(-100, 100):
                    check_conj((i, j, k, 0, 1))
        timer.toc("invert_rn")
        timer.report_total_time()

    def test_add_hopping_intra(self):
        """
        Test if 'add_hopping' method of 'IntraHopping' works as expected.

        :return: None
        """
        hop_dict = IntraHopping()

        # Add a normal term that does not need to invert.
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        self.assertEqual(hop_dict.hoppings[(1, -1, 2)][(1, 2)], 1.2+1j)
        # Overwriting.
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.5+1j)
        self.assertEqual(hop_dict.hoppings[(1, -1, 2)][(1, 2)], 1.5+1j)
        hop_dict.add_hopping(rn=(-1, 1, -2), orb_i=2, orb_j=1, energy=1.5+2j)
        self.assertEqual(hop_dict.hoppings[(1, -1, 2)][(1, 2)], 1.5-2j)

        # Add a term that needs to invert.
        hop_dict.add_hopping(rn=(-1, -1, 2), orb_i=1, orb_j=2, energy=1.2+2j)
        self.assertEqual(hop_dict.hoppings[(1, 1, -2)][(2, 1)], 1.2-2j)
        # Overwriting.
        hop_dict.add_hopping(rn=(-1, -1, 2), orb_i=1, orb_j=2, energy=1.2+5j)
        self.assertEqual(hop_dict.hoppings[(1, 1, -2)][(2, 1)], 1.2-5j)
        hop_dict.add_hopping(rn=(1, 1, -2), orb_i=2, orb_j=1, energy=1.2+7j)
        self.assertEqual(hop_dict.hoppings[(1, 1, -2)][(2, 1)], 1.2+7j)

        # Add a normal term in (0, 0, 0) cell.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.2+2j)
        self.assertEqual(hop_dict.hoppings[(0, 0, 0)][(1, 2)], 1.2+2j)
        # Overwriting.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.2+3j)
        self.assertEqual(hop_dict.hoppings[(0, 0, 0)][(1, 2)], 1.2+3j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=2, orb_j=1, energy=1.2-5j)
        self.assertEqual(hop_dict.hoppings[(0, 0, 0)][(1, 2)], 1.2+5j)

        # Add a term in (0, 0, 0) cell that needs to invert.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=3, orb_j=1, energy=1.2+2j)
        self.assertEqual(hop_dict.hoppings[(0, 0, 0)][(1, 3)], 1.2-2j)
        # Overwriting.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=3, orb_j=1, energy=1.2+1j)
        self.assertEqual(hop_dict.hoppings[(0, 0, 0)][(1, 3)], 1.2-1j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=3, energy=1.2+5j)
        self.assertEqual(hop_dict.hoppings[(0, 0, 0)][(1, 3)], 1.2+5j)

    def test_add_hopping_inter(self):
        """
        Test if 'add_hopping' method of 'InterHopping' works as expected.

        :return: None
        """
        hop_dict = InterHopping(None, None)
        th = tb.TestHelper(self)

        # Add a normal term that does not need auto-complete.
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        self.assertEqual(hop_dict.hoppings[(1, -1, 2)][(1, 2)], 1.2+1j)

        # Overwriting.
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.5+1j)
        self.assertEqual(hop_dict.hoppings[(1, -1, 2)][(1, 2)], 1.5+1j)

        # Add an opposite hopping term (not conjugate)
        hop_dict.add_hopping(rn=(-1, 1, -2), orb_i=2, orb_j=1, energy=1.5+2j)
        self.assertEqual(hop_dict.hoppings[(-1, 1, -2)][(2, 1)], 1.5+2j)
        self.assertEqual(hop_dict.hoppings[(1, -1, 2)][(1, 2)], 1.5+1j)

        # Add a normal term that needs auto-complete
        hop_dict.add_hopping(rn=(-1, 1), orb_i=1, orb_j=1, energy=1.5+2j)
        self.assertEqual(hop_dict.hoppings[(-1, 1, 0)][(1, 1)], 1.5+2j)

        # Add an abnormal term that triggers CellIndexLenError
        def _test():
            hop_dict.add_hopping(rn=(-1,), orb_i=1, orb_j=1, energy=1.5 + 2j)
        th.test_raise(_test, exc.CellIndexLenError,
                      r"length of cell index .+ not in \(2, 3\)")

    def test_get_hopping_intra(self):
        """
        Test if 'get_hopping' method of 'IntraHopping' works as expected.

        :return: None
        """
        hop_dict = IntraHopping()

        # Void case
        energy, status = hop_dict.get_hopping((1, -1, 2), 1, 2)
        self.assertEqual(energy, None)
        self.assertFalse(status)

        # Get a normal term that does not need to invert.
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        energy, status = hop_dict.get_hopping((1, -1, 2), 1, 2)
        self.assertEqual(energy, 1.2+1j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping((-1, 1, -2), 2, 1)
        self.assertEqual(energy, 1.2-1j)
        self.assertTrue(status)

        # Get a term that needs to invert.
        hop_dict.add_hopping(rn=(-1, -1, 2), orb_i=1, orb_j=2, energy=1.2+2j)
        energy, status = hop_dict.get_hopping((1, 1, -2), 2, 1)
        self.assertEqual(energy, 1.2-2j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping((-1, -1, 2), 1, 2)
        self.assertEqual(energy, 1.2+2j)
        self.assertTrue(status)

        # Get a normal term in (0, 0, 0) cell.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.2+2j)
        energy, status = hop_dict.get_hopping((0, 0, 0), 1, 2)
        self.assertEqual(energy, 1.2+2j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping((0, 0, 0), 2, 1)
        self.assertEqual(energy, 1.2-2j)
        self.assertTrue(status)

        # Get a term in (0, 0, 0) cell that needs to invert.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=3, orb_j=1, energy=1.2+2j)
        energy, status = hop_dict.get_hopping((0, 0, 0), 1, 3)
        self.assertEqual(energy, 1.2-2j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping((0, 0, 0), 3, 1)
        self.assertEqual(energy, 1.2+2j)
        self.assertTrue(status)

        # Get a non-existing term
        energy, status = hop_dict.get_hopping((1, 1, 0), 1, 5)
        self.assertEqual(energy, None)
        self.assertFalse(status)
        energy, status = hop_dict.get_hopping((0, 0, 0), 1, 5)
        self.assertEqual(energy, None)
        self.assertFalse(status)

    def test_get_hopping_inter(self):
        """
        Test if 'get_hopping' method of 'InterHopping' works as expected.

        :return: None
        """
        hop_dict = InterHopping(None, None)

        # Void case
        energy, status = hop_dict.get_hopping((1, -1, 2), 1, 2)
        self.assertEqual(energy, None)
        self.assertFalse(status)

        # Get a normal term that does not need auto-complete
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        energy, status = hop_dict.get_hopping((1, -1, 2), 1, 2)
        self.assertEqual(energy, 1.2+1j)
        self.assertTrue(status)

        # Overwriting
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.5+1j)
        energy, status = hop_dict.get_hopping((1, -1, 2), 1, 2)
        self.assertEqual(energy, 1.5+1j)
        self.assertTrue(status)

        # Get an opposite hopping term (not conjugate)
        hop_dict.add_hopping(rn=(-1, 1, -2), orb_i=2, orb_j=1, energy=1.5+2j)
        energy, status = hop_dict.get_hopping((-1, 1, -2), 2, 1)
        self.assertEqual(energy, 1.5+2j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping((1, -1, 2), 1, 2)
        self.assertEqual(energy, 1.5+1j)
        self.assertTrue(status)

        # Get a normal term that needs auto-complete
        hop_dict.add_hopping(rn=(-1, 1), orb_i=1, orb_j=1, energy=1.5+2j)
        energy, status = hop_dict.get_hopping((-1, 1, 0), 1, 1)
        self.assertEqual(energy, 1.5+2j)
        self.assertTrue(status)

        # Get a non-existing term
        energy, status = hop_dict.get_hopping((1, 1, 0), 1, 5)
        self.assertEqual(energy, None)
        self.assertFalse(status)
        energy, status = hop_dict.get_hopping((0, 0, 0), 1, 5)
        self.assertEqual(energy, None)
        self.assertFalse(status)

    def test_hopping_speed_intra(self):
        """
        Test the performance of 'add_hopping' and 'get_hopping' methods of
        'IntraHopping'.

        :return: None
        """
        hop_dict = IntraHopping()
        timer = tb.Timer()
        timer.tic("add_hopping_intra")
        for i in range(-500, 500):
            for j in range(-500, 500):
                for k1 in range(5):
                    for k2 in range(5):
                        hop_dict.add_hopping((i, j, 0), k1, k2, 0.5)
        timer.toc("add_hopping_intra")
        timer.tic("get_hopping_intra")
        for i in range(-500, 500):
            for j in range(-500, 500):
                for k1 in range(5):
                    for k2 in range(5):
                        hop_dict.get_hopping((i, j, 0), k1, k2)
        timer.toc("get_hopping_intra")
        timer.report_total_time()

    def test_hopping_speed_inter(self):
        """
        Test the performance of 'add_hopping' and 'get_hopping' methods of
        'InterHopping'.

        :return: None
        """
        hop_dict = IntraHopping()
        timer = tb.Timer()
        timer.tic("add_hopping_inter")
        for i in range(-500, 500):
            for j in range(-500, 500):
                for k1 in range(5):
                    for k2 in range(5):
                        hop_dict.add_hopping((i, j, 0), k1, k2, 0.5)
        timer.toc("add_hopping_inter")
        timer.tic("get_hopping_inter")
        for i in range(-500, 500):
            for j in range(-500, 500):
                for k1 in range(5):
                    for k2 in range(5):
                        hop_dict.get_hopping((i, j, 0), k1, k2)
        timer.toc("get_hopping_inter")
        timer.report_total_time()

    def test_remove_hopping_intra(self):
        """
        Test the 'remove_hoppings' method of 'IntraHopping'.

        :return: None
        """
        hop_dict = IntraHopping()

        # Void case
        status = hop_dict.remove_hopping(rn=(1, -1, 2), orb_i=2, orb_j=2)
        self.assertFalse(status)

        # Remove the term directly
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        hop_dict.remove_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2)
        self.assertEqual(hop_dict.num_hop, 0)

        # Remove the conjugate term
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        hop_dict.remove_hopping(rn=(-1, 1, -2), orb_i=2, orb_j=1)
        self.assertEqual(hop_dict.num_hop, 0)

        # Remove a non-existing term
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        status = hop_dict.remove_hopping(rn=(1, -1, 2), orb_i=2, orb_j=2)
        self.assertFalse(status)
        status = hop_dict.remove_hopping(rn=(0, -1, 2), orb_i=2, orb_j=2)
        self.assertFalse(status)

    def test_remove_hopping_inter(self):
        """
        Test the 'remove_hoppings' method of 'InterHopping'.

        :return: None
        """
        hop_dict = InterHopping(None, None)

        # Void case
        status = hop_dict.remove_hopping(rn=(1, -1, 2), orb_i=2, orb_j=2)
        self.assertFalse(status)

        # Remove the term directly
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        hop_dict.remove_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2)
        self.assertEqual(hop_dict.num_hop, 0)

        # Remove the opposite term (not conjugate)
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        status = hop_dict.remove_hopping(rn=(-1, 1, -2), orb_i=2, orb_j=1)
        self.assertFalse(status)

        # Remove a non-existing term
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        status = hop_dict.remove_hopping(rn=(1, -1, 2), orb_i=2, orb_j=2)
        self.assertFalse(status)
        status = hop_dict.remove_hopping(rn=(0, -1, 2), orb_i=2, orb_j=2)
        self.assertFalse(status)

    def test_remove_orbitals(self):
        """
        Test the 'remove_orbitals' method of 'InterHopping'.

        :return: None.
        """
        hop_dict = IntraHopping()

        # Void case
        hop_dict.remove_orbitals([0, 2])
        self.assertEqual(hop_dict.num_hop, 0)

        # Normal case
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=0, orb_j=1, energy=1.0-1j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.0-3j)
        hop_dict.add_hopping(rn=(-1, 0, 0), orb_i=1, orb_j=2, energy=1.2-1j)
        hop_dict.add_hopping(rn=(-1, 0, 0), orb_i=1, orb_j=3, energy=1.2-3j)
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        hop_dict.add_hopping(rn=(1, 1, -2), orb_i=1, orb_j=2, energy=1.2+3j)
        hop_dict.remove_orbitals([0, 2])

        energy, status = hop_dict.get_hopping(rn=(0, 0, 0), orb_i=0, orb_j=1)
        self.assertEqual(energy, None)
        self.assertFalse(status)
        energy, status = hop_dict.get_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2)
        self.assertEqual(energy, None)
        self.assertFalse(status)
        energy, status = hop_dict.get_hopping(rn=(-1, 0, 0), orb_i=1, orb_j=2)
        self.assertEqual(energy, None)
        self.assertFalse(status)
        energy, status = hop_dict.get_hopping(rn=(-1, 0, 0), orb_i=0, orb_j=1)
        self.assertEqual(energy, 1.2-3j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2)
        self.assertEqual(energy, None)
        self.assertFalse(status)
        energy, status = hop_dict.get_hopping(rn=(1, 1, -2), orb_i=1, orb_j=2)
        self.assertEqual(energy, None)
        self.assertFalse(status)

    def test_remove_rn(self):
        """
        Test the 'remove_rn' method of 'IntraHopping'.

        :return: None
        """
        hop_dict = IntraHopping()

        # Void case
        self.assertFalse(hop_dict.remove_rn(rn=(0, 0, 0)))

        # Normal case
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=0, orb_j=1, energy=1.0-1j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.0-3j)
        hop_dict.add_hopping(rn=(-1, 0, 0), orb_i=1, orb_j=2, energy=1.2-1j)
        hop_dict.add_hopping(rn=(-1, 0, 0), orb_i=1, orb_j=3, energy=1.2-3j)
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        hop_dict.add_hopping(rn=(1, 1, -2), orb_i=1, orb_j=2, energy=1.2+3j)

        # Original rn
        self.assertTrue(hop_dict.remove_rn(rn=(0, 0, 0)))

        # Conjugate rn
        self.assertTrue(hop_dict.remove_rn(rn=(1, 0, 0)))

        # Non-existing rn
        self.assertFalse(hop_dict.remove_rn(rn=(2, 0, 0)))

    def test_purge(self):
        """
        Test the 'purge' and 'to_list' methods of 'IntraHopping'.

        :return: None
        """
        hop_dict = IntraHopping()

        # Void case
        hop_dict.purge()
        self.assertEqual(hop_dict.num_hop, 0)

        # Normal case
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=0, orb_j=1, energy=1.0-1j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.0-3j)
        hop_dict.add_hopping(rn=(-1, 0, 0), orb_i=1, orb_j=2, energy=1.2-1j)
        hop_dict.add_hopping(rn=(-1, 0, 0), orb_i=1, orb_j=3, energy=1.2-3j)
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        hop_dict.add_hopping(rn=(1, 1, -2), orb_i=1, orb_j=2, energy=1.2+3j)
        hop_dict.remove_hopping(rn=(0, 0, 0), orb_i=0, orb_j=1)
        hop_dict.remove_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2)
        hop_dict.purge()
        ref_list = [(1, 0, 0, 2, 1, 1.2+1j), (1, 0, 0, 3, 1, 1.2+3j),
                    (1, -1, 2, 1, 2, 1.2+1j), (1, 1, -2, 1, 2, 1.2+3j)]
        self.assertListEqual(ref_list, hop_dict.to_list())

    def test_to_array(self):
        """
        Test the 'to_array' and 'to_list' methods of 'IntraHopping'.

        :return: None
        """
        hop_dict = IntraHopping()
        th = tb.TestHelper(self)

        # Void case
        self.assertListEqual(hop_dict.to_list(), [])
        hop_ind, hop_eng = hop_dict.to_array(use_int64=False)
        th.test_equal_array(hop_ind, np.array([]))
        th.test_equal_array(hop_eng, np.array([]))
        hop_ind, hop_eng = hop_dict.to_array(use_int64=True)
        th.test_equal_array(hop_ind, np.array([]))
        th.test_equal_array(hop_eng, np.array([]))

        # Normal case
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        hop_dict.add_hopping(rn=(1, 1, -2), orb_i=1, orb_j=2, energy=1.2+3j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.2-1j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=3, energy=1.2-3j)
        hop_ind, hop_eng = hop_dict.to_array()
        hop_ind_ref = np.array([(1, -1, 2, 1, 2), (1, 1, -2, 1, 2),
                                (0, 0, 0, 1, 2), (0, 0, 0, 1, 3)])
        hop_eng_ref = np.array([1.2+1j, 1.2+3j, 1.2-1j, 1.2-3j])
        th.test_equal_array(hop_ind, hop_ind_ref)
        th.test_equal_array(hop_eng, hop_eng_ref)

    def test_hop_dict(self):
        """Test if 'HopDict' class works as expected."""
        cell = tb.make_graphene_diamond()
        hop_dict = tb.HopDict(cell.num_orb)
        th = tb.TestHelper(self)

        # Test initialization
        self.assertDictEqual(hop_dict.hoppings, dict())
        self.assertEqual(hop_dict.num_orb, cell.num_orb)
        self.assertEqual(hop_dict.mat_shape, (cell.num_orb, cell.num_orb))

        # Test setting matrix
        # Exception handling
        def _test():
            hop_dict[(2, 2, -1, 3)] = np.zeros(hop_dict.mat_shape)
        th.test_raise(_test, exc.CoordLenError, r"length of cell index .+ not "
                                                r"in \(2, 3\)")

        def _test():
            hop_dict[(2, 2, -1)] = np.zeros((3, 3))
        th.test_raise(_test, ValueError, r"Shape of hopping matrix .+ does not "
                                         r"match .+")

        def _test():
            hop_dict[(0, 0, 0)] = np.eye(hop_dict.mat_shape[0])
        th.test_raise(_test, exc.PCHopDiagonalError, r"hopping term .+ is "
                                                     r"diagonal")

        # Normal case
        # Set hopping matrix directly
        hop_mat0 = np.array([[0.0, 1-1.2j], [1+1.2j, 0.0]])
        hop_mat1 = np.array([[1.0, 1-0.5j], [1+0.2j, 2.0]])

        # = for non-existing hop_mat
        hop_dict[(0, 0)] = hop_mat0
        th.test_equal_array(hop_dict[(0, 0, 0)], hop_mat0)
        hop_dict[(0, 1)] = hop_mat1
        th.test_equal_array(hop_dict[(0, 1, 0)], hop_mat1)

        # += for existing hop_mat
        hop_dict[(0, 1)] += hop_mat0
        th.test_equal_array(hop_dict[(0, 1, 0)], hop_mat1+hop_mat0)

        # += for non-existing hop_mat
        hop_dict[(0, 2)] += hop_mat0
        th.test_equal_array(hop_dict[(0, 2, 0)], hop_mat0)

        # Set hopping matrix elements
        hop_dict[(0, 0)] = hop_mat0
        hop_dict[(0, 1)] = hop_mat1
        hop_dict[(0, 2)] = np.zeros(hop_dict.mat_shape, dtype=complex)

        # = for existing terms
        hop_dict[(0, 0)][0, 1] = 1+1.3j
        self.assertEqual(hop_dict[(0, 0)][0, 1], 1+1.3j)
        hop_dict[(0, 1)][1, 1] = 1+1.5j
        self.assertEqual(hop_dict[(0, 1)][1, 1], 1+1.5j)

        # = for non-existing terms
        hop_dict[(0, -1)][0, 1] = 0.5-1.5j
        self.assertEqual(hop_dict[(0, -1, 0)][0, 1], 0.5-1.5j)
        hop_dict[(1, 0)][0, 1] = 1.2j
        self.assertEqual(hop_dict[(1, 0, 0)][0, 1], 1.2j)

        # += for existing terms
        hop_dict[(0, 0)][0, 1] += 0.3j
        self.assertEqual(hop_dict[(0, 0)][0, 1], 1+1.6j)
        hop_dict[(0, 1)][1, 1] += 1
        self.assertEqual(hop_dict[(0, 1)][1, 1], 2+1.5j)

        # += for non-existing terms
        hop_dict[(0, 2)][0, 1] += 0.5-1.5j
        self.assertEqual(hop_dict[(0, 2, 0)][0, 1], 0.5-1.5j)
        hop_dict[(2, 0)][0, 1] += 1.2j
        self.assertEqual(hop_dict[(2, 0, 0)][0, 1], 1.2j)

    def test_add_hopping_dict(self):
        """
        Test function 'add_hopping_dict' of 'PrimitiveCell' class.

        :return: None
        """
        cell = tb.make_graphene_diamond()
        hop_dict = tb.HopDict(cell.num_orb)
        hop_mat_00 = np.array([[0.0, -2.5], [-2.5, 0.0]])
        hop_mat_10 = np.array([[1.2, -2.6], [-2.3, 1.1]])
        hop_mat_01 = np.array([[1.6, -2.8], [-2.7, 1.2]])
        hop_dict[(0, 0)] = hop_mat_00
        hop_dict[(1, 0)] = hop_mat_10
        hop_dict[(0, 1)] = hop_mat_01
        cell.add_hopping_dict(hop_dict)
        self.assertEqual(cell.get_hopping((0, 0), 0, 1), -2.5)
        self.assertEqual(cell.get_hopping((1, 0), 0, 0), 1.2)
        self.assertEqual(cell.get_hopping((1, 0), 1, 1), 1.1)
        self.assertEqual(cell.get_hopping((1, 0), 0, 1), -2.6)
        self.assertEqual(cell.get_hopping((1, 0), 1, 0), -2.3)
        self.assertEqual(cell.get_hopping((0, 1), 0, 0), 1.6)
        self.assertEqual(cell.get_hopping((0, 1), 1, 1), 1.2)
        self.assertEqual(cell.get_hopping((0, 1), 0, 1), -2.8)
        self.assertEqual(cell.get_hopping((0, 1), 1, 0), -2.7)


if __name__ == "__main__":
    unittest.main()
