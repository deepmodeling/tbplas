#! /usr/bin/env python

import unittest

import numpy as np

from tbplas import Timer, TestHelper
from tbplas.builder.base import invert_rn, IntraHopping


class TestHop(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_invert_rn(self):
        """Test if function 'invert_rn' works as expected."""
        test_dict = {
            (1, -1, 2): False, (-1, -1, 2): True,
            (0, 1, 2): False, (0, -1, 2): True,
            (0, 0, 1): False, (0, 0, 0): False, (0, 0, -1): True
        }
        for key, value in test_dict.items():
            self.assertEqual(invert_rn(key), value)

    def test_invert_rn_speed(self):
        """Test the speed of function 'invert_rn'."""
        timer = Timer()
        timer.tic("invert_rn")
        for i in range(-100, 100):
            for j in range(-100, 100):
                for k in range(-100, 100):
                    invert_rn((i, j, k))
        timer.toc("invert_rn")
        timer.report_total_time()

    def test_add_hopping(self):
        """Test if 'add_hopping' method works as expected."""
        hop_dict = IntraHopping()

        # Add a normal term that does not need to invert.
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        self.assertEqual(hop_dict.dict[(1, -1, 2)][(1, 2)], 1.2+1j)
        # Overwriting.
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.5+1j)
        self.assertEqual(hop_dict.dict[(1, -1, 2)][(1, 2)], 1.5+1j)
        hop_dict.add_hopping(rn=(-1, 1, -2), orb_i=2, orb_j=1, energy=1.5+2j)
        self.assertEqual(hop_dict.dict[(1, -1, 2)][(1, 2)], 1.5-2j)

        # Add a term that needs to invert.
        hop_dict.add_hopping(rn=(-1, -1, 2), orb_i=1, orb_j=2, energy=1.2+2j)
        self.assertEqual(hop_dict.dict[(1, 1, -2)][(2, 1)], 1.2-2j)
        # Overwriting.
        hop_dict.add_hopping(rn=(-1, -1, 2), orb_i=1, orb_j=2, energy=1.2+5j)
        self.assertEqual(hop_dict.dict[(1, 1, -2)][(2, 1)], 1.2-5j)
        hop_dict.add_hopping(rn=(1, 1, -2), orb_i=2, orb_j=1, energy=1.2+7j)
        self.assertEqual(hop_dict.dict[(1, 1, -2)][(2, 1)], 1.2+7j)

        # Add a normal term in (0, 0, 0) cell.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.2+2j)
        self.assertEqual(hop_dict.dict[(0, 0, 0)][(1, 2)], 1.2+2j)
        # Overwriting.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.2+3j)
        self.assertEqual(hop_dict.dict[(0, 0, 0)][(1, 2)], 1.2+3j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=2, orb_j=1, energy=1.2-5j)
        self.assertEqual(hop_dict.dict[(0, 0, 0)][(1, 2)], 1.2+5j)

        # Add a term in (0, 0, 0) cell that needs to invert.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=3, orb_j=1, energy=1.2+2j)
        self.assertEqual(hop_dict.dict[(0, 0, 0)][(1, 3)], 1.2-2j)
        # Overwriting.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=3, orb_j=1, energy=1.2+1j)
        self.assertEqual(hop_dict.dict[(0, 0, 0)][(1, 3)], 1.2-1j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=3, energy=1.2+5j)
        self.assertEqual(hop_dict.dict[(0, 0, 0)][(1, 3)], 1.2+5j)

    def test_get_hopping(self):
        """Test if 'get_hopping' method works as expected."""
        hop_dict = IntraHopping()

        # Add a normal term that does not need to invert.
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        energy, status = hop_dict.get_hopping((1, -1, 2), 1, 2)
        self.assertEqual(energy, 1.2+1j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping((-1, 1, -2), 2, 1)
        self.assertEqual(energy, 1.2-1j)
        self.assertTrue(status)

        # Add a term that needs to invert.
        hop_dict.add_hopping(rn=(-1, -1, 2), orb_i=1, orb_j=2, energy=1.2+2j)
        energy, status = hop_dict.get_hopping((1, 1, -2), 2, 1)
        self.assertEqual(energy, 1.2-2j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping((-1, -1, 2), 1, 2)
        self.assertEqual(energy, 1.2+2j)
        self.assertTrue(status)

        # Add a normal term in (0, 0, 0) cell.
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.2+2j)
        energy, status = hop_dict.get_hopping((0, 0, 0), 1, 2)
        self.assertEqual(energy, 1.2+2j)
        self.assertTrue(status)
        energy, status = hop_dict.get_hopping((0, 0, 0), 2, 1)
        self.assertEqual(energy, 1.2-2j)
        self.assertTrue(status)

        # Add a term in (0, 0, 0) cell that needs to invert.
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

    def test_to_array(self):
        """Test if 'to_array' method works as expected."""
        hop_dict = IntraHopping()
        hop_dict.add_hopping(rn=(1, -1, 2), orb_i=1, orb_j=2, energy=1.2+1j)
        hop_dict.add_hopping(rn=(1, 1, -2), orb_i=1, orb_j=2, energy=1.2+3j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=2, energy=1.2-1j)
        hop_dict.add_hopping(rn=(0, 0, 0), orb_i=1, orb_j=3, energy=1.2-3j)

        hop_ind, hop_eng = hop_dict.to_array()
        hop_ind_ref = np.array([(1, -1, 2, 1, 2), (1, 1, -2, 1, 2),
                                (0, 0, 0, 1, 2), (0, 0, 0, 1, 3)])
        hop_eng_ref = np.array([1.2+1j, 1.2+3j, 1.2-1j, 1.2-3j])

        th = TestHelper(self)
        th.test_equal_array(hop_ind, hop_ind_ref)
        th.test_equal_array(hop_eng, hop_eng_ref)

    def test_hop_dict_speed(self):
        """Test the performance of 'add_hopping' and 'get_hopping' method."""
        hop_dict = IntraHopping()
        timer = Timer()
        timer.tic("add_hopping")
        for i in range(-500, 500):
            for j in range(-500, 500):
                for k1 in range(5):
                    for k2 in range(5):
                        hop_dict.add_hopping((i, j, 0), k1, k2, 0.5)
        timer.toc("add_hopping")
        timer.tic("get_hopping")
        for i in range(-500, 500):
            for j in range(-500, 500):
                for k1 in range(5):
                    for k2 in range(5):
                        hop_dict.get_hopping((i, j, 0), k1, k2)
        timer.toc("get_hopping")
        timer.report_total_time()


if __name__ == "__main__":
    unittest.main()
