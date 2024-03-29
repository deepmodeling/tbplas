#! /usr/bin/env python

import unittest

import numpy as np

from tbplas.cython import super as core

from tbplas import (gen_lattice_vectors, PrimitiveCell, SuperCell, TestHelper)
import tbplas.builder.exceptions as exc
from tbplas.builder.super import OrbitalSet


def make_cell():
    vectors = gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
    cell = PrimitiveCell(vectors)
    cell.add_orbital((0.0, 0.0), 0.0)
    cell.add_orbital((1. / 3, 1. / 3), 0.0)
    cell.add_hopping((0, 0), 0, 1, -2.7)
    cell.add_hopping((1, 0), 1, 0, -2.7)
    cell.add_hopping((0, 1), 1, 0, -2.7)
    return cell


def make_cell_orb():
    vectors = gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
    cell = PrimitiveCell(vectors)
    cell.add_orbital((0.0, 0.0), 0.0, label="C_pz")
    cell.add_orbital((1. / 3, 1. / 3), 0.0, label="C_pz")
    return cell


def make_cell_empty():
    vectors = gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
    cell = PrimitiveCell(vectors)
    return cell


class MyTest(unittest.TestCase):

    def test_init_illegal(self):
        """
        Test if exception handling of OrbitalSet.__init__ works as expected.

        :return: None.
        """
        th = TestHelper(self)
        cell = make_cell()

        # PCLockError
        def _test():
            orb_set = OrbitalSet(cell, dim=(3, 3, 1))
            orb_set._prim_cell.add_orbital((0.0, 0.0), 1.5)
        th.test_raise(_test, exc.LockError, r"trying to modify a locked object")

        # DimLenError
        def _test():
            OrbitalSet(cell, dim=(1, 1, 1, 2))
        th.test_raise(_test, exc.SCDimLenError,
                      r"length of supercell dimension .+ not in .+")

        # DimSizeError
        msg_dim = r"^dimension on direction [0-2] should" \
                  r" be no less than [0-9]+"
        for dim in ((-1, 1, 1), (2, -1, 1), (2, 1, -1)):
            def _test():
                OrbitalSet(cell, dim=dim)
            th.test_raise(_test, exc.SCDimSizeError, msg_dim)

        # PBCLenError
        def _test():
            OrbitalSet(cell, dim=(3, 3, 1), pbc=(True, False, True, False))
        th.test_raise(_test, exc.PBCLenError,
                      r"length of pbc .+ not in \(2, 3\)")

    def test_init(self):
        """
        Test if OrbitalSet.__init__ works as expected for legal input.

        :return: None.
        """
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        self.assertTupleEqual(tuple(orb_set._dim), (3, 3, 1))
        self.assertTupleEqual(tuple(orb_set._pbc), (False, False, False))
        self.assertEqual(orb_set._vacancy_set, set())
        self.assertEqual(orb_set._vac_id_sc.size, 0)
        self.assertEqual(orb_set._orb_id_pc.shape, (18, 4))

    def test_check_id_pc_illegal(self):
        """
        Check if error handling of OrbitalSet.check_id_pc works as expected.

        :return: None.
        """
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))

        msg_len = r"length of id_pc .+ is not 4"
        msg_cell = r"cell index .+ of id_pc .+ out of range"
        msg_orb = r"orbital index .+ of id_pc .+ out of range"
        msg_type = r"illegal type .+ of id_pc"

        # IDPCLenError for tuple
        with self.assertRaises(exc.IDPCLenError) as cm:
            orb_set._check_id_pc((1, 2, 3))
        self.assertRegex(str(cm.exception), msg_len)

        # IDPCIndexError for tuple
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set._check_id_pc((3, 0, 0, 0))
        self.assertRegex(str(cm.exception), msg_cell)
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set._check_id_pc((0, -1, 0, 0))
        self.assertRegex(str(cm.exception), msg_cell)
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set._check_id_pc((0, 0, 5, 0))
        self.assertRegex(str(cm.exception), msg_cell)
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set._check_id_pc((0, 0, 0, 5))
        self.assertRegex(str(cm.exception), msg_orb)

        # IDPCLenError for array
        with self.assertRaises(exc.IDPCLenError) as cm:
            orb_set._check_id_pc(np.array((1, 2, 3)))
        self.assertRegex(str(cm.exception), msg_len)

        # IDPCIndexError for array
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set._check_id_pc(np.array((3, 0, 0, 0)))
        self.assertRegex(str(cm.exception), msg_cell)
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set._check_id_pc(np.array((0, -1, 0, 0)))
        self.assertRegex(str(cm.exception), msg_cell)
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set._check_id_pc(np.array((0, 0, 5, 0)))
        self.assertRegex(str(cm.exception), msg_cell)
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set._check_id_pc(np.array((0, 0, 0, 5)))
        self.assertRegex(str(cm.exception), msg_orb)

        # TDPCTypeError
        with self.assertRaises(exc.IDPCTypeError) as cm:
            orb_set._check_id_pc([0, 0, 0, 0])
        self.assertRegex(str(cm.exception), msg_type)

    def test_check_id_pc(self):
        """
        Tests if OrbitalSet.check_id_pc works as expected for legal input.
        :return:
        """
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        th = TestHelper(self)

        def _test():
            orb_set._check_id_pc((1, 0, 0, 1))
        th.test_no_raise(_test, exc.IDPCLenError)
        th.test_no_raise(_test, exc.IDPCIndexError)
        th.test_no_raise(_test, exc.IDPCTypeError)

        def _test():
            orb_set._check_id_pc(np.array((1, 0, 0, 1)))
        th.test_no_raise(_test, exc.IDPCLenError)
        th.test_no_raise(_test, exc.IDPCIndexError)
        th.test_no_raise(_test, exc.IDPCTypeError)

    def test_set_vacancies_illegal(self):
        """
        Test if error handling of OrbitalSet.set_vacancies works as expected.

        :return: None.
        """
        # Feed a list of vacancies with an illegal one.
        # Errors should be raised
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        vacancies = [(0, 0, 0, 1), (3, 0, 0, 1)]

        # VacIDPCLenError
        with self.assertRaises(exc.IDPCLenError) as cm:
            orb_set.set_vacancies([(0, 1, 0), (0, 0, 1, 0, 0)])
        self.assertRegex(str(cm.exception),
                         r"length of id_pc .+ is not 4")

        # VacIDPCIndexError
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set.set_vacancies(vacancies)
        self.assertRegex(str(cm.exception),
                         r"cell index .+ of id_pc .+ out of range")

    def test_set_vacancies(self):
        """
        Test if OrbitalSet.set_vacancies works as expected for legal input.

        :return: None.
        """
        th = TestHelper(self)
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        vac_id_sc = np.array([1, 2, 9])
        orb_id_pc = [(i_a, i_b, 0, i_o)
                     for i_a in range(3)
                     for i_b in range(3)
                     for i_o in range(2)]
        for vac in vacancies:
            orb_id_pc.remove(vac)
        orb_id_pc = np.array(orb_id_pc)

        # First we feed some effective input to set_vacancies.
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        orb_set.set_vacancies(vacancies)
        orb_set.sync_array()
        self.assertSetEqual(orb_set._vacancy_set, set(vacancies))
        th.test_equal_array(orb_set._vac_id_sc, vac_id_sc)
        th.test_equal_array(orb_set._orb_id_pc, orb_id_pc)

        # Then we feed a blank to set_vacancies.
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        orb_set.set_vacancies([])
        orb_set.sync_array()
        self.assertSetEqual(orb_set._vacancy_set, set())
        self.assertEqual(orb_set._vac_id_sc.size, 0)
        self.assertEqual(orb_set._orb_id_pc.shape, (18, 4))

        # Finally, we feed a list of vacancies with redundant items.
        # These redundant items should be detected and removed.
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        vacancies = [(0, 0, 0, 1), (1, 0, 0, 0), (0, 0, 0, 1)]
        orb_set.set_vacancies(vacancies)
        reduced_vacancies = [(0, 0, 0, 1), (1, 0, 0, 0)]
        self.assertSetEqual(orb_set._vacancy_set, set(reduced_vacancies))

    def test_orb_id_sc2pc(self):
        """
        Test if OrbitalSet.orb_id_sc2pc works as expected.

        :return: None.
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        orb_set.set_vacancies(vacancies)
        th = TestHelper(self)

        # Feed a wrong input and check if errors are raised.
        with self.assertRaises(exc.IDSCIndexError) as cm:
            orb_set.orb_id_sc2pc(21)
        self.assertRegex(str(cm.exception), r"id_sc .+ out of range")

        # Then test the normal case
        for i in range(orb_set.num_orb_sc):
            th.test_equal_array(orb_set.orb_id_sc2pc(i), orb_set._orb_id_pc[i])

    def test_orb_id_pc2sc(self):
        """
        Test if OrbitalSet.orb_id_pc2sc works as expected.

        :return: None.
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        orb_set.set_vacancies(vacancies)

        # Test if the function can detect errors
        with self.assertRaises(exc.IDPCVacError) as cm:
            orb_set.orb_id_pc2sc((1, 1, 0, 1))
        self.assertRegex(str(cm.exception),
                         r"orbital id_pc .+ seems to be a vacancy")

        # Then test the normal case
        for i in range(orb_set.num_orb_sc):
            self.assertEqual(orb_set.orb_id_pc2sc(orb_set._orb_id_pc[i]), i)

    def test_orb_id_sc2pc_array(self):
        """
        Test if OrbitalSet.orb_od_sc2pc_array works as expected.

        :return: None
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        orb_set.set_vacancies(vacancies)
        th = TestHelper(self)

        # Error handling
        with self.assertRaises(exc.IDSCIndexError) as cm:
            orb_set.orb_id_sc2pc_array(np.array([-1, 18], dtype=np.int64))
        self.assertRegex(str(cm.exception), r"id_sc .+ out of range")

        # Normal case
        id_sc = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                            dtype=np.int64)
        id_pc = orb_set.orb_id_sc2pc_array(id_sc)
        th.test_equal_array(id_pc, orb_set._orb_id_pc)

    def test_orb_id_pc2sc_array(self):
        """
        Test if OrbitalSet.orb_od_pc2sc_array works as expected.

        :return: None
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(make_cell(), dim=(3, 3, 1))
        orb_set.set_vacancies(vacancies)
        th = TestHelper(self)

        # Error handling
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set.orb_id_pc2sc_array(np.array([[-1, 0, 0, 1]],
                                                dtype=np.int32))
        self.assertRegex(str(cm.exception), r"cell index .+ out of range")
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set.orb_id_pc2sc_array(np.array([[0, 5, 0, 1]],
                                                dtype=np.int32))
        self.assertRegex(str(cm.exception), r"cell index .+ out of range")
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set.orb_id_pc2sc_array(np.array([[0, 0, -3, 1]],
                                                dtype=np.int32))
        self.assertRegex(str(cm.exception), r"cell index .+ out of range")
        with self.assertRaises(exc.IDPCIndexError) as cm:
            orb_set.orb_id_pc2sc_array(np.array([[0, 0, 0, 3]],
                                                dtype=np.int32))
        self.assertRegex(str(cm.exception), r"orbital index .+ out of range")
        with self.assertRaises(exc.IDPCVacError) as cm:
            orb_set.orb_id_pc2sc_array(np.array([[0, 0, 0, 1]],
                                                dtype=np.int32))
        self.assertRegex(str(cm.exception), r"orbital id_pc .+ seems to be"
                                            r" a vacancy")

        # Normal case
        id_sc_test = orb_set.orb_id_pc2sc_array(orb_set._orb_id_pc)
        id_sc_ref = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                                dtype=np.int64)
        th.test_equal_array(id_sc_test, id_sc_ref)

    def test_sc_properties(self):
        """
        Test methods of SuperCell decorated with '@property'.

        :return: None
        """
        sc = SuperCell(make_cell(), dim=(3, 3, 1))
        th = TestHelper(self)
        self.assertEqual(sc.num_orb_pc, sc.prim_cell.num_orb)
        self.assertEqual(sc.num_orb_sc, sc._orb_id_pc.shape[0])
        self.assertEqual(sc.num_vac, 0)
        th.test_equal_array(sc.pc_lat_vec, sc.prim_cell.lat_vec, almost=True)
        th.test_equal_array(sc.pc_orb_pos, sc.prim_cell.orb_pos, almost=True)
        th.test_equal_array(sc.pc_orb_eng, sc.prim_cell.orb_eng, almost=True)
        th.test_equal_array(sc.pc_hop_ind, sc.prim_cell.hop_ind)
        th.test_equal_array(sc.pc_hop_eng, sc.prim_cell.hop_eng, almost=True)
        sc_lat_vec_ref = sc.prim_cell.lat_vec.copy()
        sc_lat_vec_ref[0] *= 3
        sc_lat_vec_ref[1] *= 3
        th.test_equal_array(sc.sc_lat_vec, sc_lat_vec_ref)

    def test_get_orb_eng(self):
        """
        Test if SuperCell.get_orb_eng works as expected.

        :return: None
        """
        sc = SuperCell(make_cell(), dim=(3, 3, 1))
        orb_eng = sc.get_orb_eng()
        self.assertTupleEqual(orb_eng.shape, (18,))

    def test_get_orb_pos(self):
        """
        Test if SuperCell.get_orb_pos works as expected.

        :return: None
        """
        sc = SuperCell(make_cell(), dim=(3, 3, 1))
        orb_pos = sc.get_orb_pos()
        self.assertTupleEqual(orb_pos.shape, (18, 3))

    def test_get_hop_dr(self):
        """
        Test if SuperCell.get_hop and SuperCell.get_dr work as expected.

        :return: None
        """
        # Test hop_modifier
        sc = SuperCell(make_cell(), dim=(3, 3, 1))

        id_pc_bra = [
            (0, 0, 0, 0),  # same item
            (1, 1, 0, 1),  # conjugate item
            (0, 0, 0, 1),  # new term
            (0, 2, 0, 0),  # new term
        ]
        id_pc_ket = [
            (0, 0, 0, 1),  # same item
            (1, 1, 0, 0),  # conjugate item
            (1, 2, 0, 0),  # new term
            (2, 0, 0, 1),  # new term
        ]
        id_sc_bra = sc.orb_id_pc2sc_array(np.array(id_pc_bra))
        id_sc_ket = sc.orb_id_pc2sc_array(np.array(id_pc_ket))
        for i in range(id_sc_bra.shape[0]):
            sc.add_hopping((0, 0, 0), id_sc_bra[i], id_sc_ket[i], energy=0.0)

        # Inspect the arrays
        hop_i, hop_j, hop_v, dr = sc.get_hop()
        self.assertEqual(hop_i.shape[0], dr.shape[0])

    def test_fast_algo(self):
        """
        Test the fast algorithm to build hopping terms.

        :return: None
        """
        # Test the core function to split hopping terms
        pc_hop_ind = np.array([
            (0, 0, 0, 1, 0),
            (0, 0, 0, 1, 1),
            (1, 0, 0, 1, 0),
            (0, -1, 0, 1, 1),
            (0, 0, 2, 1, 1)], dtype=np.int32
        )
        pc_hop_eng = np.array([1.0, -1.0, 2.5, 1.3, 0.6], dtype=np.complex128)
        pbc = np.array([0, 1, 0], dtype=np.int32)

        ind_pbc, eng_pbc, ind_free, eng_free = \
            core.split_pc_hop(pc_hop_ind, pc_hop_eng, pbc)
        ind_pbc_ref = pc_hop_ind[[0, 1, 3]]
        eng_pbc_ref = pc_hop_eng[[0, 1, 3]]
        ind_free_ref = pc_hop_ind[[2, 4]]
        eng_free_ref = pc_hop_eng[[2, 4]]

        th = TestHelper(self)
        th.test_equal_array(ind_pbc, ind_pbc_ref)
        th.test_equal_array(eng_pbc, eng_pbc_ref)
        th.test_equal_array(ind_free, ind_free_ref)
        th.test_equal_array(eng_free, eng_free_ref)

    def test_void(self):
        """
        Test if SuperCell behaves well under void primitive cells.

        :return: None
        """
        th = TestHelper(self)

        # Create supercell with empty primitive cell
        def _test():
            SuperCell(make_cell_orb(), dim=(3, 3, 1))
        th.test_raise(_test, exc.PCHopEmptyError,
                      r"primitive cell has no hopping terms")

        def _test():
            SuperCell(make_cell_empty(), dim=(3, 3, 1))
        th.test_raise(_test, exc.PCOrbEmptyError,
                      r"primitive cell has no orbitals")

        # Data updating
        prim_cell = make_cell()
        super_cell = SuperCell(prim_cell, dim=(3, 3, 1))
        prim_cell.unlock()
        prim_cell.remove_orbitals([0, 1])

        def _test():
            super_cell.sync_array()
        th.test_raise(_test, exc.PCOrbEmptyError,
                      r"primitive cell has no orbitals")

        def _test():
            prim_cell.update()
        th.test_raise(_test, exc.PCOrbEmptyError,
                      r"primitive cell has no orbitals")


if __name__ == "__main__":
    unittest.main()
