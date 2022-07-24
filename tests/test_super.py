#! /usr/bin/env python

import unittest

import numpy as np

from tbplas import gen_lattice_vectors, PrimitiveCell, SuperCell
import tbplas.builder.exceptions as exc
import tbplas.builder.core as core
from tbplas.builder.super import OrbitalSet
from tbplas.utils import TestHelper


class TestSuper(unittest.TestCase):
    def setUp(self) -> None:
        vectors = gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
        self.cell = PrimitiveCell(vectors)
        self.cell.add_orbital((0.0, 0.0), 0.0)
        self.cell.add_orbital((1./3, 1./3), 0.0)
        self.cell.add_hopping((0, 0), 0, 1, -2.7)
        self.cell.add_hopping((1, 0), 1, 0, -2.7)
        self.cell.add_hopping((0, 1), 1, 0, -2.7)

    def tearDown(self) -> None:
        pass

    def test00_init_illegal(self):
        """
        Test if exception handling of OrbitalSet.__init__ works as expected.

        NOTE: exceptions related to set_vacancy are tested with that function,
        not here.

        :return: None.
        """
        # PCLockError
        with self.assertRaises(exc.PCLockError) as cm:
            orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
            orb_set.prim_cell.add_orbital((0.0, 0.0), 1.5)
        self.assertRegex(str(cm.exception),
                         r"trying to modify a locked primitive cell")

        # DimLenError
        with self.assertRaises(exc.SCDimLenError) as cm:
            OrbitalSet(self.cell, dim=(1, 1, 1, 2))
        self.assertRegex(str(cm.exception),
                         r"length of supercell dimension .+ not in .+")

        # DimSizeError
        msg_dim = r"^dimension on direction [0-2] should" \
                  r" be no less than [0-9]+"
        with self.assertRaises(exc.SCDimSizeError) as cm:
            OrbitalSet(self.cell, dim=(-1, 1, 1))
        self.assertRegex(str(cm.exception), msg_dim)
        with self.assertRaises(exc.SCDimSizeError) as cm:
            OrbitalSet(self.cell, dim=(2, -1, 1))
        self.assertRegex(str(cm.exception), msg_dim)
        with self.assertRaises(exc.SCDimSizeError) as cm:
            OrbitalSet(self.cell, dim=(2, 1, -1))
        self.assertRegex(str(cm.exception), msg_dim)

        # PBCLenError
        with self.assertRaises(exc.PBCLenError) as cm:
            OrbitalSet(self.cell, dim=(3, 3, 1), pbc=(True, False, True, False))
        self.assertRegex(str(cm.exception), r"length of pbc .+ not in \(2, 3\)")

    def test01_init(self):
        """
        Test if OrbitalSet.__init__ works as expected for legal input.

        :return: None.
        """
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
        self.assertEqual(orb_set.vacancy_list, [])
        self.assertIsNone(orb_set.vac_id_pc)
        self.assertIsNone(orb_set.vac_id_sc)
        self.assertEqual(orb_set.orb_id_pc.shape, (18, 4))

    def test02_check_id_pc_illegal(self):
        """
        Check if error handling of OrbitalSet.check_id_pc works as expected.

        :return: None.
        """
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))

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

    def test03_check_id_pc(self):
        """
        Tests if OrbitalSet.check_id_pc works as expected for legal input.
        :return:
        """
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
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

    def test04_set_vacancies_illegal(self):
        """
        Test if error handling of OrbitalSet.set_vacancies works as expected.

        :return: None.
        """
        # Feed a list of vacancies with a illegal one.
        # Errors should be raised
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
        vacancies = [(0, 0, 0, 1), (3, 0, 0, 1)]

        # VacIDPCLenError
        with self.assertRaises(exc.VacIDPCLenError) as cm:
            orb_set.set_vacancies([(0, 1, 0), (0, 0, 1, 0, 0)])
        self.assertRegex(str(cm.exception),
                         r"length of id_pc .+ is not 4")

        # VacIDPCIndexError
        with self.assertRaises(exc.VacIDPCIndexError) as cm:
            orb_set.set_vacancies(vacancies)
        self.assertRegex(str(cm.exception),
                         r"cell index .+ of id_pc .+ out of range")

    def test05_set_vacancies(self):
        """
        Test if OrbitalSet.set_vacancies works as expected for legal input.

        :return: None.
        """
        th = TestHelper(self)
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        vac_id_pc = np.array(vacancies)
        vac_id_sc = np.array([1, 2, 9])
        orb_id_pc = [(i_a, i_b, 0, i_o)
                     for i_a in range(3)
                     for i_b in range(3)
                     for i_o in range(2)]
        for vac in vacancies:
            orb_id_pc.remove(vac)
        orb_id_pc = np.array(orb_id_pc)

        # First we feed some effective input to set_vacancies.
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
        orb_set.set_vacancies(vacancies, force_sync=True)
        self.assertListEqual(orb_set.vacancy_list, vacancies)
        th.test_equal_array(orb_set.vac_id_pc, vac_id_pc)
        th.test_equal_array(orb_set.vac_id_pc, vac_id_pc)
        th.test_equal_array(orb_set.vac_id_sc, vac_id_sc)
        th.test_equal_array(orb_set.orb_id_pc, orb_id_pc)

        # Then we feed a blank to set_vacancies.
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
        orb_set.set_vacancies([])
        self.assertEqual(orb_set.vacancy_list, [])
        self.assertIsNone(orb_set.vac_id_pc)
        self.assertIsNone(orb_set.vac_id_sc)
        self.assertEqual(orb_set.orb_id_pc.shape, (18, 4))

        # Finally, we feed a list of vacancies with redundant items.
        # These redundant items should be detected and removed.
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
        vacancies = [(0, 0, 0, 1), (1, 0, 0, 0), (0, 0, 0, 1)]
        orb_set.set_vacancies(vacancies)
        reduced_vacancies = [(0, 0, 0, 1), (1, 0, 0, 0)]
        self.assertListEqual(orb_set.vacancy_list, reduced_vacancies)

    def test06_sync_array(self):
        """
        Test if OrbitalSet.sync_array works as expected.

        :return: None.
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
        th = TestHelper(self)
        update = [r"INFO: updating sc vacancy and orbital arrays"]
        no_update = [r"INFO: no need to update sc vacancy and orbital arrays"]

        def _test():
            orb_set.sync_array(verbose=True)

        # 1st call
        # Nothing will be updated as orb_set.vacancy_list does not change.
        th.test_stdout(_test, no_update)

        # 2nd call
        # As we have updated orb_set.vacancy_list, arrays will be updated.
        orb_set.vacancy_list = vacancies
        th.test_stdout(_test, update)

        # 3rd call
        # Nothing will be updated.
        th.test_stdout(_test, no_update)

        # 4th call with vacancy_list set to []
        orb_set.vacancy_list = []
        th.test_stdout(_test, update)

        # 5th call
        # Nothing will be updated.
        th.test_stdout(_test, no_update)

    def test07_properties(self):
        """
        Test methods of SuperCell decorated with '@property'.

        :return: None
        """
        sc = SuperCell(self.cell, dim=(3, 3, 1))
        th = TestHelper(self)
        self.assertEqual(sc.num_orb_pc,
                         sc.prim_cell.num_orb)
        self.assertEqual(sc.num_orb_sc, sc.orb_id_pc.shape[0])
        th.test_equal_array(sc.pc_lat_vec, sc.prim_cell.lat_vec, almost=True)
        th.test_equal_array(sc.pc_orb_pos, sc.prim_cell.orb_pos, almost=True)
        th.test_equal_array(sc.pc_orb_eng, sc.prim_cell.orb_eng, almost=True)
        th.test_equal_array(sc.pc_hop_ind, sc.prim_cell.hop_ind)
        th.test_equal_array(sc.pc_hop_eng, sc.prim_cell.hop_eng, almost=True)

    def test08_orb_id_sc2pc(self):
        """
        Test if OrbitalSet.orb_id_sc2pc works as expected.

        :return: None.
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
        orb_set.set_vacancies(vacancies)
        th = TestHelper(self)

        # Feed a wrong input and check if errors are raised.
        with self.assertRaises(exc.IDSCIndexError) as cm:
            orb_set.orb_id_sc2pc(21)
        self.assertRegex(str(cm.exception), r"id_sc .+ out of range")

        # Then test the normal case
        for i in range(orb_set.num_orb_sc):
            th.test_equal_array(orb_set.orb_id_sc2pc(i), orb_set.orb_id_pc[i])

    def test09_orb_id_pc2sc(self):
        """
        Test if OrbitalSet.orb_id_pc2sc works as expected.

        NOTE: errors related to check_id_pc are tested with that function,
        not here.

        :return: None.
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
        orb_set.set_vacancies(vacancies)

        # Test if the function can detect errors
        with self.assertRaises(exc.IDPCVacError) as cm:
            orb_set.orb_id_pc2sc((1, 1, 0, 1))
        self.assertRegex(str(cm.exception),
                         r"orbital id_pc .+ seems to be a vacancy")

        # Then test the normal case
        for i in range(orb_set.num_orb_sc):
            self.assertEqual(orb_set.orb_id_pc2sc(orb_set.orb_id_pc[i]), i)

    def test10_orb_id_sc2pc_array(self):
        """
        Test if OrbitalSet.orb_od_sc2pc_array works as expected.

        :return: None
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
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
        th.test_equal_array(id_pc, orb_set.orb_id_pc)

    def test11_orb_id_pc2sc_array(self):
        """
        Test if OrbitalSet.orb_od_pc2sc_array works as expected.

        :return: None
        """
        vacancies = [(0, 0, 0, 1), (0, 1, 0, 0), (1, 1, 0, 1)]
        orb_set = OrbitalSet(self.cell, dim=(3, 3, 1))
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
        id_sc_test = orb_set.orb_id_pc2sc_array(orb_set.orb_id_pc)
        id_sc_ref = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                                dtype=np.int)
        th.test_equal_array(id_sc_test, id_sc_ref)

    def test13_get_orb_eng(self):
        """
        Test if SuperCell.get_orb_eng works as expected.

        :return: None
        """
        sc = SuperCell(self.cell, dim=(3, 3, 1))
        orb_eng = sc.get_orb_eng()
        self.assertTupleEqual(orb_eng.shape, (18,))

    def test14_get_orb_pos(self):
        """
        Test if SuperCell.get_orb_pos works as expected.

        :return: None
        """
        sc = SuperCell(self.cell, dim=(3, 3, 1))
        orb_pos = sc.get_orb_pos()
        self.assertTupleEqual(orb_pos.shape, (18, 3))

    def test15_get_hop_dr(self):
        """
        Test if SuperCell.get_hop and SuperCell.get_dr work as expected.

        :return: None
        """
        # Test hop_modifier
        sc = SuperCell(self.cell, dim=(3, 3, 1))

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
        id_sc_bra = sc.orb_id_pc2sc_array(id_pc_bra)
        id_sc_ket = sc.orb_id_pc2sc_array(id_pc_ket)
        for i in range(id_sc_bra.shape[0]):
            sc.add_hopping((0, 0, 0), id_sc_bra[i], id_sc_ket[i], energy=0.0)

        # Inspect the arrays
        hop_i = sc.get_hop()[0]
        dr = sc.get_dr()
        self.assertEqual(hop_i.shape[0], dr.shape[0])


if __name__ == "__main__":
    unittest.main()
