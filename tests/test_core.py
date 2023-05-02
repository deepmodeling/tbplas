#! /usr/bin/env python

import unittest

import numpy as np

from tbplas import gen_lattice_vectors, PrimitiveCell, Timer, TestHelper
import tbplas.builder.core as core
from tbplas.builder.super import OrbitalSet


class TestCore(unittest.TestCase):
    def setUp(self) -> None:
        vectors = gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
        self.cell = PrimitiveCell(vectors)
        self.cell.add_orbital((0.0, 0.0), 0.0)
        self.cell.add_orbital((1./3, 1./3), 0.0)
        self.cell.add_hopping((0, 0), 0, 1, -2.7)
        self.cell.add_hopping((1, 0), 1, 0, -2.7)
        self.cell.add_hopping((0, 1), 1, 0, -2.7)
        vac = [(0, 0, 0, 1), (4, 6, 0, 0), (2, 0, 0, 0), (3, 7, 0, 0),
               (0, 6, 0, 0), (9, 0, 0, 0), (9, 9, 0, 1), (5, 7, 0, 1)]
        self.vac = vac[:8]
        self.dim_small = (17, 29, 1)
        self.dim_large = (5000, 1000, 1)
        self.timer = Timer()

    def tearDown(self) -> None:
        self.timer.report_time()

    def test00_acc_sps(self):
        """
        Test the accuracy of core functions id_sc -> id_pc -> id_sc.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_small)
        record = "acc_sps"
        self.timer.tic(record)
        result = core.test_acc_sps(orb_set._dim,
                                   orb_set.num_orb_pc,
                                   orb_set._orb_id_pc)
        self.timer.toc(record)
        self.assertEqual(result, 0)

    def test01_acc_psp(self):
        """
        Test the accuracy of core functions id_pc -> id_sc -> id_pc.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_small)
        record = "acc_psp"
        self.timer.tic(record)
        result = core.test_acc_psp(orb_set._dim,
                                   orb_set.num_orb_pc,
                                   orb_set._orb_id_pc)
        self.timer.toc(record)
        self.assertEqual(result, 0)

    def test02_acc_sps_vac(self):
        """
        Test the accuracy of core functions id_sc -> id_pc -> id_sc
        in presence of vacancies.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_small)
        orb_set.set_vacancies(self.vac)
        orb_set.sync_array()
        record = "acc_sps_vac"
        self.timer.tic(record)
        result = core.test_acc_sps_vac(orb_set._dim,
                                       orb_set.num_orb_pc,
                                       orb_set._orb_id_pc,
                                       orb_set._vac_id_sc)
        self.timer.toc(record)
        self.assertEqual(result, 0)

    def test03_acc_psp_vac(self):
        """
        Test the accuracy of core functions id_pc -> id_sc -> id_pc
        in presence of vacancies.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_small)
        orb_set.set_vacancies(self.vac)
        orb_set.sync_array()
        record = "acc_psp_vac"
        self.timer.tic(record)
        result = core.test_acc_psp_vac(orb_set._dim,
                                       orb_set.num_orb_pc,
                                       orb_set._orb_id_pc,
                                       orb_set._vac_id_sc)
        self.timer.toc(record)
        self.assertEqual(result, 0)

    def test04_speed_pc2sc(self):
        """
        Test the efficiency of core function id_pc2sc.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_large)
        record = "speed_pc2sc"
        self.timer.tic(record)
        core.test_speed_pc2sc(orb_set._dim,
                              orb_set.num_orb_pc,
                              orb_set._orb_id_pc)
        self.timer.toc(record)

    def test05_speed_pc2sc_vac(self):
        """
        Test the efficiency of core function id_pc2sc in presence of
        vacancies.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_large)
        orb_set.set_vacancies(self.vac)
        orb_set.sync_array()
        record = "speed_pc2sc_vac"
        self.timer.tic(record)
        core.test_speed_pc2sc_vac(orb_set._dim,
                                  orb_set.num_orb_pc,
                                  orb_set._orb_id_pc,
                                  orb_set._vac_id_sc)
        self.timer.toc(record)

    def test06_speed_sc2pc(self):
        """
        Test the efficiency of core function id_sc2pc.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_large)
        record = "speed_sc2pc"
        self.timer.tic(record)
        core.test_speed_sc2pc(orb_set._orb_id_pc)
        self.timer.toc(record)

    def test07_acc_py(self):
        """
        Test the accuracy of Python interfaces to core functions.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_small)

        result = 0
        record = "acc_sps_py"
        self.timer.tic(record)
        for i in range(orb_set.num_orb_pc):
            pc = orb_set.orb_id_sc2pc(i)
            sc = orb_set.orb_id_pc2sc(pc)
            result += abs(i-sc)
        self.timer.toc(record)
        self.assertEqual(result, 0)

        result = 0
        record = "acc_psp_py"
        self.timer.tic(record)
        for i in orb_set._orb_id_pc:
            sc = orb_set.orb_id_pc2sc(i)
            pc = orb_set.orb_id_sc2pc(sc)
            for j in range(4):
                result += abs(i.item(j)-pc.item(j))
        self.timer.toc(record)
        self.assertEqual(result, 0)

    def test08_acc_vac_py(self):
        """
        Test the accuracy of Python interfaces to core functions in presence
        of vacancies.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_small)
        orb_set.set_vacancies(self.vac)

        result = 0
        record = "acc_sps_vac_py"
        self.timer.tic(record)
        for i in range(orb_set.num_orb_sc):
            pc = orb_set.orb_id_sc2pc(i)
            sc = orb_set.orb_id_pc2sc(pc)
            result += abs(i-sc)
        self.timer.toc(record)
        self.assertEqual(result, 0)

        result = 0
        record = "acc_psp_vac_py"
        self.timer.tic(record)
        for i in orb_set._orb_id_pc:
            sc = orb_set.orb_id_pc2sc(i)
            pc = orb_set.orb_id_sc2pc(sc)
            for j in range(4):
                result += abs(i.item(j)-pc.item(j))
        self.timer.toc(record)
        self.assertEqual(result, 0)

    def test09_speed_pc2sc_py(self):
        """
        Test the efficiency of Python interface orb_id_pc2sc.

        :return: None.
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_large)
        record = "speed_pc2sc_py"
        self.timer.tic(record)
        for pc in orb_set._orb_id_pc:
            orb_set.orb_id_pc2sc(pc)
        self.timer.toc(record)

    def test10_speed_pc2sc_vac_py(self):
        """
        Test the efficiency of Python interface orb_id_pc2sc in presence
        of vacancies.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_large)
        orb_set.set_vacancies(self.vac)
        orb_set.sync_array()
        record = "speed_pc2sc_vac_py"
        self.timer.tic(record)
        for pc in orb_set._orb_id_pc:
            orb_set.orb_id_pc2sc(pc)
        self.timer.toc(record)

    def test11_speed_sc2pc_py(self):
        """
        Test the efficiency of Python interface orb_id_sc2pc.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_large)
        record = "speed_sc2pc_py"
        self.timer.tic(record)
        for sc in range(orb_set.num_orb_sc):
            orb_set.orb_id_sc2pc(sc)
        self.timer.toc(record)

    def test12_acc_pc2sc_array(self):
        """
        Test the accuracy of Python interface orb_id_pc2sc_array.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_small)
        th = TestHelper(self)

        record = "acc_pc2sc_array"
        self.timer.tic(record)
        sc_test = orb_set.orb_id_pc2sc_array(orb_set._orb_id_pc)
        sc_ref = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                             dtype=np.int64)
        self.timer.toc(record)
        th.test_equal_array(sc_test, sc_ref)

        record = "acc_pc2sc_array_vac"
        orb_set.set_vacancies(self.vac)
        orb_set.sync_array()
        self.timer.tic(record)
        sc_test = orb_set.orb_id_pc2sc_array(orb_set._orb_id_pc)
        sc_ref = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                             dtype=np.int64)
        self.timer.toc(record)
        th.test_equal_array(sc_test, sc_ref)

    def test13_acc_sc2pc_array(self):
        """
        Test the accuracy of Python interface orb_id_sc2pc_array.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_small)
        th = TestHelper(self)

        record = "acc_sc2pc_array"
        sc = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                         dtype=np.int64)
        self.timer.tic(record)
        pc = orb_set.orb_id_sc2pc_array(sc)
        self.timer.toc(record)
        th.test_equal_array(pc, orb_set._orb_id_pc)

        record = "acc_sc2pc_array_vac"
        orb_set.set_vacancies(self.vac)
        orb_set.sync_array()
        sc = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                         dtype=np.int64)
        self.timer.tic(record)
        pc = orb_set.orb_id_sc2pc_array(sc)
        self.timer.toc(record)
        th.test_equal_array(pc, orb_set._orb_id_pc)

    def test14_speed_pc2sc_array(self):
        """
        Test the efficiency of Python interface orb_id_pc2sc_array.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_large)

        record = "speed_pc2sc_array"
        self.timer.tic(record)
        orb_set.orb_id_pc2sc_array(orb_set._orb_id_pc)
        self.timer.toc(record)

        record = "speed_pc2sc_array_vac"
        orb_set.set_vacancies(self.vac)
        orb_set.sync_array()
        self.timer.tic(record)
        orb_set.orb_id_pc2sc_array(orb_set._orb_id_pc)
        self.timer.toc(record)

    def test15_speed_sc2pc_array(self):
        """
        Test the efficiency of Python interface orb_id_sc2pc_array.

        :return: None
        """
        orb_set = OrbitalSet(self.cell, dim=self.dim_large)

        record = "speed_sc2pc_array"
        sc = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                         dtype=np.int64)
        self.timer.tic(record)
        orb_set.orb_id_sc2pc_array(sc)
        self.timer.toc(record)

        record = "speed_sc2pc_array_vac"
        orb_set.set_vacancies(self.vac)
        orb_set.sync_array()
        sc = np.linspace(0, orb_set.num_orb_sc-1, orb_set.num_orb_sc,
                         dtype=np.int64)
        self.timer.tic(record)
        orb_set.orb_id_sc2pc_array(sc)
        self.timer.toc(record)


if __name__ == "__main__":
    unittest.main()
