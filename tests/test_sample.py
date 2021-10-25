#! /usr/bin/env python

import unittest

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

import tipsi.builder.lattice as lat
import tipsi.builder.kpoints as kpt
import tipsi.builder.exceptions as exc
import tipsi.builder.core as core
from tipsi.builder import PrimitiveCell, SuperCell, InterHopping, Sample
from tipsi.utils import Timer
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


def make_test_set():
    # Make 3 super cells
    sc1 = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))
    sc2 = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))
    sc3 = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))

    # Make 2 inter-hopping instances
    inter_hop1 = InterHopping(sc_bra=sc1, sc_ket=sc2)
    inter_hop1.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(1, 2), orb_j=1,
                           energy=-1.2)
    inter_hop1.add_hopping(rn_i=(2, 1), orb_i=1, rn_j=(1, 2), orb_j=0,
                           energy=1.5)
    inter_hop1.add_hopping(rn_i=(2, 0), orb_i=0, rn_j=(0, 2), orb_j=1,
                           energy=-0.7)

    inter_hop2 = InterHopping(sc_bra=sc2, sc_ket=sc3)
    inter_hop2.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(1, 2), orb_j=1,
                           energy=-1.2)
    inter_hop2.add_hopping(rn_i=(2, 1), orb_i=1, rn_j=(1, 2), orb_j=0,
                           energy=1.5)
    inter_hop2.add_hopping(rn_i=(2, 0), orb_i=0, rn_j=(0, 2), orb_j=1,
                           energy=-0.7)
    return sc1, sc2, sc3, inter_hop1, inter_hop2


class TestSample(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test00_add_hop(self):
        """
        Test 'add_hop' method of 'InterHopping' class.

        :return: None
        """
        sc1 = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))
        sc2 = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))
        th = TestHelper(self)

        # Exception handling
        def _test():
            inter_hop2 = InterHopping(sc_bra=sc1, sc_ket=sc2)
            inter_hop2.lock()
            inter_hop2.add_hopping(rn_i=(2, 1, 3), orb_i=1, rn_j=(2, 0, 3),
                                   orb_j=1, energy=-1.1)
        th.test_raise(_test, exc.InterHopLockError, r"trying to modify a locked"
                                                    r" inter-hopping object")

        def _test():
            inter_hop2 = InterHopping(sc_bra=sc1, sc_ket=sc2)
            inter_hop2.add_hopping(rn_i=(2,), orb_i=1, rn_j=(2, 1), orb_j=1,
                                   energy=-1.1)
        th.test_raise(_test, exc.IDPCLenError, r"length of id_pc .+ is not 4")

        def _test():
            inter_hop2 = InterHopping(sc_bra=sc1, sc_ket=sc2)
            inter_hop2.add_hopping(rn_i=(2, 1), orb_i=1, rn_j=(2, 1, 3, 3),
                                   orb_j=1, energy=-1.1)
        th.test_raise(_test, exc.IDPCLenError, r"length of id_pc .+ is not 4")

        # Normal case
        inter_hop = InterHopping(sc_bra=sc1, sc_ket=sc2)

        # Test default attributes
        self.assertListEqual(inter_hop.indices, [])
        self.assertListEqual(inter_hop.energies, [])

        # Add one hopping term and check the attributes
        inter_hop.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(2, 1), orb_j=1,
                              energy=-1.2)
        self.assertTupleEqual(inter_hop.indices[0][0], (0, 0, 0, 0))
        self.assertTupleEqual(inter_hop.indices[0][1], (2, 1, 0, 1))
        self.assertListEqual(inter_hop.energies, [-1.2])

        # Overwrite the same item
        inter_hop.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(2, 1), orb_j=1,
                              energy=-1.5)
        self.assertTupleEqual(inter_hop.indices[0][0], (0, 0, 0, 0))
        self.assertTupleEqual(inter_hop.indices[0][1], (2, 1, 0, 1))
        self.assertListEqual(inter_hop.energies, [-1.5])

    def test01_get_hop(self):
        """
        Test 'get_hop' method method of 'InterHopping' class.

        :return: None
        """
        pc1 = make_cell()
        pc2 = make_cell()
        pc2.set_orbital(orb_i=0, position=[1./3, 1./3, 0.0])
        pc2.set_orbital(orb_i=1, position=[2./3, 2./3, 0.0])
        sc1 = SuperCell(pc1, dim=(3, 3, 1), pbc=(True, True, False))
        sc2 = SuperCell(pc2, dim=(3, 3, 1), pbc=(True, True, False))
        th = TestHelper(self)

        # Exception handling
        def _test():
            inter_hop2 = InterHopping(sc_bra=sc1, sc_ket=sc2)
            inter_hop2.get_hop()
        th.test_raise(_test, exc.InterHopVoidError, r"no hopping terms added to"
                                                    r" InterHopping instance")

        def _test():
            inter_hop2 = InterHopping(sc_bra=sc1, sc_ket=sc2)
            inter_hop2.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(3, 3), orb_j=1)
            inter_hop2.get_hop()
        th.test_raise(_test, exc.IDPCIndexError, r"cell index .+ of .+ out"
                                                 r" of range")

        def _test():
            inter_hop2 = InterHopping(sc_bra=sc1, sc_ket=sc2)
            inter_hop2.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(0, 0), orb_j=3)
            inter_hop2.get_hop()
        th.test_raise(_test, exc.IDPCIndexError, r"orbital index .+ of .+ out"
                                                 r" of range")

        # Normal case
        inter_hop = InterHopping(sc_bra=sc1, sc_ket=sc2)
        inter_hop.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(1, 2), orb_j=1,
                              energy=-1.2)
        inter_hop.add_hopping(rn_i=(2, 1), orb_i=1, rn_j=(1, 2), orb_j=0,
                              energy=1.5)
        inter_hop.add_hopping(rn_i=(2, 0), orb_i=0, rn_j=(0, 2), orb_j=1,
                              energy=-0.7)
        hop_i_ref = [0, 15, 12]
        hop_j_ref = [11, 10, 5]
        hop_v_ref = [-1.2, 1.5, -0.7]
        hop_i, hop_j, hop_v = inter_hop.get_hop()
        th.test_equal_array(hop_i, hop_i_ref)
        th.test_equal_array(hop_j, hop_j_ref)
        th.test_equal_array(hop_v, hop_v_ref)

    def test02_get_dr(self):
        """
        Test 'get_dr' method method of 'InterHopping' class.

        :return: None
        """
        pc1 = make_cell()
        pc2 = make_cell()
        pc2.set_orbital(orb_i=0, position=[1./3, 1./3, 0.0])
        pc2.set_orbital(orb_i=1, position=[2./3, 2./3, 0.0])
        sc1 = SuperCell(pc1, dim=(3, 3, 1), pbc=(True, True, False))
        sc2 = SuperCell(pc2, dim=(3, 3, 1), pbc=(True, True, False))
        th = TestHelper(self)

        # Exception handling
        def _test():
            inter_hop2 = InterHopping(sc_bra=sc1, sc_ket=sc2)
            inter_hop2.get_hop()
        th.test_raise(_test, exc.InterHopVoidError, r"no hopping terms added to"
                                                    r" InterHopping instance")

        # Normal case
        inter_hop = InterHopping(sc_bra=sc1, sc_ket=sc2)
        inter_hop.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(1, 2), orb_j=1,
                              energy=-1.2)
        inter_hop.add_hopping(rn_i=(2, 1), orb_i=1, rn_j=(1, 2), orb_j=0,
                              energy=1.5)
        inter_hop.add_hopping(rn_i=(2, 0), orb_i=0, rn_j=(0, 2), orb_j=1,
                              energy=-0.7)

        hop_i, hop_j, hop_v = inter_hop.get_hop()
        orb_pos1 = sc1.get_orb_pos()
        orb_pos2 = sc2.get_orb_pos()
        dr_ref = []
        for i in range(hop_i.shape[0]):
            dr_ref.append(orb_pos2[hop_j.item(i)] - orb_pos1[hop_i.item(i)])
        dr = inter_hop.get_dr()
        th.test_equal_array(dr, dr_ref)

    def test03_init(self):
        """
        Test initialization of 'Sample' class.

        :return: None
        """
        sc1 = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))
        sc2 = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))
        th = TestHelper(self)

        # Exception handling
        def _test():
            Sample()
        th.test_raise(_test, exc.SampleVoidError, r"no components assigned to"
                                                  r" Sample instance")

        def _test():
            Sample(sc1, sc2, make_cell())
        th.test_raise(_test, exc.SampleCompError,
                      r"component .+ should be instance of SuperCell or"
                      r" InterHopping")

        def _test():
            sc3 = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))
            inter_hop2 = InterHopping(sc1, sc3)
            Sample(sc1, sc2, inter_hop2)
        th.test_raise(_test, exc.SampleClosureError,
                      r".+ of inter_hop .+ not included in sample")

        # Normal case
        inter_hop = InterHopping(sc_bra=sc1, sc_ket=sc2)
        inter_hop.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(1, 2), orb_j=1,
                              energy=-1.2)
        inter_hop.add_hopping(rn_i=(2, 1), orb_i=1, rn_j=(1, 2), orb_j=0,
                              energy=1.5)
        inter_hop.add_hopping(rn_i=(2, 0), orb_i=0, rn_j=(0, 2), orb_j=1,
                              energy=-0.7)
        Sample(sc1)
        Sample(sc1, sc2)
        Sample(sc1, sc2, inter_hop)

    def test04_private(self):
        """
        Test '__get_num_orb' and '__get_ind_start' methods of 'Sample' class.

        :return: None
        """
        sc1, sc2, sc3, inter_hop1, inter_hop2 = make_test_set()

        # Normal case with 1 super cell
        sample = Sample(sc1)
        num_orb = sample._Sample__get_num_orb()
        ind_start = sample._Sample__get_ind_start()
        self.assertListEqual(num_orb, [18])
        self.assertListEqual(ind_start, [0])

        # Normal case with 2 super cells and 1 inter-hopping
        sample = Sample(sc1, sc2, inter_hop1)
        num_orb = sample._Sample__get_num_orb()
        ind_start = sample._Sample__get_ind_start()
        self.assertListEqual(num_orb, [18, 18])
        self.assertListEqual(ind_start, [0, 18])

        # Normal case with 3 super cells and 2 inter-hopping
        sample = Sample(sc1, sc2, sc3, inter_hop1, inter_hop2)
        num_orb = sample._Sample__get_num_orb()
        ind_start = sample._Sample__get_ind_start()
        self.assertListEqual(num_orb, [18, 18, 18])
        self.assertListEqual(ind_start, [0, 18, 36])

        # Abnormal case with 2 super cells and no inter-hopping
        sample = Sample(sc1, sc2)
        num_orb = sample._Sample__get_num_orb()
        ind_start = sample._Sample__get_ind_start()
        self.assertListEqual(num_orb, [18, 18])
        self.assertListEqual(ind_start, [0, 18])

        # Abnormal case with 3 super cells and no inter-hopping
        sample = Sample(sc1, sc2, sc3)
        num_orb = sample._Sample__get_num_orb()
        ind_start = sample._Sample__get_ind_start()
        self.assertListEqual(num_orb, [18, 18, 18])
        self.assertListEqual(ind_start, [0, 18, 36])

    def test05_init_orb(self):
        """
        Test 'init_orb_eng' and 'init_orb_pos' methods of 'Sample' class.

        :return: None
        """
        sc1, sc2, sc3, inter_hop1, inter_hop2 = make_test_set()

        # Normal case with 1 super cell
        sample = Sample(sc1)
        sample.init_orb_eng()
        sample.init_orb_pos()
        self.assertTupleEqual(sample.orb_eng.shape, (18,))
        self.assertTupleEqual(sample.orb_pos.shape, (18, 3))

        # Normal case with 2 super cells and 1 inter-hopping
        sample = Sample(sc1, sc2, inter_hop1)
        sample.init_orb_eng()
        sample.init_orb_pos()
        self.assertTupleEqual(sample.orb_eng.shape, (36,))
        self.assertTupleEqual(sample.orb_pos.shape, (36, 3))

        # Normal case with 3 super cells and 2 inter-hopping
        sample = Sample(sc1, sc2, sc3, inter_hop1, inter_hop2)
        sample.init_orb_eng()
        sample.init_orb_pos()
        self.assertTupleEqual(sample.orb_eng.shape, (54,))
        self.assertTupleEqual(sample.orb_pos.shape, (54, 3))

        # Abnormal case with 2 super cells and no inter-hopping
        sample = Sample(sc1, sc2)
        sample.init_orb_eng()
        sample.init_orb_pos()
        self.assertTupleEqual(sample.orb_eng.shape, (36,))
        self.assertTupleEqual(sample.orb_pos.shape, (36, 3))

        # Abnormal case with 3 super cells and no inter-hopping
        sample = Sample(sc1, sc2, sc3)
        sample.init_orb_eng()
        sample.init_orb_pos()
        self.assertTupleEqual(sample.orb_eng.shape, (54,))
        self.assertTupleEqual(sample.orb_pos.shape, (54, 3))

    def test06_init_hop(self):
        """
        Test 'init_hop' and 'init_dr' methods of 'Sample' class.

        :return: None
        """
        sc1, sc2, sc3, inter_hop1, inter_hop2 = make_test_set()

        # Normal case with 1 super cell
        sample = Sample(sc1)
        sample.init_hop()
        sample.init_dr()
        self.assertTupleEqual(sample.hop_i.shape, (27,))
        self.assertTupleEqual(sample.dr.shape, (27, 3))

        # Normal case with 2 super cells and 1 inter-hopping
        sample = Sample(sc1, sc2, inter_hop1)
        sample.init_hop()
        sample.init_dr()
        self.assertTupleEqual(sample.hop_i.shape, (57,))
        self.assertTupleEqual(sample.dr.shape, (57, 3))

        # Normal case with 3 super cells and 2 inter-hopping
        sample = Sample(sc1, sc2, sc3, inter_hop1, inter_hop2)
        sample.init_hop()
        sample.init_dr()
        self.assertTupleEqual(sample.hop_i.shape, (87,))
        self.assertTupleEqual(sample.dr.shape, (87, 3))

        # Abnormal case with 2 super cells and no inter-hopping
        sample = Sample(sc1, sc2)
        sample.init_hop()
        sample.init_dr()
        self.assertTupleEqual(sample.hop_i.shape, (54,))
        self.assertTupleEqual(sample.dr.shape, (54, 3))

        # Abnormal case with 3 super cells and no inter-hopping
        sample = Sample(sc1, sc2, sc3)
        sample.init_hop()
        sample.init_dr()
        self.assertTupleEqual(sample.hop_i.shape, (81,))
        self.assertTupleEqual(sample.dr.shape, (81, 3))

    def test07_rescale(self):
        """
        Test if the the new code to determine rescale factor yields the same
        result as the old version and compare their efficiency.

        :return: None
        """
        def _rescale_old():
            value = 0.0
            for i in range(len(ham_csr.indptr) - 1):
                max_val = np.sum(
                    [np.absolute(ham_csr.data[ham_csr.indptr[i]:ham_csr.indptr[i + 1]])])
                value = np.amax((max_val, value))
            return value

        def _rescale_new():
            return core.get_rescale(sample.orb_eng, sample.hop_i, sample.hop_j,
                                    sample.hop_v)

        # Build sample
        sc = SuperCell(make_cell(), dim=(500, 1000, 1),
                       pbc=(False, False, False))
        sample = Sample(sc)
        ham_csr = sample.build_ham_csr()

        # Test performance
        timer = Timer()
        timer.tic("rescale_old")
        value_old = _rescale_old()
        timer.toc("rescale_old")
        timer.tic("rescale_new")
        value_new = _rescale_new()
        timer.toc("rescale_new")
        print()
        timer.report_total_time()

        # Test accuracy
        self.assertAlmostEqual(value_old, value_new)

    def test08_set_magnetic_field(self):
        """
        Test if the the new code of adding magnetic field yields the same result
        as the old version and compare their efficiency.

        :return: None.
        """
        def _set_mag_old(b_field):
            for i0 in range(sc.num_orb_sc):
                for i in range(indptr[i0], indptr[i0+1]):
                    i1 = indices[i]
                    y_tot = sample.orb_pos[i0, 1] + sample.orb_pos[i1, 1]
                    phase = 1j * np.pi * b_field * dx[i] * y_tot / 4135.666734
                    hop[i] = hop[i] * np.exp(phase)

        def _set_mag_new(b_field):
            sample.set_magnetic_field(b_field)

        # Build the sample
        sc = SuperCell(make_cell(), dim=(500, 1000, 1),
                       pbc=(False, False, False))
        sample = Sample(sc)
        indptr, indices, hop, dx, dy = sample.build_ham_dxy()

        # Test performance
        timer = Timer()
        timer.tic("set_mag_old")
        _set_mag_old(10.0)
        timer.toc("set_mag_old")
        timer.tic("set_mag_new")
        _set_mag_new(10.0)
        timer.toc("set_mag_new")
        print()
        timer.report_total_time()

        # Test accuracy
        shape = (sc.num_orb_sc, sc.num_orb_sc)
        ham_csr_old = csr_matrix((hop, indices, indptr), shape)
        ham_csr_new = sample.build_ham_csr()
        self.assertAlmostEqual((ham_csr_old - ham_csr_new).sum(), 0.0)

    def test09_build_ham_dxy(self):
        """
        Test the algorithms under 'build_ham_dxy' method of 'Sample' class.

        :return: None.
        """
        sc = SuperCell(make_cell(), dim=(500, 500, 1), pbc=(True, True))
        sample = Sample(sc)
        indptr1, indices1, hop1, dx1, dy1 = sample.build_ham_dxy(algo="v1")
        indptr2, indices2, hop2, dx2, dy2 = sample.build_ham_dxy(algo="v2")

        # The two algorithms should agree.
        shape = (sc.num_orb_sc, sc.num_orb_sc)
        ham_csr1 = csr_matrix((hop1, indices1, indptr1), shape)
        ham_csr2 = csr_matrix((hop2, indices2, indptr2), shape)
        dx_csr1 = csr_matrix((dx1, indices1, indptr1), shape)
        dx_csr2 = csr_matrix((dx2, indices2, indptr2), shape)
        dy_csr1 = csr_matrix((dy1, indices1, indptr1), shape)
        dy_csr2 = csr_matrix((dy2, indices2, indptr2), shape)
        self.assertAlmostEqual((ham_csr1 - ham_csr2).sum(), 0.0)
        self.assertAlmostEqual((dx_csr1 - dx_csr2).sum(), 0.0)
        self.assertAlmostEqual((dy_csr1 - dy_csr2).sum(), 0.0)

        # Of course, they have to agree with build_*_csr.
        ham_csr3 = sample.build_ham_csr()
        dx_csr3, dy_csr3 = sample.build_dxy_csr()
        self.assertAlmostEqual((ham_csr1 - ham_csr3).sum(), 0.0)
        self.assertAlmostEqual((dx_csr1 - dx_csr3).sum(), 0.0)
        self.assertAlmostEqual((dy_csr1 - dy_csr3).sum(), 0.0)
        self.assertEqual(sample.extended, 1)

    def test10_speed(self):
        """
        Test the efficiency of time-consuming methods of 'Sample' class.

        :return: None.
        """
        timer = Timer()

        # Sample initialization
        timer.tic("sc_init")
        sc = SuperCell(make_cell(), dim=(5000, 1000), pbc=(True, True))
        sample = Sample(sc)
        timer.toc("sc_init")

        # init_*
        timer.tic("init_hop")
        sample.init_hop()
        timer.toc("init_hop")
        timer.tic("init_orb_pos")
        sample.init_orb_pos()
        timer.toc("init_orb_pos")
        timer.tic("init_orb_eng")
        sample.init_orb_eng()
        timer.toc("init_orb_eng")
        timer.tic("init_dr")
        sample.init_dr()
        timer.toc("init_dr")

        # build_*
        timer.tic("build_ham_csr")
        sample.build_ham_csr()
        timer.toc("build_ham_csr")
        timer.tic("build_dxy_csr")
        sample.build_dxy_csr()
        timer.toc("build_dxy_csr")
        timer.tic("build_ham_dxy_v1")
        sample.build_ham_dxy(algo="v1")
        timer.toc("build_ham_dxy_v1")
        timer.tic("build_ham_dxy_v2")
        sample.build_ham_dxy(algo="v2")
        timer.toc("build_ham_dxy_v2")
        print()
        timer.report_total_time()

    def test11_plot(self):
        """
        Test the 'plot' method of 'Sample' class.

        :return: None.
        """
        print("\n3x3 Graphene super cell with pbc")
        sc = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False))
        sample = Sample(sc)
        sample.plot()

        print("3x3 Graphene super cell with open bc")
        sc = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(False, False, False))
        sample = Sample(sc)
        sample.plot()

        print("3x3 Graphene super cell with pbc and vacancies")
        vacancies = [(1, 1, 0, 0), (1, 1, 0, 1)]
        sc = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(True, True, False),
                       vacancies=vacancies)
        sample = Sample(sc)
        sample.plot()

        print("3x3 Graphene super cell with open bc and vacancies")
        vacancies = [(1, 1, 0, 0), (1, 1, 0, 1)]
        sc = SuperCell(make_cell(), dim=(3, 3, 1), pbc=(False, False, False),
                       vacancies=vacancies)
        sample = Sample(sc)
        sample.plot()

        print("3x3 Graphene super cell with 2 layers")
        pc1 = make_cell()
        pc2 = make_cell()
        pc2.set_orbital(orb_i=0, position=[1./3, 1./3, 0.0])
        pc2.set_orbital(orb_i=1, position=[2./3, 2./3, 0.0])
        sc1 = SuperCell(pc1, dim=(3, 3, 1), pbc=(True, True, False))
        sc2 = SuperCell(pc2, dim=(3, 3, 1), pbc=(True, True, False))
        inter_hop = InterHopping(sc_bra=sc1, sc_ket=sc2)
        inter_hop.add_hopping(rn_i=(0, 0), orb_i=0, rn_j=(1, 2), orb_j=1,
                              energy=-1.2)
        inter_hop.add_hopping(rn_i=(2, 1), orb_i=1, rn_j=(1, 2), orb_j=0,
                              energy=1.5)
        inter_hop.add_hopping(rn_i=(2, 0), orb_i=0, rn_j=(0, 2), orb_j=1,
                              energy=-0.7)
        sample = Sample(sc1, sc2, inter_hop)
        sample.plot()

    def test12_plot_advanced(self):
        """
        Test the 'plot' method of 'Sample' class with a complicated structure.

        :return: None.
        """
        sc = SuperCell(make_cell(), dim=(24, 24, 1))
        sample = Sample(sc)
        sample.init_orb_pos()
        positions = sample.orb_pos

        holes = []
        center = np.array([2.101, 1.361, 0.0])
        for i, id_pc in enumerate(sc.orb_id_pc):
            if np.linalg.norm(positions[i] - center) <= 0.5:
                holes.append(tuple(id_pc))
        center = np.array([3.101, 3.361, 0.0])
        for i, id_pc in enumerate(sc.orb_id_pc):
            if np.linalg.norm(positions[i] - center) <= 0.5:
                holes.append(tuple(id_pc))
        center = np.array([5.84, 3.51, 0.0])
        for i, id_pc in enumerate(sc.orb_id_pc):
            if np.linalg.norm(positions[i] - center) <= 0.5:
                holes.append(tuple(id_pc))
        center = np.array([4.82, 1.11, 0.0])
        for i, id_pc in enumerate(sc.orb_id_pc):
            if np.linalg.norm(positions[i] - center) <= 0.5:
                holes.append(tuple(id_pc))

        print("\n24*24 Graphene super cell with 4 holes")
        sc = SuperCell(make_cell(), dim=(24, 24, 1), vacancies=holes)
        sample = Sample(sc)
        sample.plot(with_cells=False)

    def test13_calc_bands(self):
        """
        Test the 'calc_bands' method of 'Sample' class.

        :return: None
        """
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_path = kpt.gen_kpath(k_points, [40, 40, 40])
        sc = SuperCell(make_cell(), dim=(5, 5, 1), pbc=(True, True, False))
        sample = Sample(sc)

        with self.assertRaises(exc.SolverError) as cm:
            sample.calc_bands(k_path, solver="abc")
        self.assertRegex(str(cm.exception), r"illegal solver abc")

        k_len, bands = sample.calc_bands(k_path)
        num_bands = bands.shape[1]
        for i in range(num_bands):
            plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
        plt.show()

    def test14_calc_dos(self):
        """
        Test the 'calc_dos' method of 'Sample' class.

        :return: None
        """
        k_points = kpt.gen_kmesh((24, 24, 1))
        sc = SuperCell(make_cell(), dim=(5, 5, 1), pbc=(True, True, False))
        sample = Sample(sc)

        with self.assertRaises(exc.BasisError) as cm:
            sample.calc_dos(k_points, basis="abc")
        self.assertRegex(str(cm.exception), r"illegal basis function abc")

        energies, dos = sample.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()


if __name__ == "__main__":
    unittest.main()