#! /usr/bin/env python

import unittest
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix

# Old version of Tipsi builder
import tipsi

# New version of Tipsi builder
# import tipsi.builder.lattice as lat
from tipsi.builder import PrimitiveCell, SuperCell, Sample
# from test_utils import TestHelper


# def make_sample_old(shape, enable_pbc=True):
#     """Make sample for given shape and pbc with old version of builder."""
#     # Lattice
#     a = 0.246
#     vectors = [[0.5*a, 0.5*sqrt(3)*a, 0.], [-0.5*a, 0.5*sqrt(3)*a, 0.]]
#     orbital_coords = [[0., 0., 0.], [0, 1./3*sqrt(3)*a, 0.]]
#     lattice = tipsi.Lattice(vectors, orbital_coords)
#
#     # SiteSet
#     W, H = shape[0], shape[1]
#     site_set = tipsi.SiteSet()
#     for i in range(W):
#         for j in range(H):
#             unit_cell_coords = (i, j, 0)
#             site_set.add_site(unit_cell_coords, 0)
#             site_set.add_site(unit_cell_coords, 1)
#
#     # HopDict
#     t = 2.7
#     e = 0.0
#     hop_dict = tipsi.HopDict()
#     rn = (0, 0, 0)
#     hop_dict.empty(rn, (2, 2))
#     hop_dict.set_element(rn, (0, 0), e)
#     hop_dict.set_element(rn, (1, 0), t)
#     hop_dict.set_element(rn, (1, 1), e)
#     hop_dict.set_element(rn, (0, 1), t)
#     rn = (1, 0, 0)
#     hop_dict.empty(rn, (2, 2))
#     hop_dict.set_element(rn, (1, 0), t)
#     rn = (0, 1, 0)
#     hop_dict.empty(rn, (2, 2))
#     hop_dict.set_element(rn, (1, 0), t)
#
#     # Boundary condition
#     def pbc(cell_ind, orbital):
#         n0, n1, n2 = cell_ind
#         if enable_pbc:
#             return (n0 % shape[0], n1 % shape[1], n2), orbital
#         else:
#             return (n0, n1, n2), orbital
#
#     # Sample
#     sample = tipsi.Sample(lattice, site_set, pbc)
#     sample.add_hop_dict(hop_dict)
#     sample.rescale_H(9.0)
#     return sample


def make_sample_new(shape, enable_pbc=True):
    """Make sample for given shape and pbc with new version of builder."""
    a = 0.246
    vectors = [[0.5*a, 0.5*sqrt(3)*a, 0.],
               [-0.5*a, 0.5*sqrt(3)*a, 0.],
               [0.0, 0.0, 1.]]
    cell = PrimitiveCell(np.array(vectors), unit=1.0)
    cell.add_orbital((0.0,  0.0), 0.0)
    cell.add_orbital((1./3, 1./3), 0.0)
    cell.add_hopping((0, 0), 0, 1, 2.7)
    cell.add_hopping((1, 0), 1, 0, 2.7)
    cell.add_hopping((0, 1), 1, 0, 2.7)
    if enable_pbc:
        pbc = (True, True)
    else:
        pbc = (False, False)
    sc = SuperCell(cell, dim=shape, pbc=pbc)
    sample = Sample(sc)
    sample.rescale_ham(9.0)
    return sample


# def build_ham_csr_old(sample):
#     """Build CSR Hamiltonian for sample with old builder."""
#     shape = (sample.site_x.shape[0], sample.site_x.shape[0])
#     ham_csr = csr_matrix((sample.hop, sample.indices, sample.indptr), shape)
#     return ham_csr
#
#
# def build_dxy_csr_old(sample):
#     """Build CSR dx and dy for sample with older builder."""
#     shape = (sample.site_x.shape[0], sample.site_x.shape[0])
#     dx_csr = csr_matrix((sample.dx, sample.indices, sample.indptr), shape)
#     dy_csr = csr_matrix((sample.dy, sample.indices, sample.indptr), shape)
#     return dx_csr, dy_csr


def dump_csr(indptr, indices):
    """Print row and column indices of CSR matrix."""
    for i0 in range(indptr.shape[0]-1):
        for ptr in range(indptr[i0], indptr[i0+1]):
            print(i0, indices[ptr])


class TestSample(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    # def test00_builder(self):
    #     """
    #     Test if the old and new versions of builder are equivalent by comparing
    #     their attributes one-by-one.
    #
    #     :return: None.
    #     """
    #     sample_old = make_sample_old((100, 100), enable_pbc=True)
    #     sample_new = make_sample_new((100, 100), enable_pbc=True)
    #     th = TestHelper(self)
    #
    #     # 1. Lattice
    #     lattice = sample_old.lattice
    #     prim_cell = sample_new.sc_list[0].prim_cell
    #
    #     # 1.1 Vectors
    #     th.test_equal_array(prim_cell.lat_vec, lattice.vectors)
    #     th.test_equal_array(prim_cell.lat_vec.T, lattice.vectorsT)
    #
    #     # 1.2  Orbital positions
    #     th.test_equal_array(prim_cell.orb_pos,
    #                         lat.cart2frac(lattice.vectors,
    #                                       lattice.orbital_coords))
    #
    #     # 1.3  Other attributes
    #     self.assertEqual(sample_new.extended, lattice.extended)
    #
    #     # 1.4 Methods
    #     self.assertAlmostEqual(sample_new.area_unit_cell,
    #                            lattice.area_unit_cell())
    #     self.assertAlmostEqual(sample_new.volume_unit_cell,
    #                            lattice.volume_unit_cell())
    #     th.test_equal_array(prim_cell.get_reciprocal_vectors(),
    #                         lattice.reciprocal_latt(), almost=True)
    #
    #     # 2. Sample
    #     self.assertAlmostEqual(sample_new.rescale, sample_old.rescale)
    #
    #     # 2.1 Orbital information
    #     sample_new.init_orb_pos()
    #     th.test_equal_array(sample_new.sc_list[0].orb_id_pc,
    #                         np.array(sample_old.index_to_tag))
    #     th.test_equal_array(sample_new.orb_pos[:, 0],
    #                         sample_old.site_x, almost=True)
    #     th.test_equal_array(sample_new.orb_pos[:, 1],
    #                         sample_old.site_y, almost=True)
    #     th.test_equal_array(sample_new.orb_pos[:, 2],
    #                         sample_old.site_z, almost=True)
    #
    #     # 2.2 Sparse Hamiltonian from build_ham_csr
    #     ham_csr_new = sample_new.build_ham_csr()
    #     ham_csr_old = build_ham_csr_old(sample_old)
    #     th.test_equal_array(np.sort(ham_csr_new.indices),
    #                         np.sort(sample_old.indices))
    #     th.test_equal_array(ham_csr_new.indptr, sample_old.indptr)
    #     th.test_equal_array(ham_csr_new.data, sample_old.hop)
    #     self.assertAlmostEqual((ham_csr_new - ham_csr_old).sum(), 0.0)
    #
    #     # 2.3 dx and dy from build_dxr_csr
    #     dx_csr_new, dy_csr_new = sample_new.build_dxy_csr()
    #     dx_csr_old, dy_csr_old = build_dxy_csr_old(sample_old)
    #     self.assertEqual(dx_csr_new.shape, dx_csr_old.shape)
    #     self.assertEqual(dy_csr_new.shape, dy_csr_old.shape)
    #     self.assertAlmostEqual((dx_csr_new - dx_csr_old).sum(), 0.0)
    #     self.assertAlmostEqual((dy_csr_new - dy_csr_old).sum(), 0.0)
    #
    #     # 2.4 Hamiltonian, dx and dy from build_ham_dxy
    #     indptr, indices, hop, dx, dy = sample_new.build_ham_dxy()
    #     num_orb_sc = sample_new.sc_list[0].num_orb_sc
    #     shape = (num_orb_sc, num_orb_sc)
    #     ham_csr_new = csr_matrix((hop, indices, indptr), shape)
    #     dx_csr_new = csr_matrix((dx, indices, indptr), shape)
    #     dy_csr_new = csr_matrix((dy, indices, indptr), shape)
    #     self.assertAlmostEqual((ham_csr_new - ham_csr_old).sum(), 0.0)
    #     self.assertAlmostEqual((dx_csr_new - dx_csr_old).sum(), 0.0)
    #     self.assertAlmostEqual((dy_csr_new - dy_csr_old).sum(), 0.0)

    def test01_tipsi(self):
        """
        Test the new builder with real-case calculations.

        :return: None.
        """
        # sample = make_sample_old((100, 100), enable_pbc=True)
        sample = make_sample_new((100, 100), enable_pbc=True)

        config = tipsi.Config(sample)
        config.generic['nr_time_steps'] = 256
        config.generic['nr_random_samples'] = 4
        config.generic['energy_range'] = 20.
        config.generic['correct_spin'] = True
        config.dyn_pol['q_points'] = [[1., 0., 0.]]
        config.DC_conductivity['energy_limits'] = (-0.3, 0.3)
        config.LDOS['site_indices'] = [0]
        config.LDOS['delta'] = 0.1
        config.LDOS['recursion_depth'] = 2000
        config.save()

        # set config parameters
        config = tipsi.Config(sample)
        config.generic['nr_time_steps'] = 256
        config.generic['nr_random_samples'] = 4
        config.generic['energy_range'] = 20.
        config.generic['correct_spin'] = True
        config.dyn_pol['q_points'] = [[1., 0., 0.]]
        config.DC_conductivity['energy_limits'] = (-0.3, 0.3)
        config.LDOS['site_indices'] = [0]
        config.LDOS['delta'] = 0.1
        config.LDOS['recursion_depth'] = 2000
        config.save()

        # get DOS
        corr_DOS = tipsi.corr_DOS(sample, config)
        energies_DOS, DOS = tipsi.analyze_corr_DOS(config, corr_DOS)
        plt.plot(energies_DOS, DOS)
        plt.xlabel("E (eV)")
        plt.ylabel("DOS")
        plt.savefig("DOS.png")
        plt.close()

        # get AC conductivity
        corr_AC = tipsi.corr_AC(sample, config)
        omegas_AC, AC = tipsi.analyze_corr_AC(config, corr_AC)
        plt.plot(omegas_AC, AC[0])
        plt.xlabel("hbar * omega (eV)")
        plt.ylabel("sigma_xx (sigma_0)")
        plt.savefig("ACxx.png")
        plt.close()

        # get dyn pol
        corr_dyn_pol = tipsi.corr_dyn_pol(sample, config)
        qval, omegas, dyn_pol = tipsi.analyze_corr_dyn_pol(config, corr_dyn_pol)
        qval, omegas, epsilon = tipsi.analyze_corr_dyn_pol(config, dyn_pol)
        plt.plot(omegas, -1 * dyn_pol[0, :].imag)
        plt.xlabel("hbar * omega (eV)")
        plt.ylabel("-Im(dp)")
        plt.savefig("dp_imag.png")
        plt.close()

        # get DC conductivity
        corr_DOS, corr_DC = tipsi.corr_DC(sample, config)
        energies_DC, DC = tipsi.analyze_corr_DC(config, corr_DOS, corr_DC)
        plt.plot(energies_DC, DC[0, :])
        plt.xlabel("E (eV)")
        plt.ylabel("DC conductivity")
        plt.savefig("DC.png")
        plt.close()


if __name__ == "__main__":
    unittest.main()
