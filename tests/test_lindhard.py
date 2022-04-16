#! /usr/bin/env python

import unittest

import numpy as np
import scipy.linalg.lapack as lapack
import matplotlib.pyplot as plt

import tbplas as tb
import tbplas.builder.core as core
from tbplas.utils import TestHelper


class TestLindhard(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_init(self):
        """
        Test the initialization procedure of 'Lindhard' class.

        :return: None.
        """
        prim_cell = tb.make_graphene_diamond()
        kmesh_size = (2, 3, 2)
        lindhard = tb.Lindhard(prim_cell, energy_max=10.0, energy_step=10,
                               kmesh_size=kmesh_size)

        th = TestHelper(self)
        omegas = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                           10.0])
        kmesh_grid = np.array([
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1),
            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)])
        th.test_equal_array(omegas, lindhard.omegas)
        th.test_equal_array(kmesh_grid, lindhard.kmesh_grid)

    def test_grid_conversion(self):
        """
        Test functions of converting grid coordinates.

        :return: None
        """
        lattice = tb.gen_lattice_vectors(a=10, b=8, c=6)
        prim_cell = tb.PrimitiveCell(lattice, unit=tb.ANG)
        kmesh_size = (5, 3, 2)
        lindhard = tb.Lindhard(prim_cell, energy_max=10.0, energy_step=10,
                               kmesh_size=kmesh_size)

        def _test_grid2frac():
            frac_coord_test = lindhard.grid2frac(grid_coord)
            th.test_equal_array(frac_coord_ref, frac_coord_test)

        def _test_grid2cart_nm():
            recip_lattice = tb.gen_reciprocal_vectors(prim_cell.lat_vec)
            cart_coord_nm_ref = tb.frac2cart(recip_lattice, frac_coord_ref)
            cart_coord_nm_test = lindhard.grid2cart(grid_coord, unit=tb.NM)
            th.test_equal_array(cart_coord_nm_ref, cart_coord_nm_test)

        def _test_grid2cart_ang():
            recip_lattice = tb.gen_reciprocal_vectors(prim_cell.lat_vec * 10)
            cart_coord_ang_ref = tb.frac2cart(recip_lattice, frac_coord_ref)
            cart_coord_ang_test = lindhard.grid2cart(grid_coord, unit=tb.ANG)
            th.test_equal_array(cart_coord_ang_ref, cart_coord_ang_test)

        # Test with wrap=True
        th = TestHelper(self)
        grid_coord = np.array([(0, 0, 0), (1, 1, 0), (5, 0, 1),
                               (1, 3, 2), (2, 3, 3), (1, 1, 1)])
        frac_coord_ref = np.array([
            [0/5, 0/3, 0/2], [1/5, 1/3, 0/2], [5/5, 0/3, 1/2],
            [1/5, 3/3, 2/2], [2/5, 3/3, 3/2], [1/5, 1/3, 1/2],
        ])
        _test_grid2frac()
        _test_grid2cart_nm()
        _test_grid2cart_ang()

    def test_cart_frac_conversion(self):
        """
        Test frac2cart and cart2frac.

        :return: None
        """
        lattice = tb.gen_lattice_vectors(a=10, b=8, c=6)
        prim_cell = tb.PrimitiveCell(lattice, unit=tb.ANG)
        kmesh_size = (5, 3, 2)
        lindhard = tb.Lindhard(prim_cell, energy_max=10.0, energy_step=10,
                               kmesh_size=kmesh_size)

        th = TestHelper(self)
        q_frac = np.array([[0.15, -0.13, 0.35]])
        q_cart_ang_ref = 2 * np.pi * np.array([[0.15/10, -0.13/8, 0.35/6]])
        q_cart_nm_ref = q_cart_ang_ref * 10
        q_cart_ang_test = lindhard.frac2cart(q_frac, unit=tb.ANG)
        q_cart_nm_test = lindhard.frac2cart(q_frac, unit=tb.NM)
        th.test_equal_array(q_cart_ang_test, q_cart_ang_ref, almost=True)
        th.test_equal_array(q_cart_nm_test, q_cart_nm_ref, almost=True)

        q_cart_nm = np.array([[2.3, -1.5, 0.7]])
        q_frac_ref = 1 / (2 * np.pi) * np.array([[2.3, -1.5*0.8, 0.7*0.6]])
        q_frac_test = lindhard.cart2frac(q_cart_nm, unit=tb.NM)
        th.test_equal_array(q_frac_test, q_frac_ref, almost=True)

        q_cart_ang = np.array([[2.3, -1.5, 0.7]])
        q_frac_ref = 1 / (2 * np.pi) * np.array([2.3*10, -1.5*8, 0.7*6])
        q_frac_test = lindhard.cart2frac(q_cart_ang, unit=tb.ANG)
        th.test_equal_array(q_frac_test, q_frac_ref, almost=True)

    def test_get_eigenstates(self):
        """
        Test _get_eigenstates.

        :return: None
        """
        def _calc_bands(cell, k_points):
            cell.sync_array()
            num_k_points = k_points.shape[0]
            bands = np.zeros((num_k_points, cell.num_orb), dtype=np.float64)
            states = np.zeros((num_k_points, cell.num_orb, cell.num_orb),
                              dtype=np.complex128)
            ham_k = np.zeros((cell.num_orb, cell.num_orb), dtype=np.complex128)

            for i_k, k_point in enumerate(k_points):
                ham_k *= 0.0
                core.set_ham(cell.orb_pos, cell.orb_eng,
                             cell.hop_ind, cell.hop_eng,
                             k_point, ham_k)
                eigenvalues, eigenstates, info = lapack.zheev(ham_k)
                bands[i_k] = eigenvalues
                states[i_k] = eigenstates.T
            return bands, states

        th = TestHelper(self)
        prim_cell = tb.make_graphene_diamond()
        kmesh_size = (6, 6, 2)
        lindhard = tb.Lindhard(prim_cell, energy_max=10.0, energy_step=10,
                               kmesh_size=kmesh_size)
        k_grid_frac = lindhard.grid2frac(lindhard.kmesh_grid)
        bands_ref, states_ref = _calc_bands(prim_cell, k_grid_frac)
        bands_test, states_test = lindhard._get_eigen_states(k_grid_frac,
                                                             convention=1)
        th.test_equal_array(bands_ref, bands_test)
        th.test_equal_array(states_ref, states_test)

    def test_kq_map(self):
        """
        Test the algorithm to remap k+q to q.

        :return: None.
        """
        lattice = tb.gen_lattice_vectors(a=10, b=10, c=10)
        prim_cell = tb.PrimitiveCell(lattice, unit=tb.ANG)
        kmesh_size = (2, 3, 2)
        lindhard = tb.Lindhard(prim_cell, energy_max=10.0, energy_step=10,
                               kmesh_size=kmesh_size)

        th = TestHelper(self)
        q_point = np.array([3, 1, 2], dtype=np.int64)
        kq_map_test = core.build_kq_map(lindhard.kmesh_size,
                                        lindhard.kmesh_grid, q_point)

        # Hand-crafted reference
        kq_map_ref = np.array([8, 9, 10, 11, 6, 7, 2, 3, 4, 5, 0, 1])
        th.test_equal_array(kq_map_ref, kq_map_test)

        # Auto-generated reference
        kq_map_ref = []
        k_mesh_grid = [tuple(row) for row in lindhard.kmesh_grid]
        for k_point in k_mesh_grid:
            kq_a = (k_point[0] + q_point[0]) % kmesh_size[0]
            kq_b = (k_point[1] + q_point[1]) % kmesh_size[1]
            kq_c = (k_point[2] + q_point[2]) % kmesh_size[2]
            kq_map_ref.append(k_mesh_grid.index((kq_a, kq_b, kq_c)))
        kq_map_ref = np.array(kq_map_ref, dtype=np.int64)
        th.test_equal_array(kq_map_ref, kq_map_test)

    def test_eigen_remap(self):
        """
        Check if the energies and wave functions are correctly remapped for
        dyn_pol_regular.

        :return: None
        """
        prim_cell = tb.make_graphene_diamond()
        kmesh_size = (6, 6, 1)
        lindhard = tb.Lindhard(prim_cell, energy_max=10.0, energy_step=10,
                               kmesh_size=kmesh_size)

        # Get k-grid and k+q grid
        q_point = np.array([2, 3, 0], dtype=np.int64)
        k_grid = lindhard.kmesh_grid
        kq_grid = q_point + k_grid
        k_frac = lindhard.grid2frac(k_grid)
        kq_frac = lindhard.grid2frac(kq_grid)

        # Get energies and wave functions
        bands_k, states_k = lindhard._get_eigen_states(k_frac, convention=2)
        band_kq, states_kq = lindhard._get_eigen_states(kq_frac, convention=2)

        # Compare results
        kq_map = core.build_kq_map(lindhard.kmesh_size, lindhard.kmesh_grid,
                                   q_point)
        th = TestHelper(self)
        th.test_equal_array(bands_k[kq_map], band_kq, almost=True)

    def test_dyn_pol_consistency(self):
        """
        Test if dyn_pol produced by cython and fortran backends, and by
        regular/arbitrary q-point algorithms are consistent.

        :return: None
        """
        cell = tb.make_graphene_diamond()
        energy_max = 10
        energy_step = 2048
        mu = 0.0
        temp = 300
        back_epsilon = 1
        mesh_size = (120, 120, 1)
        q_points_grid = np.array([[100, 100, 0]], dtype=np.int64)
        q_points_cart = np.array([[21.28450307, 12.28861358, 0.]])
        th = TestHelper(self)

        lindhard = tb.Lindhard(cell=cell, energy_max=energy_max,
                               energy_step=energy_step, kmesh_size=mesh_size,
                               mu=mu, temperature=temp,
                               back_epsilon=back_epsilon)

        # Test regular algorithm of cython and fortran versions
        dyn_pol_c = lindhard.calc_dyn_pol_regular(q_points_grid,
                                                  use_fortran=False)[1]
        dyn_pol_f = lindhard.calc_dyn_pol_regular(q_points_grid,
                                                  use_fortran=True)[1]
        th.test_equal_array(dyn_pol_c, dyn_pol_f, almost=True)

        # Test arbitrary algorithm of cython and fortran versions
        dyn_pol_c = lindhard.calc_dyn_pol_arbitrary(q_points_cart,
                                                    use_fortran=False)[1]
        dyn_pol_f = lindhard.calc_dyn_pol_arbitrary(q_points_cart,
                                                    use_fortran=True)[1]
        th.test_equal_array(dyn_pol_c, dyn_pol_f, almost=True)

        # Test regular and arbitrary algorithms
        dyn_pol_reg = lindhard.calc_dyn_pol_regular(q_points_grid,
                                                    use_fortran=True)[1]
        dyn_pol_arb = lindhard.calc_dyn_pol_arbitrary(q_points_cart,
                                                      use_fortran=True)[1]
        diff = np.sum(np.abs(dyn_pol_reg - dyn_pol_arb)).item(0)
        self.assertAlmostEqual(diff, 0.0, delta=1e-3)

    def test_dyn_pol_prb(self):
        """
        Reproducing Phys. Rev. B 84, 035439 (2011) with |q| = 1/a and theta = 30
        degrees.

        :return: None
        """
        # Construct primitive cell
        t = 3.0  # Absolute hopping energy in eV
        a = 0.142  # C-C distance in NM
        vectors = tb.gen_lattice_vectors(a=0.246, b=0.246, c=1.0, gamma=60)
        cell = tb.PrimitiveCell(vectors, unit=tb.NM)
        cell.add_orbital([0.0, 0.0], label="C_pz")
        cell.add_orbital([1 / 3., 1 / 3.], label="C_pz")
        cell.add_hopping([0, 0], 0, 1, t)
        cell.add_hopping([1, 0], 1, 0, t)
        cell.add_hopping([0, 1], 1, 0, t)

        # Set parameter for Lindhard function
        energy_max = 10
        energy_step = 2048
        mu = 0.0
        temp = 300
        back_epsilon = 1
        mesh_size = (1200, 1200, 1)
        use_fortran = True
        q_points = 1 / a * np.array([[0.86602540, 0.5, 0.0]])

        # Instantiate Lindhard calculator
        lindhard = tb.Lindhard(cell=cell, energy_max=energy_max,
                               energy_step=energy_step, kmesh_size=mesh_size,
                               mu=mu, temperature=temp, g_s=1,
                               back_epsilon=back_epsilon)

        # Calculate and plot dyn_pol
        omegas, dyn_pol = lindhard.calc_dyn_pol_arbitrary(q_points, use_fortran)
        for i in range(len(q_points)):
            plt.plot(omegas/t, -dyn_pol.imag[i]*t*a**2)
        plt.savefig("dyn_pol.png")
        plt.close()

    def test_epsilon_consistency(self):
        """
        Test if regular and arbitrary algorithms produce consistent results.

        :return: None
        """
        cell = tb.make_graphene_diamond()
        energy_max = 10
        energy_step = 2048
        mu = 0.0
        temp = 300
        back_epsilon = 1
        mesh_size = (120, 120, 1)
        q_points_grid = np.array([[100, 100, 0]], dtype=np.int64)
        q_points_cart = np.array([[21.28450307, 12.28861358, 0.]])

        lindhard = tb.Lindhard(cell=cell, energy_max=energy_max,
                               energy_step=energy_step, kmesh_size=mesh_size,
                               mu=mu, temperature=temp,
                               back_epsilon=back_epsilon)
        eps_ref = lindhard.calc_epsilon_regular(q_points_grid)[1]
        eps_test = lindhard.calc_epsilon_arbitrary(q_points_cart)[1]
        diff = np.sum(np.abs(eps_test - eps_ref)).item(0)
        self.assertAlmostEqual(diff, 0.0, delta=1e-3)

    def test_epsilon_prb(self):
        """
        Reproducing Phys. Rev. B 84, 035439 (2011) with |q| = 0.476 / Angstrom
        and theta = 30 degrees.

        :return: None
        """
        # Construct primitive cell
        t = 3.0  # Absolute hopping energy in eV
        vectors = tb.gen_lattice_vectors(a=0.246, b=0.246, c=1.0, gamma=60)
        cell = tb.PrimitiveCell(vectors, unit=tb.NM)
        cell.add_orbital([0.0, 0.0], label="C_pz")
        cell.add_orbital([1 / 3., 1 / 3.], label="C_pz")
        cell.add_hopping([0, 0], 0, 1, t)
        cell.add_hopping([1, 0], 1, 0, t)
        cell.add_hopping([0, 1], 1, 0, t)

        # Set parameter for Lindhard function
        energy_max = 10
        energy_step = 2048
        mu = 0.0
        temp = 300
        back_epsilon = 1
        mesh_size = (1200, 1200, 1)
        use_fortran = True
        q_points = np.array([[4.122280922013927, 2.38, 0.0]])

        # Instantiate Lindhard calculator
        lindhard = tb.Lindhard(cell=cell, energy_max=energy_max,
                               energy_step=energy_step, kmesh_size=mesh_size,
                               mu=mu, temperature=temp, g_s=1,
                               back_epsilon=back_epsilon)

        # Evaluate dielectric function
        omegas, epsilon = lindhard.calc_epsilon_arbitrary(q_points, use_fortran)

        # Plot
        for i in range(len(q_points)):
            plt.plot(omegas, epsilon[i].real, color="r")
        plt.minorticks_on()
        plt.savefig("epsilon.png")
        plt.close()

    def test_ac_cond(self):
        """
        Calculate the AC conductivity of monolayer graphene.

        :return: None
        """
        cell = tb.make_graphene_diamond()

        energy_max = 10
        energy_step = 2048
        mu = 0.0
        temp = 300
        back_epsilon = 1.0
        mesh_size = (2048, 2048, 1)

        lindhard = tb.Lindhard(cell=cell, energy_max=energy_max,
                               energy_step=energy_step, kmesh_size=mesh_size,
                               mu=mu, temperature=temp,
                               back_epsilon=back_epsilon, delta=0.005, g_s=2)

        omegas, ac_cond = lindhard.calc_ac_cond_kg()
        omegas /= 2.7

        plt.plot(omegas, ac_cond.real, color="r")
        plt.minorticks_on()
        plt.savefig("sigma_xx.png")
        plt.close()


if __name__ == "__main__":
    unittest.main()
