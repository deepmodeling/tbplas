#! /usr/bin/env python

import unittest
from math import cos, sin, pi, sqrt

import numpy as np

import tbplas as tb


class MyTest(unittest.TestCase):

    def test_analytical_ham(self):
        """Test analytical Hamiltonian."""
        def _exp(x):
            return cos(x) + 1j * sin(x)

        def _exp2(x):
            return cos(2 * pi * x) + 1j * sin(2 * pi * x)

        def _hk1(kpt, ham):
            # Working in Cartesian coordinates
            a, t, sqrt3 = 0.246, 2.7, sqrt(3.0)
            recip_lat = cell.get_reciprocal_vectors()
            ka = np.matmul(kpt, recip_lat) * a
            kxa, kya = ka.item(0), ka.item(1)
            fk = _exp(kya / sqrt3) + 2 * _exp(-kya / 2 / sqrt3) * cos(kxa / 2)
            ham[0, 1] = t * fk
            ham[1, 0] = t * fk.conjugate()

        def _hk2(kpt, ham):
            # Working in fractional coordinates, convention 1
            ka, kb = kpt.item(0), kpt.item(1)
            ham[0, 1] = 2.7 * (_exp2(1./3 * ka + 1./3 * kb) +
                               _exp2(-2./3 * ka + 1./3 * kb) +
                               _exp2(1./3 * ka - 2./3 * kb))
            ham[1, 0] = ham[0, 1].conjugate()

        def _hk3(kpt, ham):
            # Working in fractional coordinates, convention 2
            ka, kb = kpt.item(0), kpt.item(1)
            ham[0, 1] = 2.7 * (1 + _exp2(-ka) + _exp2(-kb))
            ham[1, 0] = ham[0, 1].conjugate()

        cell = tb.make_graphene_diamond()
        vis = tb.Visualizer()

        # Bands
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
        k_label = ["G", "K", "M", "G"]
        for _hk in (_hk1, _hk2, _hk3):
            solver = tb.DiagSolver(cell, hk_dense=_hk)
            k_len, bands = solver.calc_bands(k_path)[:2]
            vis.plot_bands(k_len, bands, k_idx, k_label)

        # DOS
        k_mesh = tb.gen_kmesh((120, 120, 1))
        for _hk in (_hk1, _hk2, _hk3):
            solver = tb.DiagSolver(cell, hk_dense=_hk)
            energies, dos = solver.calc_dos(k_mesh)
            vis.plot_dos(energies, dos)

    def test_sparse(self):
        """Test algorithms for sparse Hamiltonian."""
        cell = tb.make_graphene_diamond()
        cell = tb.extend_prim_cell(cell, dim=(5, 5, 1))
        solver = tb.DiagSolver(cell)
        vis = tb.Visualizer()

        # Bands
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
        k_label = ["G", "K", "M", "G"]

        # Dense Hamiltonian for reference
        k_len, bands = solver.calc_bands(k_path)[:2]
        vis.plot_bands(k_len, bands, k_idx, k_label)

        # Sparse Hamiltonian
        k_len, bands = solver.calc_bands(k_path, solver="arpack",
                                         which="LA")[:2]
        vis.plot_bands(k_len, bands, k_idx, k_label)

        # DOS
        k_mesh = tb.gen_kmesh((24, 24, 1))
        energies, dos = solver.calc_dos(k_mesh)
        vis.plot_dos(energies, dos)
        energies, dos = solver.calc_dos(k_mesh, solver="arpack",
                                        which="LA")
        vis.plot_dos(energies, dos)

    def test_dense_overlap(self):
        """
        Test non-orthogonal basis with overlap matrix for dense Hamiltonian.
        """
        cell = tb.make_graphene_diamond()
        solver = tb.DiagSolver(cell, s_mat=np.eye(2))
        vis = tb.Visualizer()

        # Bands
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = solver.calc_bands(k_path)[:2]
        vis.plot_bands(k_len, bands, k_idx, k_label=["G", "K", "M", "G"])

        # DOS
        k_mesh = tb.gen_kmesh((120, 120, 1))
        energies, dos = solver.calc_dos(k_mesh)
        vis.plot_dos(energies, dos)

    def test_sparse_overlap(self):
        """
        Test non-orthogonal basis with overlap matrix for sparse Hamiltonian.
        """
        cell = tb.make_graphene_diamond()
        cell = tb.extend_prim_cell(cell, dim=(5, 5, 1))
        solver = tb.DiagSolver(cell, s_mat=np.eye(cell.num_orb))
        vis = tb.Visualizer()

        # Bands
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
        k_label = ["G", "K", "M", "G"]
        k_len, bands = solver.calc_bands(k_path, solver="arpack", k=50)[:2]
        vis.plot_bands(k_len, bands, k_idx, k_label)

    def test_derived_overlap(self):
        """Test derived classes of DiagSolver with overlap."""
        # Z2
        cell = tb.make_graphene_soc(is_qsh=True)
        z2 = tb.Z2(cell, num_occ=2, s_mat=np.eye(cell.num_orb))
        vis = tb.Visualizer()
        ka_array = np.linspace(-0.5, 0.5, 200)
        kb_array = np.linspace(0.0, 0.5, 200)
        kc = 0.0
        kb_array, phases = z2.calc_phases(ka_array, kb_array, kc)
        vis.plot_phases(kb_array, phases)

        # Spin texture
        k_grid = 2 * tb.gen_kmesh((240, 240, 1)) - 1
        spin_texture = tb.SpinTexture(cell, k_grid, spin_major=False,
                                      s_mat=np.eye(cell.num_orb))
        k_cart = spin_texture.k_cart
        sz = spin_texture.eval("z")
        vis.plot_scalar(x=k_cart[:, 0], y=k_cart[:, 1], z=sz[:, 2],
                        num_grid=(480, 480), cmap="jet")
        k_grid = 2 * tb.gen_kmesh((48, 48, 1)) - 1
        spin_texture.k_grid = k_grid
        k_cart = spin_texture.k_cart
        sx = spin_texture.eval("x")
        sy = spin_texture.eval("y")
        vis.plot_vector(x=k_cart[:, 0], y=k_cart[:, 1], u=sx[:, 2], v=sy[:, 2])

        # AC conductivity
        cell = tb.make_graphene_diamond()
        t = 3.0
        lind = tb.Lindhard(cell=cell, energy_max=t * 3.5, energy_step=2048,
                           kmesh_size=(600, 600, 1), mu=0.0, temperature=300.0,
                           g_s=2, back_epsilon=1.0, dimension=2,
                           s_mat=np.eye(cell.num_orb))
        omegas, ac_cond = lind.calc_ac_cond(component="xx", use_fortran=True)
        omegas /= 4
        ac_cond *= 4
        vis.plot_xy(omegas, ac_cond.real)


if __name__ == "__main__":
    unittest.main()
