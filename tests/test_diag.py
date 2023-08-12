#! /usr/bin/env python

import unittest
from math import cos, sin, pi, sqrt

import numpy as np
from scipy.sparse import csr_matrix

import tbplas as tb


def exp_i(x):
    return cos(x) + 1j * sin(x)


def exp_i2(x):
    return cos(2 * pi * x) + 1j * sin(2 * pi * x)


class FakePC(tb.FakePC):
    def set_ham_dense(self, kpt: np.ndarray,
                      ham: np.ndarray,
                      convention: int = 1) -> None:
        if convention == 1:
            # Working in Cartesian coordinates
            a, t, sqrt3 = 0.246, 2.7, sqrt(3.0)
            recip_lat = self.get_reciprocal_vectors()
            ka = np.matmul(kpt, recip_lat) * a
            kxa, kya = ka.item(0), ka.item(1)
            fk = exp_i(kya / sqrt3) + 2 * exp_i(-kya / 2 / sqrt3) * cos(kxa / 2)
            ham[0, 1] = t * fk
            ham[1, 0] = t * fk.conjugate()
        elif convention == 2:
            # Working in fractional coordinates, convention 1
            ka, kb = kpt.item(0), kpt.item(1)
            ham[0, 1] = 2.7 * (exp_i2(1./3 * ka + 1./3 * kb) +
                               exp_i2(-2./3 * ka + 1./3 * kb) +
                               exp_i2(1./3 * ka - 2./3 * kb))
            ham[1, 0] = ham[0, 1].conjugate()
        else:
            # Working in fractional coordinates, convention 2
            ka, kb = kpt.item(0), kpt.item(1)
            ham[0, 1] = 2.7 * (1 + exp_i2(-ka) + exp_i2(-kb))
            ham[1, 0] = ham[0, 1].conjugate()


class FakeOverlap:
    def __init__(self, num_orb: int) -> None:
        self._num_orb = num_orb

    def set_overlap_dense(self, k_point: np.ndarray,
                          overlap_dense: np.ndarray,
                          convention: int = 1):
        for i in range(self._num_orb):
            overlap_dense[i, i] = 1.0

    def set_overlap_csr(self, k_point: np.ndarray,
                        convention: int = 1) -> csr_matrix:
        return csr_matrix(np.eye(self._num_orb))


class MyTest(unittest.TestCase):

    def test_analytical_ham(self):
        """Test analytical Hamiltonian."""
        cell = tb.make_graphene_diamond()
        cell = FakePC(cell.num_orb, cell.lat_vec, unit=tb.NM)
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
        for convention in (1, 2, 3):
            k_len, bands = solver.calc_bands(k_path, convention)[:2]
            vis.plot_bands(k_len, bands, k_idx, k_label)

        # DOS
        k_mesh = tb.gen_kmesh((120, 120, 1))
        for convention in (1, 2, 3):
            energies, dos = solver.calc_dos(k_mesh, convention=convention)
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
        overlap = FakeOverlap(cell.num_orb)
        solver = tb.DiagSolver(cell, overlap)
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
        overlap = FakeOverlap(cell.num_orb)
        solver = tb.DiagSolver(cell, overlap)
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

        # DOS
        k_mesh = tb.gen_kmesh((24, 24, 1))
        energies, dos = solver.calc_dos(k_mesh, solver="arpack", k=50)
        vis.plot_dos(energies, dos)

    def test_derived_overlap(self):
        """Test derived classes of DiagSolver with overlap."""
        # Z2
        cell = tb.make_graphene_soc(is_qsh=True)
        overlap = FakeOverlap(cell.num_orb)
        z2 = tb.Z2(cell, num_occ=2, overlap=overlap)
        vis = tb.Visualizer()
        ka_array = np.linspace(-0.5, 0.5, 200)
        kb_array = np.linspace(0.0, 0.5, 200)
        kc = 0.0
        kb_array, phases = z2.calc_phases(ka_array, kb_array, kc)
        vis.plot_phases(kb_array, phases)

        # Spin texture
        k_grid = 2 * tb.gen_kmesh((240, 240, 1)) - 1
        spin_texture = tb.SpinTexture(cell, k_grid, spin_major=False,
                                      overlap=overlap)
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
        overlap = FakeOverlap(cell.num_orb)
        t = 3.0
        lind = tb.Lindhard(cell=cell, energy_max=t * 3.5, energy_step=2048,
                           kmesh_size=(600, 600, 1), mu=0.0, temperature=300.0,
                           g_s=2, back_epsilon=1.0, dimension=2,
                           overlap=overlap)
        omegas, ac_cond = lind.calc_ac_cond(component="xx", use_fortran=True)
        omegas /= 4
        ac_cond *= 4
        vis.plot_xy(omegas, ac_cond.real)


if __name__ == "__main__":
    unittest.main()
