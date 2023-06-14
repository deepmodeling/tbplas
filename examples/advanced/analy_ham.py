#! /usr/bin/env python
"""
Example for the derivation and usage of analytical Hamiltonian.
"""

from math import cos, sin, pi
import numpy as np
import tbplas as tb


def exp_i(x: float) -> complex:
    """
    Evaluate exp(i*2pi*x) using Euler formula.

    :param x: incoming x
    :return: exp(i*2pi*x)
    """
    return cos(2 * pi * x) + 1j * sin(2 * pi * x)


def hk1(kpt: np.ndarray, ham: np.ndarray) -> None:
    """
    Analytical Hamiltonian modifying ham in-place.

    :param kpt: (3,) float64 array
        fractional coordinate of k-points
    :param ham: (num_orb, num_orb) complex128 array
        Hamiltonian matrix
    :return: None
    """
    ka, kb = kpt.item(0), kpt.item(1)
    ham[0, 0] = 0.0
    ham[1, 1] = 0.0
    ham[0, 1] = -2.7 * (exp_i(1. / 3 * ka + 1. / 3 * kb) +
                        exp_i(-2. / 3 * ka + 1. / 3 * kb) +
                        exp_i(1. / 3 * ka - 2. / 3 * kb))
    ham[1, 0] = ham[0, 1].conjugate()


def hk2(kpt: np.ndarray, ham: np.ndarray) -> None:
    """
    Analytical Hamiltonian modifying ham in-place.

    :param kpt: (3,) float64 array
        fractional coordinate of k-points
    :param ham: (num_orb, num_orb) complex128 array
        Hamiltonian matrix
    :return: None
    """
    ka, kb = kpt.item(0), kpt.item(1)
    ham[0, 0] = 0.0
    ham[1, 1] = 0.0
    ham[0, 1] = -2.7 * (1.0 + exp_i(-ka) + exp_i(-kb))
    ham[1, 0] = ham[0, 1].conjugate()


def main():
    # Print the analytical Hamiltonian of monolayer graphene
    cell = tb.make_graphene_diamond()
    print("---- convention 1 ----")
    cell.print_hk(convention=1)
    print("---- convention 2 ----")
    cell.print_hk(convention=2)

    # Create a cell without hopping terms
    vectors = tb.gen_lattice_vectors(a=0.246, b=0.246, c=1.0, gamma=60)
    cell = tb.PrimitiveCell(vectors, unit=tb.NM)
    cell.add_orbital((0.0, 0.0), label="C_pz")
    cell.add_orbital((1/3., 1/3.), label="C_pz")

    # Usage of analytical Hamiltonian
    k_points = np.array([
        [0.0, 0.0, 0.0],
        [2. / 3, 1. / 3, 0.0],
        [1. / 2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    k_label = ["G", "M", "K", "G"]
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    for hk in (hk1, hk2):
        solver = tb.DiagSolver(cell, hk_dense=hk)
        k_len, bands = solver.calc_bands(k_path)[:2]
        vis = tb.Visualizer()
        vis.plot_bands(k_len, bands, k_idx, k_label)

    # Evaluation of DOS
    k_mesh = tb.gen_kmesh((120, 120, 1))
    for hk in (hk1, hk2):
        solver = tb.DiagSolver(cell, hk_dense=hk)
        energies, dos = solver.calc_dos(k_mesh)
        vis = tb.Visualizer()
        vis.plot_dos(energies, dos)


if __name__ == "__main__":
    main()
