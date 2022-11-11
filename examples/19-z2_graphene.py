#! /usr/bin/env python

from math import sqrt, pi

import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb


def make_hop_dict(t, lamb_so, lamb_r):
    def _hop_1st(vec):
        a = vec[1] + 1j * vec[0]
        ac = vec[1] - 1j * vec[0]
        return np.array([[t, 1j * a * lamb_r], [1j * ac * lamb_r, t]])

    def _hop_2nd(vec0, vec1):
        b = 2. / sqrt(3.) * np.cross(vec0, vec1)
        return np.array([[1j * b[2] * lamb_so, 0.], [0., -1j * b[2] * lamb_so]])

    # Carbon-carbon vectors
    ac_vec = np.array([[1., 0., 0.],
                       [-0.5, np.sqrt(3.) / 2., 0.],
                       [-0.5, -np.sqrt(3.) / 2., 0.]])
    bc_vec = np.array([[-1., 0., 0.],
                       [0.5, -np.sqrt(3.) / 2., 0.],
                       [0.5, np.sqrt(3.) / 2., 0.]])

    # Initialize hop_dict
    hop_dict = tb.HopDict(4)

    # 1st nearest neighbours
    # (0, 0, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 2:4] = _hop_1st(ac_vec[0])
    hop_dict.set_mat((0, 0, 0), hop_mat)

    # (-1, 0, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 2:4] = _hop_1st(ac_vec[1])
    hop_dict.set_mat((-1, 0, 0), hop_mat)

    # (0, -1, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 2:4] = _hop_1st(ac_vec[2])
    hop_dict.set_mat((0, -1, 0), hop_mat)

    # 2nd nearest neighbours
    # (0, 1, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 0:2] = _hop_2nd(ac_vec[0], bc_vec[2])
    hop_mat[2:4, 2:4] = _hop_2nd(bc_vec[2], ac_vec[0])
    hop_dict.set_mat((0, 1, 0), hop_mat)

    # (1, 0, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 0:2] = _hop_2nd(ac_vec[0], bc_vec[1])
    hop_mat[2:4, 2:4] = _hop_2nd(bc_vec[1], ac_vec[0])
    hop_dict.set_mat((1, 0, 0), hop_mat)

    # (1, -1, 0) cell
    hop_mat = np.zeros((4, 4), dtype=complex)
    hop_mat[0:2, 0:2] = _hop_2nd(ac_vec[2], bc_vec[1])
    hop_mat[2:4, 2:4] = _hop_2nd(bc_vec[1], ac_vec[2])
    hop_dict.set_mat((1, -1, 0), hop_mat)

    return hop_dict


def main():
    # Reference:
    # https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.075119

    # Parameters
    lat = 1.
    t = -1.
    lamb_so = 0.06 * t
    lamb_r = 0.05 * t

    # QSH phase
    lamb_nu = 0.1 * t

    # normal insulating phase
    # lamb_nu = 0.4 * t

    # Whether to reorder the phases for improving continuity and smoothness
    # CAUTION: this operation may fail!
    reorder_phases = False

    # Lattice
    vectors = np.array([[0.5 * lat * sqrt(3), -0.5 * lat, 0.],
                        [0.5 * lat * sqrt(3), 0.5 * lat, 0.],
                        [0, 0, 1]])
    prim_cell = tb.PrimitiveCell(vectors)

    # Add orbitals
    prim_cell.add_orbital([0, 0], lamb_nu)
    prim_cell.add_orbital([0, 0], lamb_nu)
    prim_cell.add_orbital([1./3, 1./3], -lamb_nu)
    prim_cell.add_orbital([1./3, 1./3], -lamb_nu)

    # Add hopping terms
    hop_dict = make_hop_dict(t, lamb_so, lamb_r)
    prim_cell.add_hopping_dict(hop_dict)

    # Plot model
    prim_cell.plot(hop_as_arrows=False)

    # Evaluate band structure
    k_points = np.array([
        [0.0, 0.0, 0.0],
        [1. / 2, 0.0, 0.0],
        [2. / 3, 1. / 3, 0.0],
        [0.0, 0.0, 0.0],
    ])
    k_label = ["G", "M", "K", "G"]
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    k_len, bands = prim_cell.calc_bands(k_path)
    vis = tb.Visualizer()
    vis.plot_bands(k_len, bands, k_idx, k_label)

    # Get phases
    ka_array = np.linspace(-0.5, 0.5, 200)
    kb_array = np.linspace(0.0, 0.5, 200)
    kc = 0.0
    z2 = tb.Z2(prim_cell, num_occ=2)
    kb_array, phases = z2.calc_phases(ka_array, kb_array, kc)
    if reorder_phases:
        phases = z2.reorder_phases(phases)
        num_crossing = z2.count_crossing(phases, phase_ref=0.2)
        print(f"Number of crossing: {num_crossing}")

    # Regular plot
    fig, ax = plt.subplots()
    for i in range(2):
        if reorder_phases:
            ax.plot(kb_array, phases[:, i] / pi)
        else:
            ax.scatter(kb_array, phases[:, i] / pi, s=1, c="r")
    ax.grid()
    plt.show()
    plt.close()

    # Polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(2):
        if reorder_phases:
            ax.plot(phases[:, i], kb_array)
        else:
            ax.scatter(phases[:, i], kb_array, s=1, c="r")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
