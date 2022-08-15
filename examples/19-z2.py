#! /usr/bin/env python

from math import pi, sqrt

import numpy as np
import scipy.linalg.lapack as spla
import matplotlib.pyplot as plt

import tbplas as tb
from tbplas.builder import core


def get_eigsys_hermitian(h_mat: np.ndarray):
    """Get eigenvalues and eigenstates for Hermitian matrix."""
    eigenvalues, eigenstates, info = spla.zheev(h_mat)
    idx = eigenvalues.argsort()[::1]
    return eigenvalues[idx], eigenstates[:, idx]


def get_eigsys_generic(g_mat: np.ndarray):
    """Get eigenvalues and eigenstates for generic matrix."""
    eigenvalues, l_eigenstates, r_eigenstates, info = spla.zgeev(g_mat)
    idx = eigenvalues.argsort()[::1]
    return eigenvalues[idx], l_eigenstates[:, idx], r_eigenstates[:, idx]


def get_phase(z: np.ndarray):
    """Calculates the phase of a complex array."""
    phase = np.angle(z)
    norm = np.absolute(z)
    for i in range(phase.shape[0]):
        if norm.item(i) < 1e-14:
            phase[i] = 0.0
    return phase


def set_ham(prim_cell: tb.PrimitiveCell, ham: np.ndarray, kpt: np.ndarray):
    """Set up Hamiltonian for given k-point."""
    ham *= 0.0
    core.set_ham(prim_cell.orb_pos, prim_cell.orb_eng,
                 prim_cell.hop_ind, prim_cell.hop_eng,
                 kpt, ham)


def get_z2(prim_cell: tb.PrimitiveCell, num_occ: int, mom_res: int):
    """
    Evaluate the phase factor in ref arxiv: 1101.2011v1.

    :param prim_cell: primitive cell to investigate
    :param num_occ: number of occupied orbitals
    :param mom_res: resolution of momenta grid
    :return phase: phase factor
    :rtype: np.ndarray
    """
    # Set up and plot momenta grid
    momenta = np.array([[(0., 0., 0.) for _ in range(mom_res)]
                        for _ in range(mom_res)])
    for i in range(mom_res):
        for j in range(mom_res):
            x_frac = 1. * i / mom_res
            y_frac = 0.5 * j / mom_res
            momenta[i, j] = (x_frac, y_frac, 0)

    # Declare arrays
    ham = np.zeros((prim_cell.num_orb, prim_cell.num_orb), dtype=complex)
    f_mat = np.zeros((num_occ, num_occ, mom_res), dtype=complex)
    theta = np.zeros((num_occ, mom_res))

    # Iterate over y-momentum
    for j in range(mom_res):
        momentum_new = momenta[-1, j]
        set_ham(prim_cell, ham, momentum_new)
        eigenvalues_new, eigenstates_new = get_eigsys_hermitian(ham)

        # Iterate over x-momentum
        for i in range(mom_res):
            eigenstates_old = eigenstates_new[:, :]
            momentum_new = momenta[i, j]

            # Get eigenvalues and eigenstates.
            set_ham(prim_cell, ham, momentum_new)
            eigenvalues_new, eigenstates_new = get_eigsys_hermitian(ham)

            # Calculate F
            # eigenstates[:, 0] is first eigenstate
            # iterate up to eigenstate N, which is Fermi level
            for m in range(num_occ):
                for n in range(num_occ):
                    f_mat[m, n, i] = np.vdot(eigenstates_old[:, m],
                                             eigenstates_new[:, n])

        # Calculate d_mat
        d_mat = np.matrix(f_mat[:, :, 1])
        for i in range(2, mom_res):
            d_mat = d_mat * np.matrix(f_mat[:, :, i])
        d_mat = d_mat * np.matrix(f_mat[:, :, 0])

        # Get eigenvalues and eigenstates of D and theta
        eigenvalues = get_eigsys_generic(d_mat)[0]
        eigenphases = get_phase(eigenvalues)
        idx = eigenphases.argsort()[::1]
        theta[:, j] = eigenphases[idx]
    return theta


def make_hop_dict(hop, lamb_so, lamb_r):
    def _hop_1st(vec):
        a = vec[1] + 1j * vec[0]
        ac = vec[1] - 1j * vec[0]
        return np.array([[hop, 1j * a * lamb_r], [1j * ac * lamb_r, hop]])

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
    hop = -1.
    lamb_so = 0.06 * hop
    lamb_r = 0.05 * hop
    mom_res = 200
    theta_file_name = "test_graphene"

    # QSH phase
    # lamb_nu = 0.1 * hop

    # normal insulating phase
    lamb_nu = 0.4 * hop

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
    hop_dict = make_hop_dict(hop, lamb_so, lamb_r)
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

    # Get theta
    theta = get_z2(prim_cell, num_occ=2, mom_res=mom_res)
    for i in range(2):
        plt.scatter(0.5 / mom_res * np.arange(len(theta[i])),
                    0.5 / pi * theta[i], s=3, lw=0)
    plt.ylim((-0.5, 0.5))
    plt.xlim((0, 0.5))
    plt.ylabel("angle")
    plt.xlabel("ky")
    plt.savefig(theta_file_name+".png")


if __name__ == '__main__':
    main()
