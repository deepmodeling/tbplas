"""Module for calculating magnetic properties."""

from typing import Tuple

import numpy as np

import scipy.linalg.lapack as spla

from .builder import PrimitiveCell, frac2cart
from .builder import core


__all__ = ["SpinTexture"]


class SpinTexture:
    """
    Spin texture calculator.

    Attributes
    ----------
    _cell: 'PrimitiveCell' instance
        primitive cell for which properties will be evaluated
    _k_grid: (num_kpt, 3) float64 array
        FRACTIONAL coordinates of k-points
    _spin_major: bool
        whether the orbitals are stored in spin-major order
    _states: (num_kpt, num_orb) complex128 array
        cache of wave functions
    """
    def __init__(self, cell: PrimitiveCell,
                 k_grid: np.ndarray,
                 spin_major: bool = True) -> None:
        """
        :param cell: primitive cell for which properties will be evaluated
        :param k_grid: FRACTIONAL coordinates of k-points
        :param spin_major: whether the orbitals are stored in spin-major order
        """
        self._cell = cell
        self._k_grid = k_grid
        self._spin_major = spin_major
        self._states = None

    def calc_states(self) -> None:
        """
        Calculate wave functions on self._k_grid.

        :return: None
        """
        self._cell.sync_array()
        num_kpt = self._k_grid.shape[0]
        num_orb = self._cell.num_orb
        states = np.zeros((num_kpt, num_orb, num_orb), dtype=np.complex128)
        ham_k = np.zeros((num_orb, num_orb), dtype=np.complex128)
        for i_k, kpt in enumerate(self._k_grid):
            ham_k *= 0.0
            core.set_ham(self._cell.orb_pos, self._cell.orb_eng,
                         self._cell.hop_ind, self._cell.hop_eng,
                         kpt, ham_k)
            eigenvalues, eigenstates, info = spla.zheev(ham_k)
            states[i_k] = eigenstates.T
        self._states = states

    def split_spin(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split spin-up and spin-down components for wave function at given
        k-point and band.

        Two kinds of orbital orders are implemented:
        spin major: psi_{0+}, psi_{1+}, psi_{0-}, psi_{1-}
        orbital major: psi_{0+}, psi_{0-}, psi_{1+}, psi_{1-}

        If the orbitals are sorted in neither spin nor orbital major,
        derive a new class from SpinTexture and overwrite this method.

        :param state: (num_orb,) complex128 array
            wave function at given k-point and band
        :return: (u, d)
            u: (num_orb//2,) complex128 array
            d: (num_orb//2,) complex128 array
            spin-up and spin-down components of the wave function
        """
        num_orb = self._cell.num_orb // 2
        if self._spin_major:
            u, d = state[:num_orb], state[num_orb:]
        else:
            u, d = state[0::2], state[1::2]
        return u, d

    def eval(self, component: str = "z") -> np.ndarray:
        """
        Calculate the expectation of Pauli matrix.

        :param component: which Pauli matrix to evaluate
        :return: (num_kpt, num_orb) float64 array
            expectation of Pauli matrix
        """
        # Get eigenstates
        if self._states is None:
            self.calc_states()

        # Evaluate spin texture
        num_kpt = self._k_grid.shape[0]
        num_orb = self._cell.num_orb
        expectation = np.zeros((num_kpt, num_orb), dtype=np.float64)
        for i_k in range(num_kpt):
            for i_b in range(num_orb):
                state = self._states[i_k, i_b]
                u, d = self.split_spin(state)
                if component == "x":
                    expect = np.vdot(u, d) + np.vdot(d, u)
                elif component == "y":
                    expect = -1j * np.vdot(u, d) + 1j * np.vdot(d, u)
                else:
                    expect = np.vdot(u, u) - np.vdot(d, d)
                expectation[i_k, i_b] = expect.real
        return expectation

    @property
    def k_grid(self) -> np.ndarray:
        """Get FRACTIONAL coordinates of k-grid."""
        return self._k_grid

    @k_grid.setter
    def k_grid(self, k_grid: np.ndarray) -> None:
        """Set FRACTIONAL coordinates of k-grid."""
        self._k_grid = k_grid

    @property
    def k_cart(self) -> np.ndarray:
        """Get CARTESIAN coordinates of k-grid."""
        return frac2cart(self._cell.get_reciprocal_vectors(), self._k_grid)
