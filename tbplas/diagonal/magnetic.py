"""Module for calculating magnetic properties."""

from typing import Tuple, Any

import numpy as np

from ..base import frac2cart
from .base import DiagSolver


__all__ = ["SpinTexture"]


class SpinTexture(DiagSolver):
    """
    Spin texture calculator.

    Attributes
    ----------
    k_grid: (num_kpt, 3) float64 array
        FRACTIONAL coordinates of k-points
    spin_major: bool
        whether the orbitals are stored in spin-major order
    states: (num_kpt, num_orb) complex128 array
        cache of wave functions
    """
    def __init__(self, cell: Any,
                 k_grid: np.ndarray,
                 spin_major: bool = True,
                 **kwargs) -> None:
        """
        :param cell: primitive cell for which properties will be evaluated
        :param k_grid: FRACTIONAL coordinates of k-points
        :param spin_major: whether the orbitals are stored in spin-major order
        :param kwargs: parallelization arguments for DiagSolver.__init__
        """
        super().__init__(cell, **kwargs)
        self.k_grid = k_grid
        self.spin_major = spin_major
        self.states = None

    def update_states(self) -> None:
        """
        Update wave functions on self._k_grid.

        :return: None
        """
        self.states = self.calc_states(self.k_grid, all_reduce=False)[1]

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
        num_orb = self.num_orb // 2
        if self.spin_major:
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
        if self.states is None:
            self.update_states()

        # Evaluate spin texture
        num_kpt = self.k_grid.shape[0]
        num_orb = self.num_orb
        expectation = np.zeros((num_kpt, num_orb), dtype=np.float64)
        k_index = self.dist_range(num_kpt)
        for i_k in k_index:
            for i_b in range(num_orb):
                state = self.states[i_k, i_b]
                u, d = self.split_spin(state)
                if component == "x":
                    expect = np.vdot(u, d) + np.vdot(d, u)
                elif component == "y":
                    expect = -1j * np.vdot(u, d) + 1j * np.vdot(d, u)
                else:
                    expect = np.vdot(u, u) - np.vdot(d, d)
                expectation[i_k, i_b] = expect.real

        # Collect data
        expectation = self.all_reduce(expectation)
        return expectation

    @property
    def k_cart(self) -> np.ndarray:
        """
        Get CARTESIAN coordinates of k-grid.

        :return: (3, 3) float64 array
            Cartesian coordinates of k-grid in 1/nm.
        """
        return frac2cart(self.recip_lat_vec, self.k_grid)
