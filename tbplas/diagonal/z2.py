"""Module for evaluating Z2 topological invariant."""

from typing import Tuple, Union, Any
from math import pi

import numpy as np
import scipy.linalg.lapack as spla

from .base import DiagSolver


__all__ = ["Z2"]


class Z2(DiagSolver):
    """
    Class for evaluating the Z2 topological invariant.

    Reference: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.075119


    Attributes
    ----------
    num_occ: int
        number of occupied bands of the primitive cell
    h_mat: (num_orb, num_orb) complex128 array
        Hamiltonian matrix for single k-point
    d_mat: (num_occ, num_occ) complex128 array
        D matrix for single k-point in the reference
    f_mat: (num_occ, num_occ) complex128 array
        F matrix for (ka_i, ka_i+1)
    """
    def __init__(self, cell: Any,
                 num_occ: int,
                 enable_mpi=False) -> None:
        """
        :param cell: primitive cell under investigation
        :param num_occ: number of occupied bands of the primitive cell
        :param enable_mpi: whether to enable parallelization over k-points
        :raises ValueError: if num_occ is larger than num_orb of the
            primitive cell
        """
        # Initialize parallel environment
        super().__init__(cell, enable_mpi=enable_mpi)

        # Check and set num_occ
        num_orb = self.num_orb
        if num_occ not in range(1, num_orb+1):
            raise ValueError(f"num_occ {num_occ} should be in [1, {num_orb}]")
        # if num_occ % 2 != 0:
        #     raise ValueError(f"num_occ {num_occ} is not a even number")
        self.num_occ = num_occ

        # Initialize working arrays
        self.h_mat = np.zeros((num_orb, num_orb), dtype=np.complex128)
        self.d_mat = np.zeros((num_occ, num_occ), dtype=np.complex128)
        self.f_mat = np.zeros((num_occ, num_occ), dtype=np.complex128)

    def _get_h_eigenstates(self, kpt: np.ndarray) -> np.ndarray:
        """
        Get the eigenstates of Hamiltonian for given k-point.

        :param kpt: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :return: (num_orb, num_orb) complex128 array
            eigenstates of the given k-point
        """
        self.h_mat *= 0.0
        self.model.set_ham_dense(kpt, self.h_mat)
        eigenvalues, eigenstates, info = spla.zheev(self.h_mat)
        idx = eigenvalues.argsort()
        return eigenstates[:, idx]

    def _get_d_eigenvalues(self) -> np.ndarray:
        """
        Get the eigenvalues of D matrix.

        :return: (num_occ,) complex128 array
            eigenvalues of the D matrix
        """
        eigenvalues, l_eigenstates, r_eigenstates, info = spla.zgeev(self.d_mat)
        idx = eigenvalues.argsort()
        return eigenvalues[idx]

    def _eval_f_mat(self, vec0: np.ndarray, vec1: np.ndarray) -> None:
        """
        Evaluate the F matrix from eigenvectors of Hamiltonian at ka_i and
        ka_i+1.

        :param vec0: (num_orb, num_orb) complex128 array
            eigenvectors at ka_i
        :param vec1: (num_orb, num_orb) complex128 array
            eigenvectors at kb_i+1
        :return: None.
        """
        self.f_mat *= 0.0
        for i in range(self.num_occ):
            for j in range(self.num_occ):
                self.f_mat[i, j] = np.vdot(vec0[:, i], vec1[:, j])

    @staticmethod
    def _eval_phase(z: np.ndarray) -> np.ndarray:
        """
        Evaluate the phases of all the elements in an array.

        :param z: complex128 array for which the phases will be evaluated
        :return: phases of all the elements in the array
        """
        phase = np.angle(z)
        norm = np.absolute(z)
        for i in range(phase.shape[0]):
            if norm.item(i) < 1e-14:
                phase[i] = 0.0
        return phase

    @staticmethod
    def _get_min_key(data_dict: dict) -> Union[int, tuple, None]:
        """
        Return the key with the minimum value.

        :param data_dict: dictionary to search
            with keys being integers or tuples and values being floats
        :return: key with the minimum value
        """
        min_value = min(data_dict.values())
        result = None
        for key, value in data_dict.items():
            if abs(value - min_value) < 1.0e-15:
                result = key
                break
        return result

    def calc_phases(self, ka_array: np.ndarray = None,
                    kb_array: np.ndarray = None,
                    kc: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the phases of WF centers as function of kb (eqn. 12 of ref).

        :param ka_array: (num_ka,) float64 array
            FRACTIONAL coordinates of the loop along a-axis
        :param kb_array: (num_kb,) float64 array
            FRACTIONAL coordinates of the loop along b-axis
        :param kc: float
            FRACTIONAL coordinate of the loop along c-axis
        :return: (kb_array, phases)
            kb_array: (num_kb,) float64 array
            FRACTIONAL coordinates of the loop along b-axis
            phases: (num_kpt, num_occ) float64 array
            phases of WF centers
        """
        # Allocate working arrays
        if ka_array is None:
            ka_array = np.linspace(-0.5, 0.5, 200)
        if kb_array is None:
            kb_array = np.linspace(0.0, 0.5, 100)
        num_ka = ka_array.shape[0]
        num_kb = kb_array.shape[0]
        phases = np.zeros((num_kb, self.num_occ), dtype=np.float64)

        # Distribute kb_array among processes
        k_index = self.dist_range(num_kb)

        # Loop over b-axis to fill the phase
        for ib in k_index:
            kb = kb_array.item(ib)
            # As the D matrix is evaluated through iterative matrix
            # multiplication, it should be initialized as an eye matrix.
            self.d_mat = np.eye(self.num_occ, dtype=np.complex128)

            # Get and backup the eigenvectors of Hamiltonian at ka_0
            kpt0 = np.array([ka_array.item(0), kb, kc])
            vec0 = self._get_h_eigenstates(kpt0)
            vec0_bak = vec0

            # Loop over a-axis to build the D matrix
            for ia in range(1, num_ka):
                kpt1 = np.array([ka_array.item(ia), kb, kc])
                vec1 = self._get_h_eigenstates(kpt1)
                self._eval_f_mat(vec0, vec1)
                self.d_mat = np.matmul(self.d_mat, self.f_mat)
                vec0 = vec1

            # Special treatment for the last ka
            self._eval_f_mat(vec0, vec0_bak)
            self.d_mat = np.matmul(self.d_mat, self.f_mat)

            # Diagonalize the D matrix and evaluate phases
            eigenvalues = self._get_d_eigenvalues()
            phase_ib = self._eval_phase(eigenvalues)
            idx = phase_ib.argsort()
            phases[ib] = phase_ib[idx]

        # Collect data
        phases = self.all_reduce(phases)
        return kb_array, phases

    def reorder_phases(self, phases: np.ndarray,
                       threshold: float = 0.1,
                       smooth: bool = True) -> np.ndarray:
        """
        Reorder the phases to improve continuity and smoothness.

        NOTE: sometimes this method may not work properly. In that case,
        try to increase threshold and the density of kb_array.

        :param phases: (num_kpt, num_occ) float64 array
            phases of WF centers
        :param threshold: threshold for detecting discontinuity
        :param smooth: whether to smooth the phases by ensuring the continuity
            of 1st order derivative with respect to kb
        :return: (num_kpt, num_occ) float64 array
            reordered phases
        """
        phases = phases.copy()

        # Fix phase discontinuity
        for ib in range(phases.shape[0]-1):
            phase_i0 = phases[ib]
            phase_i1 = phases[ib + 1]
            phase_i1_cp = phase_i1.copy()
            for j in range(self.num_occ):
                phase_j0 = phase_i0.item(j)
                phase_j1 = phase_i1.item(j)

                # Correction should be done for data points will large
                # difference with respect to previous point
                if abs(phase_j0 - phase_j1) > threshold:
                    # Find candidates at i+1 that conserve phase continuity
                    # considering periodicity
                    # Key: (m,n) of |phase_m1 - phase_j1 - n * pi| < threshold
                    # value: |phase_m1 - phase_j1 - n * pi|
                    diff_dict = dict()
                    for m, phase_m1 in enumerate(phase_i1_cp):
                        for n in (-2, 0, 2):
                            phase_diff = abs(phase_j0 - (phase_m1 + n * pi))
                            if phase_diff < threshold:
                                diff_dict[(m, n)] = phase_diff

                    if len(diff_dict) > 0:
                        min_key = self._get_min_key(diff_dict)
                        min_pos = min_key[0]
                        min_shift = min_key[1] * pi
                        phases[ib + 1, j] = phase_i1_cp[min_pos] + min_shift
                        phase_i1_cp = np.delete(phase_i1_cp, min_pos)

        # Fix derivative discontinuity (smoothness)
        if smooth:
            for ib in range(phases.shape[0]-2):
                phase_i0 = phases[ib]
                phase_i1 = phases[ib + 1]
                phase_i2_cp = phases[ib + 2].copy()
                for j in range(self.num_occ):
                    phase_j0 = phase_i0.item(j)
                    phase_j1 = phase_i1.item(j)

                    # Find candidates at i+2 that conserve phase continuity
                    # phase_m2 := phase_i2[m]
                    # Key: index(m) of |phase_m2 - phase_j1| < threshold
                    # Value: phase_m2
                    diff_dict = dict()
                    for m, phase_m2 in enumerate(phase_i2_cp):
                        if abs(phase_j1 - phase_m2) < threshold:
                            diff_dict[m] = phase_m2

                    # Find the point that ensures the best smoothness
                    # Key: index(m) of |phase_m2 - phase_j1| < threshold
                    # Value: |(phase_m2 - phase_j1) - (phase_j1 - phase_j0)|
                    if len(diff_dict) > 0:
                        for m, phase_m2 in diff_dict.items():
                            phase_diff = abs(phase_m2 - 2 * phase_j1 + phase_j0)
                            diff_dict[m] = phase_diff
                        min_pos = self._get_min_key(diff_dict)

                        # Update phases at ib+2
                        phases[ib + 2, j] = phase_i2_cp[min_pos]
                        phase_i2_cp = np.delete(phase_i2_cp, min_pos)
        return phases

    @staticmethod
    def count_crossing(phases: np.ndarray, phase_ref: float = 0.5) -> int:
        """
        Count the number that the phases go across a reference value.

        The Z2 topological invariant is determined as num_crossing % 2.

        NOTE: the phases must be CORRECTLY reordered before passing to this
        function. Otherwise, the crossing number will be wrong!

        If the reordering algorithm fails to work by all means, then you have to
        plot the phases using scatter plot and count the crossing number manually.

        :param phases: (num_kpt, num_occ) float64 array
            phases of WF centers
        :param phase_ref: reference value
        :return: number of crossing
        """
        num_kpt = phases.shape[0]
        num_occ = phases.shape[1]
        num_crossing = 0
        for ik in range(0, num_kpt - 1):
            for ib in range(num_occ):
                d1 = phases.item(ik, ib) - phase_ref
                d2 = phases.item(ik+1, ib) - phase_ref
                if d1 * d2 < 0.0:
                    num_crossing += 1
        return num_crossing
