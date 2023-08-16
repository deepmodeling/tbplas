"""
Functions and classes for dealing with orbital overlap in real space.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .base import Orbital, rn_type
from .primitive import PrimitiveCell
from ..base import NM


__all__ = ["Overlap"]


class Overlap(PrimitiveCell):
    """
    Container for orbital overlap in real space.

    This class has much in common with the PrimitiveCell class. The only
    difference is that we have orbital energy as the on-site overlap term
    and hopping energy as the off-site overlap term.
    """
    def __init__(self, model: PrimitiveCell) -> None:
        """
        :param model: primitive cell for which the orbital overlaps exist
        """
        super().__init__(lat_vec=model.lat_vec, origin=model.origin, unit=NM)
        for orb in model.orbitals:
            self._orbital_list.append(Orbital(orb.position, 1.0, orb.label))

    def set_onsite(self, orb_i: int, overlap: float = 1.0) -> None:
        """
        Set the on-site overlap term.

        :param orb_i: index of orbital
        :param overlap: the overlap
        :return: None
        :raises LockError: if the overlap is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        """
        self.set_orbital(orb_i, energy=overlap)

    def add_offsite(self, rn: rn_type,
                    orb_i: int,
                    orb_j: int,
                    overlap: complex) -> None:
        """
        Add a new off-site overlap term or update an existing one.

        :param rn: cell index of the overlap term, i.e. R
        :param orb_i: index of orbital i in <i,0|j,R>
        :param orb_j: index of orbital j in <i,0|j,R>
        :param overlap: the overlap
        :return: None
        :raises LockError: if the overlap is locked
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        self.add_hopping(rn, orb_i, orb_j, energy=overlap)

    def set_overlap_dense(self, k_point: np.ndarray,
                          overlap_dense: np.ndarray,
                          convention: int = 1) -> None:
        """
        Set up dense overlap matrix for given k-point.

        This is the interface to be called by external exact solvers. The
        callers are responsible to call the 'sync_array' method and initialize
        overlap_dense as a zero matrix.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param overlap_dense: (num_orb, num_orb) complex128 array
            incoming overlap matrix
        :param convention: convention for setting up the overlap matrix
        :return: None
        :raises ValueError: if convention not in (1, 2)
        :raises PCHopEmptyError: if overlap does not contain off-site terms
        """
        self.set_ham_dense(k_point, overlap_dense, convention)

    def set_overlap_csr(self, k_point: np.ndarray,
                        convention: int = 1) -> csr_matrix:
        """
        Set up sparse overlap in csr format for given k-point.

        This is the interface to be called by external exact solvers. The
        callers are responsible to call the 'sync_array' method.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param convention: convention for setting up the overlap matrix
        :return: sparse overlap matrix
        :raises ValueError: if convention not in (1, 2)
        :raises PCHopEmptyError: if overlap does not contain off-site terms
        """
        return self.set_ham_csr(k_point, convention)
