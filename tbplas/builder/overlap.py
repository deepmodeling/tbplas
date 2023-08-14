"""
Functions and classes for dealing with orbital overlap in real space.
"""

from typing import Tuple
import math

import numpy as np
from scipy.sparse import dia_matrix, dok_matrix, csr_matrix

from . import exceptions as exc
from .base import check_rn, IntraHopping, rn_type, rn3_type
from .primitive import PrimitiveCell
from ..cython import primitive as core


__all__ = ["Overlap"]


class Overlap:
    """
    Container for orbital overlap in real space.

    Attributes
    ----------
    _model: 'PrimitiveCell' instance
        primitive cell for which the orbital overlaps exist
    _onsite: List[int]
        on-site overlap, i.e. <i,0|i,0>
    _offsite: 'IntraHopping' instance
        off-site overlap <i,0|j,R>
    _hash_dict: Dict[str, int]
        hashes of attributes for updating arrays
    _ons_val: float64 array with length of num_orb
        array for on-site overlap
    _off_ind: (num_offsite, 5) int32 array
        indices of off-site overlaps
    _off_val: complex128 array with length of num_off_site
        values of off-site overlaps
    """
    def __init__(self, model: PrimitiveCell) -> None:
        """
        :param model: primitive cell for which the orbital overlaps exist
        """
        self._model = model
        self._onsite = [1.0 for _ in range(self._model.num_orb)]
        self._offsite = IntraHopping()
        self._hash_dict = {'ons': self._get_attr_hash('ons'),
                           'off': self._get_attr_hash('off')}
        self._ons_val = np.array(self._onsite, dtype=np.float64)
        self._off_ind = np.array([], dtype=np.int32)
        self._off_val = np.array([], dtype=np.complex128)

    def _get_attr_hash(self, attr: str) -> int:
        """
        Get the hash of given attribute.

        :param attr: name of the attribute
        :return: hash of the attribute
        :raises ValueError: if attr is illegal
        """
        if attr == "ons":
            new_hash = hash(tuple(self._onsite))
        elif attr == "off":
            new_hash = hash(self._offsite)
        else:
            raise ValueError(f"Illegal attribute name {attr}")
        return new_hash

    def _update_attr_hash(self, attr: str) -> bool:
        """
        Compare and update the hash of given attribute.

        :param attr: name of the attribute
        :return: whether the hash has been updated.
        :raises ValueError: if attr is illegal
        """
        new_hash = self._get_attr_hash(attr)
        if self._hash_dict[attr] != new_hash:
            self._hash_dict[attr] = new_hash
            status = True
        else:
            status = False
        return status

    def _check_hop_index(self, rn: rn_type,
                         orb_i: int,
                         orb_j: int) -> Tuple[rn3_type, int, int]:
        """
        Check cell index and orbital pair of hopping term.

        :param rn: cell index of the hopping term, i.e. R
        :param orb_i: index of orbital i in <i,0|H|j,R>
        :param orb_j: index of orbital j in <i,0|H|j,R>
        :return: checked cell index and orbital pair
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        rn, legal = check_rn(rn)
        if not legal:
            raise exc.CellIndexLenError(rn)
        num_orbitals = self._model.num_orb
        if not (0 <= orb_i < num_orbitals):
            raise exc.PCOrbIndexError(orb_i)
        if not (0 <= orb_j < num_orbitals):
            raise exc.PCOrbIndexError(orb_j)
        if rn == (0, 0, 0) and orb_i == orb_j:
            raise exc.PCHopDiagonalError(rn, orb_i)
        return rn, orb_i, orb_j

    def set_onsite(self, orb_i: int, overlap: float) -> None:
        """
        Set the on-site overlap term.

        :param orb_i: index of orbital
        :param overlap: the overlap
        :return: None
        :raises PCOrbIndexError: if orb_i falls out of range
        """
        try:
            self._onsite[orb_i] = overlap
        except IndexError as err:
            raise exc.PCOrbIndexError(orb_i) from err

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
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        rn, orb_i, orb_j = self._check_hop_index(rn, orb_i, orb_j)
        self._offsite.add_hopping(rn, orb_i, orb_j, overlap)

    def sync_array(self, force_sync: bool = False) -> None:
        """
        Synchronize the arrays.

        :param force_sync: whether to force synchronizing
        :return: None
        """
        to_update = self._update_attr_hash('ons')
        if force_sync or to_update:
            self._ons_val = np.array(self._onsite, dtype=np.float64)
        to_update = self._update_attr_hash('off')
        if force_sync or to_update:
            off_ind, off_val = self._offsite.to_array(use_int64=False)
            self._off_ind = off_ind
            self._off_val = off_val

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
        """
        if convention not in (1, 2):
            raise ValueError(f"Illegal convention {convention}")
        core.set_ham(self._model.orb_pos, self._ons_val,
                     self._off_ind, self._off_val,
                     convention, k_point, overlap_dense)

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
        """
        if convention not in (1, 2):
            raise ValueError(f"Illegal convention {convention}")

        # Diagonal terms
        num_orb = self._model.num_orb
        overlap_shape = (num_orb, num_orb)
        overlap_dia = dia_matrix((self._ons_val, 0), shape=overlap_shape)

        # Off-diagonal terms
        overlap_half = dok_matrix(overlap_shape, dtype=np.complex128)
        orb_pos = self._model.orb_pos
        for rn, hop_rn in self._offsite.hoppings.items():
            for pair, energy in hop_rn.items():
                ii, jj = pair
                if convention == 1:
                    dr = orb_pos[jj] - orb_pos[ii] + rn
                else:
                    dr = rn
                phase = 2 * math.pi * np.dot(k_point, dr).item()
                factor = math.cos(phase) + 1j * math.sin(phase)
                overlap_half[ii, jj] += factor * energy

        # Sum up the terms
        overlap_dok = overlap_dia + overlap_half + overlap_half.getH()
        overlap_csr = overlap_dok.tocsr()
        return overlap_csr
