"""
Base functions and classes used through the builder package

Functions
---------
    check_coord: developer function
        check and auto-complete cell index, orbital coordinate,
        super-cell dimension and periodic condition
    invert_rn: developer function
        check if the cell index should be inverted

Classes
-------
    Orbital: developer class
        abstraction for orbitals in TB model
    Lockable: developer class
        base class for all lockable classes
    Hopping: developer class
        base class for hopping term container
    IntraHopping: developer class
        for holding hopping terms in a primitive cell or modifications
        to a supercell
    InterHopping: developer class
        base class for container classes holding hopping terms between
        two models
    HopDict: user class
        container for holding hopping terms of a primitive cell
        reserved for compatibility with old version of TBPlaS
"""

from collections import namedtuple
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from . import exceptions as exc


def check_coord(coord: tuple, complete_item=0):
    """
    Check and auto-complete cell index, orbital coordinate, etc.

    :param coord: tuple with 2 integers or floats
        incoming coordinate to check
    :param complete_item: integer or float
        item to be appended to coord if its length is 2
    :return: coord: tuple with 3 integers or floats
        corrected and auto-completed coordinate
    :return: legal: boolean
        True if length of incoming coord is 2 or 3, otherwise False.
    """
    coord, legal = tuple(coord), True
    len_coord = len(coord)
    if len_coord == 3:
        pass
    elif len_coord == 2:
        if isinstance(coord[0], int):
            coord += (int(complete_item),)
        else:
            coord += (float(complete_item),)
    else:
        legal = False
    return coord, legal


def invert_rn(rn: tuple, i=0):
    """
    Check if the cell index should be inverted.

    :param tuple rn: (r_a, r_b, r_c), cell index
    :param int i: component index
    :return: bool, True if to invert
    """
    if rn[i] > 0:
        return False
    elif rn[i] < 0:
        return True
    else:
        if i < 2:
            return invert_rn(rn, i+1)
        else:
            return False


Orbital = namedtuple("Orbital", ["position", "energy", "label"])


class Lockable(ABC):
    """
    Base class for all lockable objects.

    Attributes
    ----------
    is_locked: boolean
        whether the object is locked
    """
    def __init__(self) -> None:
        self.is_locked = False

    def lock(self):
        """
        Lock the object. Modifications are not allowed then unless
        the 'unlock' method is called.

        :return: None
        """
        self.is_locked = True

    def unlock(self):
        """
        Unlock the object. Modifications are allowed then.

        :return: None
        """
        self.is_locked = False

    @abstractmethod
    def check_lock(self):
        """Check the lock state of the object."""
        pass


class Hopping(ABC):
    """
    Base class for IntraHopping and InterHopping classes.

    Attributes
    ----------
    dict: dictionary containing the hopping terms
        Keys are cell indices (rn), while values are also dictionaries.
        Keys of value dictionary are orbital pairs, while values are hopping
        energies.
    """
    def __init__(self):
        super().__init__()
        self.dict = {}

    def __hash__(self):
        """Return hash value of this instance."""
        hop_list = self.to_list()
        return hash(tuple(hop_list))

    @staticmethod
    @abstractmethod
    def _norm_keys(rn: tuple, orb_i: int, orb_j: int):
        """
        Normalize cell index and orbital pair into permitted keys of self.dict.

        NOTE: This function should be overridden in derived classes.

        For IntraHopping, it should check whether to take the conjugation and
        return the status in conj. For InterHopping, it should check if rn is
        legal since the class is exposed to the user. The status conj should
        always be False for InterHopping.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :param int orb_i: orbital index or bra
        :param int orb_j: orbital index of ket
        :return: (rn, pair, conj)
            where rn is the normalized cell index,
            pair is the normalized orbital pair,
            conj is the flag of whether to take the conjugate of hopping energy
        """
        pass

    def add_hopping(self, rn: tuple, orb_i: int, orb_j: int, energy: complex):
        """
        Add a new hopping term or update existing term.

        NOTE: For IntraHopping, conjugate terms are reduced by normalizing their
        cell indices and orbital pairs. For InterHopping this is not needed.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :param int orb_i: orbital index or bra
        :param int orb_j: orbital index of ket
        :param complex energy: hopping energy
        :return: None
        """
        rn, pair, conj = self._norm_keys(rn, orb_i, orb_j)
        if conj:
            energy = energy.conjugate()
        try:
            hop_rn = self.dict[rn]
        except KeyError:
            hop_rn = self.dict[rn] = dict()
        hop_rn[pair] = energy

    def get_hopping(self, rn: tuple, orb_i: int, orb_j: int):
        """
        Get an existing hopping term.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :param int orb_i: orbital index or bra
        :param int orb_j: orbital index of ket
        :return: (energy, status)
            where energy is the hopping energy and status is the flag whether
            the term has been found
        """
        rn, pair, conj = self._norm_keys(rn, orb_i, orb_j)
        try:
            energy = self.dict[rn][pair]
            status = True
        except KeyError:
            energy = None
            status = False
        if status and conj:
            energy = energy.conjugate()
        return energy, status

    def remove_hopping(self, rn: tuple, orb_i: int, orb_j: int, clean=False):
        """
        Remove an existing hopping term.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :param int orb_i: orbital index or bra
        :param int orb_j: orbital index of ket
        :param clean: boolean
            whether to call 'clean' to remove empty cell indices
        :return: status
            where the hopping term is removed, False if not found
        """
        rn, pair, conj = self._norm_keys(rn, orb_i, orb_j)
        try:
            self.dict[rn].pop(pair)
            status = True
        except KeyError:
            status = False
        if clean:
            self.clean()
        return status

    def get_rn(self):
        """
        Get the list of cell indices.

        :return: list of cell indices
        """
        return list(self.dict.keys())

    def remove_rn(self, rn: tuple):
        """
        Remove all the hopping terms of given cell index.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :return: status
            where the hopping terms are removed, False if not found
        """
        rn, pair, conj = self._norm_keys(rn, 0, 0)
        try:
            self.dict.pop(rn)
            status = True
        except KeyError:
            status = False
        return status

    def remove_orbital(self, orb_i: int, clean=False):
        """
        Wrapper over 'remove_orbitals' to remove a single orbital.

        :param int orb_i: orbital index to remove
        :param clean: boolean
            whether to call 'clean' to remove empty cell indices
        :return: None.
        :raises LockError: if the object is locked
        """
        self.remove_orbitals([orb_i], clean=clean)

    def remove_orbitals(self, indices: List[int], clean=True):
        """
        Remove the hopping terms corresponding to a list of orbitals and update
        remaining hopping terms.

        :param indices: List[int]
            indices of orbitals to remove
        :param clean: boolean
            whether to call 'clean' to remove empty cell indices
        :return: None
        """
        indices = sorted(indices)
        idx_remap = dict()

        def _remap(orb_i):
            try:
                result = idx_remap[orb_i]
            except KeyError:
                result = orb_i
                for j in indices:
                    if j < orb_i:
                        result -= 1
                idx_remap[orb_i] = result
            return result

        for rn, hop_rn in self.dict.items():
            new_hop_rn = dict()
            for pair in hop_rn.keys():
                ii, jj = pair
                if ii in indices or jj in indices:
                    pass
                else:
                    ii = _remap(ii)
                    jj = _remap(jj)
                    new_hop_rn[(ii, jj)] = hop_rn[pair]
            self.dict[rn] = new_hop_rn

        if clean:
            self.clean()

    def clean(self):
        """
        Remove empty cell indices.

        :return: None. self.dict is modified.
        """
        for rn in list(self.dict.keys()):
            if self.dict[rn] == {}:
                self.dict.pop(rn)

    def to_list(self):
        """
        Flatten all hopping terms into a list.

        :return: hopping terms as a list
        :rtype: list of (rb, rb, rc, orb_i, orb_j, energy)
        """
        self.clean()
        hop_list = []
        for rn, hop_rn in self.dict.items():
            for pair, energy in hop_rn.items():
                hop_list.append(rn + pair + (energy,))
        return hop_list

    def to_array(self, use_int64=False):
        """
        Convert hopping terms to array of 'hop_ind' and 'hop_eng',
        for constructing attributes of 'PrimitiveCell' or 'Sample'.

        :param use_int64: boolean
            whether to use 64-bit integer for hop_ind, should be
            enabled for the 'Sample' class
        :return: (hop_ind, hop_eng)
            hop_ind: (num_hop, 5) int32 array, hopping indices
            hop_eng: (num_hop,) complex128 array, hopping energies
        """
        self.clean()
        hop_ind, hop_eng = [], []
        for rn, hop_rn in self.dict.items():
            for pair, energy in hop_rn.items():
                hop_ind.append(rn + pair)
                hop_eng.append(energy)
        if use_int64:
            hop_ind = np.array(hop_ind, dtype=np.int64)
        else:
            hop_ind = np.array(hop_ind, dtype=np.int32)
        hop_eng = np.array(hop_eng, dtype=np.complex128)
        return hop_ind, hop_eng

    def count_pair(self, orb_i, orb_j):
        """
        Count the hopping terms with given orbital index.

        :param orb_i: int
            orbital index of bra
        :param orb_j: int
            orbital index of ket
        :return: count: int
            number of hopping terms with given orbital index
        """
        self.clean()
        count = 0
        pair = (orb_i, orb_j)
        for rn, hop_rn in self.dict.items():
            if pair in hop_rn.keys():
                count += 1
        return count

    @property
    def num_hop(self):
        """Count the number of hopping terms."""
        self.clean()
        num_hop = 0
        for rn, hop_rn in self.dict.items():
            num_hop += len(hop_rn)
        return num_hop


class IntraHopping(Hopping):
    """
    Container class for holding hopping terms of a primitive cell or
    modifications to a supercell.

    NOTE: this class is intended to be called by the 'PrimitiveCell' and
    'SuperCell' classes. It is assumed that the caller will take care of all the
    parameters passed to this class. NO CHECKING WILL BE PERFORMED HERE.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _norm_keys(rn: tuple, orb_i: int, orb_j: int):
        """
        Normalize cell index and orbital pair into permitted keys of self.dict.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :param int orb_i: orbital index or bra
        :param int orb_j: orbital index of ket
        :return: (rn, pair, conj)
            where rn is the normalized cell index,
            pair is the normalized orbital pair,
            conj is the flag of whether to take the conjugate of hopping energy
        """
        if invert_rn(rn):
            rn = (-rn[0], -rn[1], -rn[2])
            pair = (orb_j, orb_i)
            conj = True
        elif rn == (0, 0, 0) and orb_i > orb_j:
            rn = rn
            pair = (orb_j, orb_i)
            conj = True
        else:
            rn = rn
            pair = (orb_i, orb_j)
            conj = False
        return rn, pair, conj


class InterHopping(Hopping):
    """
    Base class for container classes holding hopping terms between two models.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _norm_keys(rn: tuple, orb_i: int, orb_j: int):
        """
        Normalize cell index and orbital pair into permitted keys of self.dict.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :param int orb_i: orbital index or bra
        :param int orb_j: orbital index of ket
        :return: (rn, pair, conj)
            where rn is the normalized cell index,
            pair is the normalized orbital pair,
            conj is the flag of whether to take the conjugate of hopping energy
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        rn, legal = check_coord(rn)
        if not legal:
            raise exc.CellIndexLenError(rn)
        pair = (orb_i, orb_j)
        conj = False
        return rn, pair, conj


class HopDict:
    """
    Class for holding hopping terms.

    Reserved for compatibility with old version of TBPlaS.

    NOTE: DO NOT try to rewrite this class based on IntraHopping. This class
    is intended for compatibility reasons and designed following a different
    philosophy than IntraHopping.

    Attributes
    ----------
    dict: dictionary
        Keys should be cell indices and values should be complex matrices.
    mat_shape: (num_orb, num_orb)
        shape of hopping matrices
    """
    def __init__(self, num_orb):
        """
        Initialize hop_dict object.

        :param num_orb: integer
            number of orbitals
        """
        self.dict = {}
        self.mat_shape = (num_orb, num_orb)

    @staticmethod
    def _check_rn(rn):
        """
        Check and complete cell index.

        :param rn: tuple of integers
            cell index to check
        :return rn: tuple of integers
            checked cell index
        :raises CellIndexLenError: if len(rn) is not 2 or 3
        """
        rn, legal = check_coord(rn)
        if not legal:
            raise exc.CellIndexLenError(rn)
        return rn

    def set_num_orb(self, num_orb):
        """
        Reset 'mat_shape' according to num_orb.

        :param num_orb: integer
            number of orbitals
        :return: None
        """
        self.mat_shape = (num_orb, num_orb)

    def set_mat(self, rn, hop_mat: np.ndarray):
        """
        Add hopping matrix to dictionary or update an existing hopping matrix.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :param hop_mat: (num_orb, num_orb) complex128 array
            hopping matrix
        :raises CellIndexLenError: if len(rn) != 2 or 3
        :raises ValueError: if hop_mat.shape does not match number of orbitals
        :raises PCHopDiagonalError: if on-site energies are included in hopping
            matrix
        """
        # Check cell index
        rn = self._check_rn(rn)

        # Check matrix size
        hop_mat = np.array(hop_mat)
        if hop_mat.shape != self.mat_shape:
            raise ValueError(f"Shape of hopping matrix {hop_mat.shape} does not "
                             f"match {self.mat_shape}")

        # Check for diagonal terms
        if rn == (0, 0, 0):
            for i in range(hop_mat.shape[0]):
                if abs(hop_mat.item(i, i)) >= 1e-5:
                    raise exc.PCHopDiagonalError(rn, i)

        # Set hopping matrix
        self.dict[rn] = hop_mat

    def set_zero_mat(self, rn):
        """
        Add zero hopping matrix to dictionary.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        zero_mat = np.zeros(self.mat_shape, dtype=complex)
        self.set_mat(rn, zero_mat)

    def set_element(self, rn, element, hop):
        """
        Add single hopping to hopping matrix.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :param element: (orb_i, orb_j)
            element indices
        :param hop: complex float
            hopping value
        :raises CellIndexLenError: if len(rn) != 2 or 3
        :raises ValueError: if element indices are out of range
        :raises PCHopDiagonalError: if on-site energy is given as input
        """
        # Check cell index
        rn = self._check_rn(rn)

        # Check element indices
        if element[0] not in range(self.mat_shape[0]) or \
                element[1] not in range(self.mat_shape[1]):
            raise ValueError(f"Element {element} out of range {self.mat_shape}")

        # Check for on-site energy
        if rn == (0, 0, 0) and element[0] == element[1]:
            raise exc.PCHopDiagonalError(rn, element[0])

        # Set matrix element
        try:
            hop_mat = self.dict[rn]
        except KeyError:
            hop_mat = self.dict[rn] = np.zeros(self.mat_shape, dtype=complex)
        hop_mat[element[0], element[1]] = hop

    def delete_mat(self, rn):
        """
        Delete hopping matrix from dictionary.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :returns: None
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        # Check cell index
        rn = self._check_rn(rn)

        # Delete hopping matrix
        self.dict.pop(rn, None)
