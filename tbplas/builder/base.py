"""Base functions and classes used through the builder package."""

from collections import namedtuple
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict

import numpy as np
from scipy.sparse import coo_matrix

from . import exceptions as exc


# Shortcuts for typing
rn2_type = Tuple[int, int]
rn3_type = Tuple[int, int, int]
rn_type = Union[rn2_type, rn3_type]
pos2_type = Tuple[float, float]
pos3_type = Tuple[float, float, float]
pos_type = Union[pos2_type, pos3_type]
pbc2_type = Tuple[bool, bool]
pbc3_type = Tuple[bool, bool, bool]
pbc_type = Union[pbc2_type, pbc3_type]
pair_type = Tuple[int, int]
id_pc_type = Tuple[int, int, int, int]


def check_rn(coord: rn_type,
             complete_item: int = 0) -> Tuple[rn3_type, bool]:
    """
    Check and auto-complete cell index.

    :param coord: incoming cell index
    :param complete_item: item to be appended to rn if its length is 2
    :return: (rn, legal)
        rn: corrected and auto-completed coordinate
        legal: True if length of incoming coord is 2 or 3, otherwise False.
    """
    coord, legal = tuple(coord), True
    len_coord = len(coord)
    if len_coord == 3:
        pass
    elif len_coord == 2:
        coord += (int(complete_item),)
    else:
        legal = False
    return coord, legal


def check_pos(coord: pos_type,
              complete_item: float = 0.0) -> Tuple[pos3_type, bool]:
    """Same as check_rn, but for atomic positions."""
    coord, legal = tuple(coord), True
    len_coord = len(coord)
    if len_coord == 3:
        pass
    elif len_coord == 2:
        coord += (float(complete_item),)
    else:
        legal = False
    return coord, legal


def check_pbc(coord: pbc_type,
              complete_item: bool = False) -> Tuple[pbc3_type, bool]:
    """Same as check_rn, but for periodic boundary conditions."""
    coord, legal = tuple(coord), True
    len_coord = len(coord)
    if len_coord == 3:
        pass
    elif len_coord == 2:
        coord += (bool(complete_item),)
    else:
        legal = False
    return coord, legal


def invert_rn(rn: Tuple[int, int, int], i: int = 0) -> bool:
    """
    Check if the cell index should be inverted.

    :param rn: (r_a, r_b, r_c), cell index
    :param i: component index
    :return: whether to invert the cell index
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
    __locked: bool
        whether the object is locked
    __locker_id: List[str]
        ids of lockers who locked this object
    """
    def __init__(self) -> None:
        self.__locked = False
        self.__locker_id = []

    def lock(self, locker_id: str) -> None:
        """
        Lock the object. Modifications are not allowed then unless the 'unlock'
        method is called.

        :param locker_id: id of the locker
        :return: None
        """
        self.__locked = True
        try:
            self.__locker_id.remove(locker_id)
        except ValueError:
            pass
        self.__locker_id.append(locker_id)

    def unlock(self) -> None:
        """
        Unlock the object. Modifications are allowed then.

        :return: None
        """
        self.__locked = False
        self.__locker_id = []

    def check_lock(self) -> None:
        """
        Check the lock state of the object.

        :return: None
        :raises LockError: if the object is locked
        """
        if self.__locked:
            print("This object has been locked by:")
            for locker in self.__locker_id:
                print(locker)
            raise exc.LockError()


class Hopping(ABC):
    """
    Base class for IntraHopping and InterHopping.

    Attributes
    ----------
    _hoppings: Dict[Tuple[int, int, int], Dict[Tuple[int, int], complex]]
        Keys are cell indices (rn), while values are dictionaries.
        Keys of value dictionary are orbital pairs, while values are hopping
        energies.
    """
    def __init__(self) -> None:
        super().__init__()
        self._hoppings = {}

    def __hash__(self) -> int:
        """Return the hash of this instance."""
        return hash(tuple(self.to_list()))

    @staticmethod
    @abstractmethod
    def _norm_keys(rn: rn_type,
                   orb_i: int,
                   orb_j: int) -> Tuple[rn3_type, pair_type, complex]:
        """
        Normalize cell index and orbital pair into permitted keys of self.dict.

        For IntraHopping, it should check whether to take the conjugation and
        return the status in conj. For InterHopping, it should check if rn is
        legal since the class is exposed to the user. The status conj should
        always be False for InterHopping.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
        :return: (rn, pair, conj)
            where rn is the normalized cell index,
            pair is the normalized orbital pair,
            conj is the flag of whether to take the conjugate of hopping energy
        """
        pass

    def add_hopping(self, rn: rn_type,
                    orb_i: int,
                    orb_j: int,
                    energy: complex) -> None:
        """
        Add a new hopping term or update existing term.

        NOTE: For IntraHopping, conjugate terms are reduced by normalizing their
        cell indices and orbital pairs. For InterHopping this is not needed.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
        :param energy: hopping energy
        :return: None
        """
        rn, pair, conj = self._norm_keys(rn, orb_i, orb_j)
        if conj:
            energy = energy.conjugate()
        try:
            hop_rn = self._hoppings[rn]
        except KeyError:
            hop_rn = self._hoppings[rn] = dict()
        hop_rn[pair] = energy

    def get_hopping(self, rn: rn_type,
                    orb_i: int,
                    orb_j: int) -> Tuple[complex, bool]:
        """
        Get an existing hopping term.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
        :return: (energy, status)
            where energy is the hopping energy and status is the flag whether
            the term has been found
        """
        rn, pair, conj = self._norm_keys(rn, orb_i, orb_j)
        try:
            energy = self._hoppings[rn][pair]
            status = True
        except KeyError:
            energy = None
            status = False
        if status and conj:
            energy = energy.conjugate()
        return energy, status

    def remove_hopping(self, rn: rn_type,
                       orb_i: int,
                       orb_j: int,
                       purge: bool = False) -> bool:
        """
        Remove an existing hopping term.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
        :param purge: whether to call 'purge' to remove empty cell indices
        :return: where the hopping term is removed, False if not found
        """
        rn, pair, conj = self._norm_keys(rn, orb_i, orb_j)
        try:
            self._hoppings[rn].pop(pair)
            status = True
        except KeyError:
            status = False
        if purge:
            self.purge()
        return status

    def remove_orbital(self, orb_i: int, purge: bool = False) -> None:
        """
        Wrapper over 'remove_orbitals' to remove a single orbital.

        :param orb_i: orbital index to remove
        :param purge: whether to call 'purge' to remove empty cell indices
        :return: None
        :raises LockError: if the object is locked
        """
        self.remove_orbitals([orb_i], purge=purge)

    def remove_orbitals(self, indices: Union[List[int], np.ndarray],
                        purge: bool = True) -> None:
        """
        Remove the hopping terms corresponding to a list of orbitals and update
        remaining hopping terms.

        :param indices: indices of orbitals to remove
        :param purge: whether to call 'clean' to remove empty cell indices
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

        for rn, hop_rn in self._hoppings.items():
            new_hop_rn = dict()
            for pair in hop_rn.keys():
                ii, jj = pair
                if ii in indices or jj in indices:
                    pass
                else:
                    ii = _remap(ii)
                    jj = _remap(jj)
                    new_hop_rn[(ii, jj)] = hop_rn[pair]
            self._hoppings[rn] = new_hop_rn

        if purge:
            self.purge()

    def remove_rn(self, rn: rn_type) -> bool:
        """
        Remove all the hopping terms of given cell index.

        :param rn: (r_a, r_b, r_c), cell index
        :return: where the hopping terms are removed, False if not found
        """
        rn, pair, conj = self._norm_keys(rn, 0, 0)
        try:
            self._hoppings.pop(rn)
            status = True
        except KeyError:
            status = False
        return status

    def purge(self) -> None:
        """
        Remove empty cell indices.

        :return: None
        """
        for rn in list(self._hoppings.keys()):
            if self._hoppings[rn] == {}:
                self._hoppings.pop(rn)

    def to_list(self) -> List[Tuple[int, int, int, int, int, complex]]:
        """
        Flatten all hopping terms into a list.

        :return: list of hopping terms (rb, rb, rc, orb_i, orb_j, energy)
        """
        self.purge()
        hop_list = [rn + pair + (energy,)
                    for rn, hop_rn in self._hoppings.items()
                    for pair, energy in hop_rn.items()]
        return hop_list

    def to_array(self, use_int64: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert hopping terms to array of 'hop_ind' and 'hop_eng',
        for constructing attributes of 'PrimitiveCell' or 'Sample'.

        :param use_int64: whether to use 64-bit integer for hop_ind, should be
            enabled for 'SuperCell' and 'SCInterHopping' classes
        :return: (hop_ind, hop_eng)
            hop_ind: (num_hop, 5) int32/int64 array, hopping indices
            hop_eng: (num_hop,) complex128 array, hopping energies
        """
        self.purge()
        hop_ind = [rn + pair
                   for rn, hop_rn in self._hoppings.items()
                   for pair, energy in hop_rn.items()]
        hop_eng = [energy
                   for rn, hop_rn in self._hoppings.items()
                   for pair, energy in hop_rn.items()]
        if use_int64:
            hop_ind = np.array(hop_ind, dtype=np.int64)
        else:
            hop_ind = np.array(hop_ind, dtype=np.int32)
        hop_eng = np.array(hop_eng, dtype=np.complex128)
        return hop_ind, hop_eng

    def count_pair(self, orb_i: int, orb_j: int) -> int:
        """
        Count the hopping terms with given orbital index.

        :param orb_i: orbital index of bra
        :param orb_j: orbital index of ket
        :return: number of hopping terms with given orbital index
        """
        self.purge()
        count = 0
        pair = (orb_i, orb_j)
        for rn, hop_rn in self._hoppings.items():
            if pair in hop_rn.keys():
                count += 1
        return count

    @property
    def hoppings(self) -> Dict[Tuple[int, int, int], Dict[Tuple[int, int], complex]]:
        """Interface for the '_hoppings' attribute."""
        return self._hoppings

    @property
    def cell_indices(self) -> List[rn3_type]:
        """
        Get the list of cell indices.

        :return: list of cell indices
        """
        return list(self._hoppings.keys())

    @property
    def num_hop(self) -> int:
        """
        Count the number of hopping terms.

        :return: number of hopping terms
        """
        self.purge()
        num_hop = 0
        for rn, hop_rn in self._hoppings.items():
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
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _norm_keys(rn: rn_type,
                   orb_i: int,
                   orb_j: int) -> Tuple[rn3_type, pair_type, complex]:
        """
        Normalize cell index and orbital pair into permitted keys of self.dict.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
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
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _norm_keys(rn: rn_type,
                   orb_i: int,
                   orb_j: int) -> Tuple[rn3_type, pair_type, complex]:
        """
        Normalize cell index and orbital pair into permitted keys of self.dict.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
        :return: (rn, pair, conj)
            where rn is the normalized cell index,
            pair is the normalized orbital pair,
            conj is the flag of whether to take the conjugate of hopping energy
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        rn, legal = check_rn(rn)
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
    is intended for compatibility reasons and follows a different philosophy
    than IntraHopping.

    Attributes
    ----------
    _hoppings: Dict[Tuple[int, int, int], Union[coo_matrix, np.ndarray]]
        Keys should be cell indices and values should be complex matrices.
    _num_orb: int
        number of orbitals
    _mat_shape: Tuple[int, int]
        shape of hopping matrices
    """
    def __init__(self, num_orb: int) -> None:
        """
        :param num_orb: number of orbitals
        """
        self._hoppings = dict()
        self._num_orb = num_orb
        self._mat_shape = (self._num_orb, self._num_orb)

    def __setitem__(self, rn: rn_type, hop_mat: np.ndarray) -> None:
        """
        Add or update a hopping matrix according to cell index.

        :param rn: cell index of hopping matrix
        :param hop_mat: (num_orb, num_orb) complex128 array
            hopping matrix
        :return: None
        :raises CellIndexLenError: if len(rn) != 2 or 3
        :raises ValueError: if hop_mat.shape does not match number of orbitals
        :raises PCHopDiagonalError: if on-site energies are included in hopping
            matrix
        """
        # Check cell index
        rn = self._check_rn(rn)

        # Check matrix size
        if not isinstance(hop_mat, np.ndarray):
            hop_mat = np.array(hop_mat)
        if hop_mat.shape != self._mat_shape:
            raise ValueError(f"Shape of hopping matrix {hop_mat.shape} does not "
                             f"match {self._mat_shape}")

        # Set hopping matrix
        # We copy the hopping matrix to avoid changing it accidentally.
        self._hoppings[rn] = hop_mat.copy()

        # Check for diagonal terms
        if rn == (0, 0, 0):
            self._check_diag()

    def __getitem__(self, rn: rn_type) -> np.ndarray:
        """
        Get the hopping matrix according to cell index.

        :param rn: cell index of hopping matrix
        :return: (num_orb, num_orb) complex128 array
            hopping matrix of rn
        :return: None
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        rn = self._check_rn(rn)
        try:
            hop_mat = self._hoppings[rn]
        except KeyError:
            hop_mat = self._hoppings[rn] = np.zeros(self._mat_shape,
                                                    dtype=np.complex128)
        return hop_mat

    @staticmethod
    def _check_rn(rn: rn_type) -> rn3_type:
        """
        Check and complete cell index.

        :param rn: cell index to check
        :return rn: checked cell index
        :raises CellIndexLenError: if len(rn) is not 2 or 3
        """
        rn, legal = check_rn(rn)
        if not legal:
            raise exc.CellIndexLenError(rn)
        return rn

    def _check_diag(self) -> None:
        """
        Check for diagonal hopping terms (on-site energies).

        :return: None
        :raises PCHopDiagonalError: if on-site energies are included in hopping
            matrix
        """
        rn = (0, 0, 0)
        if rn in self._hoppings.keys():
            hop_mat = self._hoppings[rn]
            for i, energy in enumerate(np.abs(np.diag(hop_mat))):
                if energy >= 1e-5:
                    raise exc.PCHopDiagonalError(rn, i)

    def to_spare(self) -> None:
        """
        Convert hopping matrices from dense to sparse.

        :return: None
        :raises PCHopDiagonalError: if on-site energies are included in hopping
            matrix
        """
        self._check_diag()
        new_hoppings = dict()
        for rn, hop_mat in self._hoppings.items():
            new_hoppings[rn] = coo_matrix(hop_mat)
        self._hoppings = new_hoppings

    @property
    def num_orb(self) -> int:
        """Interface for the 'num_orb' attribute."""
        return self.num_orb

    @num_orb.setter
    def num_orb(self, num_orb) -> None:
        """Interface for the 'num_orb' attribute."""
        self._num_orb = num_orb
        self._mat_shape = (self._num_orb, self._num_orb)

    @property
    def mat_shape(self) -> Tuple[int, int]:
        """Interface for the 'mat_shape' attribute."""
        return self._mat_shape

    @property
    def hoppings(self) -> Dict[rn3_type, Union[coo_matrix, np.ndarray]]:
        """Interface for the 'hoppings' attribute."""
        return self._hoppings
