"""Base functions and classes used through the builder package."""

from collections import namedtuple
from typing import List, Tuple, Union, Dict, Any, Iterable

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


def check_conj(hop_ind: Tuple[int, ...], i: int = 0) -> bool:
    """
    Check whether to take the conjugate part of the hopping term.

    :param hop_ind: (r_a, r_b, r_c, orb_i, orb_j), hopping index
    :param i: component index
    :return: whether to take conjugate
    """
    if hop_ind[i] > 0:
        return False
    elif hop_ind[i] < 0:
        return True
    else:
        if i < 2:
            return check_conj(hop_ind, i+1)
        else:
            return hop_ind[3] > hop_ind[4]


Orbital = namedtuple("Orbital", ["position", "energy", "label"])


class Observable:
    """
    Base class for all observable objects.

    Attributes
    ----------
    __locked: bool
        whether the object is locked
    __subscribers: Dict[str, Any]
        names and subscribers to this object
    """
    def __init__(self) -> None:
        self.__locked = False
        self.__subscribers = dict()

    def add_subscriber(self, sub_name: str, sub_obj: Any) -> None:
        """
        Add a new subscriber.

        :param sub_name: subscriber name
        :param sub_obj: subscriber object
        :return: None
        """
        self.__subscribers[sub_name] = sub_obj

    def lock(self, sub_name: str) -> None:
        """
        Lock the object. Modifications are not allowed then unless the 'unlock'
        method is called.

        :param sub_name: identifier of the subscriber
        :return: None
        :raises ValueError: if sub_name is not in subscribers
        """
        if sub_name not in self.__subscribers.keys():
            raise ValueError(f"{sub_name} not in subscribers")
        self.__locked = True

    def unlock(self) -> None:
        """
        Unlock the object. Modifications are allowed then.

        :return: None
        """
        self.__locked = False

    def check_lock(self) -> None:
        """
        Check the lock state of the object.

        :return: None
        :raises LockError: if the object is locked
        """
        if self.__locked:
            raise exc.LockError()

    def sync_array(self, **kwargs) -> None:
        """Abstract interface for 'sync_array' method of derived classes."""
        pass

    def update(self) -> None:
        """
        Update all subscribers.

        :return: None
        """
        self.sync_array()
        for sub_name, sub_obj in self.__subscribers.items():
            print(f"Updating {sub_name}")
            sub_obj.update()


class IntraHopping:
    """
    Container class for holding hopping terms in a primitive cell or
    modifications to hopping terms in a supercell.

    Attributes
    ----------
    __hoppings: Dict[Tuple[int, int, int], Dict[Tuple[int, int], complex]]
        Keys are cell indices (rn), while values are dictionaries.
        Keys of value dictionary are orbital pairs, while values are hopping
        energies.

    NOTES
    -----
    This class is intended to be utilized by the PrimitiveCell and SuperCell
    classes. It is assumed that the applicant will take care of all the
    arguments passed to this class. NO CHECKING WILL BE PERFORMED HERE.
    """
    def __init__(self) -> None:
        super().__init__()
        self.__hoppings = {}

    def __hash__(self) -> int:
        """Return the hash of this instance."""
        return hash(tuple(self.to_list()))

    @staticmethod
    def _norm_keys(rn: rn_type,
                   orb_i: int,
                   orb_j: int) -> Tuple[rn3_type, pair_type, bool]:
        """
        Normalize cell index and orbital pair into permitted keys.

        For IntraHopping, it should check whether to take the conjugation and
        return the status in conj.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
        :return: (rn, pair, conj)
            where rn is the normalized cell index,
            pair is the normalized orbital pair,
            conj is the flag of whether to take the conjugate of hopping energy
        """
        pair = (orb_i, orb_j)
        conj = check_conj(rn + pair)
        if conj:
            rn = (-rn[0], -rn[1], -rn[2])
            pair = (orb_j, orb_i)
        return rn, pair, conj

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
            hop_rn = self.__hoppings[rn]
        except KeyError:
            hop_rn = self.__hoppings[rn] = dict()
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
            energy = self.__hoppings[rn][pair]
            status = True
        except KeyError:
            energy = None
            status = False
        if status and conj:
            energy = energy.conjugate()
        return energy, status

    def remove_hopping(self, rn: rn_type,
                       orb_i: int,
                       orb_j: int) -> bool:
        """
        Remove an existing hopping term.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
        :return: where the hopping term is removed, False if not found
        """
        rn, pair, conj = self._norm_keys(rn, orb_i, orb_j)
        try:
            self.__hoppings[rn].pop(pair)
            status = True
        except KeyError:
            status = False
        return status

    def remove_orbital(self, orb_i: int) -> None:
        """
        Wrapper over 'remove_orbitals' to remove a single orbital.

        :param orb_i: orbital index to remove
        :return: None
        :raises LockError: if the object is locked
        """
        self.remove_orbitals([orb_i])

    def remove_orbitals(self, indices: Union[Iterable[int], np.ndarray]) -> None:
        """
        Remove the hopping terms corresponding to a list of orbitals and update
        remaining hopping terms.

        :param indices: indices of orbitals to remove
        :return: None
        """
        indices = sorted(set(indices))
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

        for rn, hop_rn in self.__hoppings.items():
            new_hop_rn = dict()
            for pair in hop_rn.keys():
                ii, jj = pair
                if ii in indices or jj in indices:
                    pass
                else:
                    ii = _remap(ii)
                    jj = _remap(jj)
                    new_hop_rn[(ii, jj)] = hop_rn[pair]
            self.__hoppings[rn] = new_hop_rn

    def remove_rn(self, rn: rn_type) -> bool:
        """
        Remove all the hopping terms of given cell index.

        :param rn: (r_a, r_b, r_c), cell index
        :return: where the hopping terms are removed, False if not found
        """
        rn, pair, conj = self._norm_keys(rn, 0, 0)
        try:
            self.__hoppings.pop(rn)
            status = True
        except KeyError:
            status = False
        return status

    def purge(self) -> None:
        """
        Remove empty cell indices.

        :return: None
        """
        for rn in list(self.__hoppings.keys()):
            if len(self.__hoppings[rn]) == 0:
                self.__hoppings.pop(rn)

    def to_list(self) -> List[Tuple[int, int, int, int, int, complex]]:
        """
        Flatten all hopping terms into a list.

        :return: list of hopping terms (rb, rb, rc, orb_i, orb_j, energy)
        """
        self.purge()
        hop_list = [rn + pair + (energy,)
                    for rn, hop_rn in self.__hoppings.items()
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
                   for rn, hop_rn in self.__hoppings.items()
                   for pair, energy in hop_rn.items()]
        hop_eng = [energy
                   for rn, hop_rn in self.__hoppings.items()
                   for pair, energy in hop_rn.items()]
        if use_int64:
            hop_ind = np.array(hop_ind, dtype=np.int64)
        else:
            hop_ind = np.array(hop_ind, dtype=np.int32)
        hop_eng = np.array(hop_eng, dtype=np.complex128)
        return hop_ind, hop_eng

    @property
    def hoppings(self) -> Dict[rn3_type, Dict[rn2_type, complex]]:
        """Interface for the '__hoppings' attribute."""
        return self.__hoppings

    @property
    def cell_indices(self) -> List[rn3_type]:
        """
        Get the list of cell indices.

        :return: list of cell indices
        """
        return list(self.__hoppings.keys())

    @property
    def num_hop(self) -> int:
        """
        Count the number of hopping terms.

        :return: number of hopping terms
        """
        num_hop = 0
        for rn, hop_rn in self.__hoppings.items():
            num_hop += len(hop_rn)
        return num_hop


class InterHopping(Observable, IntraHopping):
    """
    Container class for holding hopping terms between two models.

    Attributes
    ----------
    __model_bra: 'PrimitiveCell' or 'SuperCell' instance
        the 'bra' model from which the hopping terms exist
    __model_ket: 'PrimitiveCell' or 'SuperCell' instance
        the 'ket' model to which the hopping terms exist
    """
    def __init__(self, model_bra: Any, model_ket: Any) -> None:
        """
        :param model_bra: the 'bra' model from which the hopping terms exist
        :param model_ket: the 'ket' model to which the hopping terms exist
        """
        Observable.__init__(self)
        IntraHopping.__init__(self)
        self.__model_bra = model_bra
        self.__model_ket = model_ket

    def __hash__(self) -> int:
        """Return the hash of this instance."""
        fp = (tuple(self.to_list()), self.__model_bra, self.__model_ket)
        return hash(fp)

    @staticmethod
    def _norm_keys(rn: rn_type,
                   orb_i: int,
                   orb_j: int) -> Tuple[rn3_type, pair_type, bool]:
        """
        Normalize cell index and orbital pair into permitted keys.

        For InterHopping, it should check if rn is legal since the class is
        exposed to the user. The status conj should always be False.

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

    def add_hopping(self, rn: rn_type,
                    orb_i: int,
                    orb_j: int,
                    energy: complex) -> None:
        """
        Add a new hopping term or update existing term.

        :param rn: (r_a, r_b, r_c), cell index
        :param orb_i: orbital index or bra
        :param orb_j: orbital index of ket
        :param energy: hopping energy
        :return: None
        :raises LockError: is the object is locked
        """
        self.check_lock()
        super().add_hopping(rn, orb_i, orb_j, energy)


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
        """Interface for the '_num_orb' attribute."""
        return self._num_orb

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
