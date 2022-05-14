"""
Base functions and classes used through the builder package

Functions
---------
    correct_coord: developer function
        check and auto-complete cell index, orbital coordinate,
        super-cell dimension and periodic condition
    invert_rn: developer function
        check if the cell index should be inverted

Classes
-------
    Orbital: developer class
        abstraction for orbitals in TB model
    LockableObject: developer class
        base class for all lockable classes
    PCIntraHopping: developer class
        container for holding hopping terms of a primitive cell
    HopDict: user class
        container for holding hopping terms of a primitive cell
        reserved for compatibility with old version of TBPlaS
"""

import numpy as np

from . import exceptions as exc


def correct_coord(coord, complete_item=0):
    """
    Check and complete cell index, orbital coordinate, etc.

    :param coord: tuple with 2 integers or floats
        incoming coordinate to check
    :param complete_item: integer or float
        item to be appended to coord if its length is 2
    :return: coord: tuple with 3 integers or floats
        corrected and completed coordinate
    :raises CoordLenError: if length of coord is not 2 or 3
    """
    if len(coord) not in (2, 3):
        raise exc.CoordLenError(coord)
    coord = tuple(coord)
    if len(coord) == 2:
        if isinstance(coord[0], int):
            coord += (int(complete_item),)
        else:
            coord += (float(complete_item),)
    return coord


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


class Orbital:
    """
    Class for representing an orbital in TB model.

    Attributes
    ----------
    position: tuple with 3 floats
        FRACTIONAL coordinate of the orbital
    energy: float
        on-site energy of the orbital in eV
    label: string
        orbital label
    """
    def __init__(self, position, energy=0.0, label="X") -> None:
        """
        :param position: tuple with 3 floats
            FRACTIONAL coordinate of the orbital
        :param energy: float
            on-site energy of the orbital in eV
        :param label: string
            orbital label
        :return: None
        """
        assert len(position) == 3
        self.position = position
        self.energy = energy
        self.label = label

    def __hash__(self):
        """Return hash value of this instance."""
        return hash(self.position + (self.energy, self.label))


class LockableObject:
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
        Unlock the primitive cell. Modifications are allowed then.

        :return: None
        """
        self.is_locked = False


class PCIntraHopping:
    """
    Container class for holding hopping terms of a primitive cell.

    NOTE: this class is intended to constitute the 'hopping_dict' attribute of
    'PrimitiveCell' class. It is assumed that the caller will take care of all
     the parameters passed to this class. NO CHECKING WILL BE PERFORMED HERE.

    Attributes
    ----------
    dict: dictionary containing the hopping terms
        Keys are cell indices (rn), while values are also dictionaries.
        Keys of value dictionary are orbital pairs, while values are hopping
        energies.
    """
    def __init__(self):
        self.dict = {}

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

    def add_hopping(self, rn: tuple, orb_i: int, orb_j: int, energy: complex):
        """
        Add a new hopping term or update existing term.

        NOTE: conjugate terms are reduced by normalizing their cell indices and
        orbital pairs.

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

    def remove_hopping(self, rn: tuple, orb_i: int, orb_j: int):
        """
        Remove an existing hopping term.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :param int orb_i: orbital index or bra
        :param int orb_j: orbital index of ket
        :return: status
            where the hopping term is removed, False if not found
        """
        rn, pair, conj = self._norm_keys(rn, orb_i, orb_j)
        try:
            self.dict[rn].pop(pair)
            status = True
        except KeyError:
            status = False
        return status

    def to_array(self):
        """
        Convert hopping terms to array of 'hop_ind' and 'hop_eng',
        for constructing attributes of 'PrimitiveCell'.

        :return: (hop_ind, hop_eng)
            hop_ind: (num_hop, 5) int32 array, hopping indices
            hop_eng: (num_hop,) complex128 array, hopping energies
        """
        hop_ind, hop_eng = [], []
        for rn, hop_rn in self.dict.items():
            for pair, energy in hop_rn.items():
                hop_ind.append(rn + pair)
                hop_eng.append(energy)
        hop_ind = np.array(hop_ind, dtype=np.int32)
        hop_eng = np.array(hop_eng, dtype=np.complex128)
        return hop_ind, hop_eng

    @property
    def num_hop(self):
        """Count the number of hopping terms."""
        num_hop = 0
        for rn, hop_rn in self.dict.items():
            num_hop += len(hop_rn)
        return num_hop


class HopDict:
    """
    Class for holding hopping terms.

    Reserved for compatibility with old version of TBPlaS.

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
    def __get_minus_rn(rn):
        """Get minus cell index."""
        return tuple([-v for v in rn])

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
            or hopping matrix at (0, 0, 0) is not Hermitian
        :raises PCHopDiagonalError: if on-site energies are included in hopping
            matrix
        """
        # Check cell index
        try:
            rn = correct_coord(rn)
        except exc.CoordLenError as err:
            raise exc.CellIndexLenError(err.coord) from err

        # Check matrix size
        hop_mat = np.array(hop_mat)
        if hop_mat.shape != self.mat_shape:
            raise ValueError(f"Shape of hopping matrix {hop_mat.shape} does not "
                             f"match {self.mat_shape}")

        # Special check for (0, 0, 0) cell
        if rn == (0, 0, 0):
            for i in range(hop_mat.shape[0]):
                if abs(hop_mat.item(i, i)) >= 1e-5:
                    raise exc.PCHopDiagonalError(rn, i)
                for j in range(i+1, hop_mat.shape[0]):
                    h_ij = hop_mat.item(i, j)
                    h_ji = hop_mat.item(j, i)
                    if abs(h_ij - h_ji.conjugate()) >= 1e-5:
                        raise ValueError(f"Hopping matrix at (0, 0, 0) is not "
                                         f"Hermitian")

        # Set hopping matrix
        if rn in self.dict.keys():
            self.dict[rn] = hop_mat
        else:
            minus_rn = self.__get_minus_rn(rn)
            if minus_rn in self.dict.keys():
                self.dict[minus_rn] = hop_mat.T.conj()
            else:
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
        try:
            rn = correct_coord(rn)
        except exc.CoordLenError as err:
            raise exc.CellIndexLenError(err.coord) from err

        # Check element indices
        if element[0] not in range(self.mat_shape[0]) or \
                element[1] not in range(self.mat_shape[1]):
            raise ValueError(f"Element {element} out of range {self.mat_shape}")

        # Check for on-site energy
        if rn == (0, 0, 0) and element[0] == element[1]:
            raise exc.PCHopDiagonalError(rn, element[0])

        # Set matrix element
        if rn in self.dict.keys():
            self.dict[rn][element[0], element[1]] = hop
        else:
            minus_rn = self.__get_minus_rn(rn)
            if minus_rn in self.dict.keys():
                self.dict[minus_rn][element[1], element[0]] = hop.conjugate()
            else:
                self.dict[rn] = np.zeros(self.mat_shape, dtype=complex)
                self.dict[rn][element[0], element[1]] = hop

        # Special treatment for (0, 0, 0) cell
        if rn == (0, 0, 0):
            self.dict[rn][element[1], element[0]] = hop.conjugate()

    def delete_mat(self, rn):
        """
        Delete hopping matrix from dictionary.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :returns: None
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        # Check cell index
        try:
            rn = correct_coord(rn)
        except exc.CoordLenError as err:
            raise exc.CellIndexLenError(err.coord) from err

        # Delete hopping matrix
        if rn in self.dict.keys():
            self.dict.pop(rn, None)
        else:
            minus_rn = self.__get_minus_rn(rn)
            if minus_rn in self.dict.keys():
                self.dict.pop(minus_rn, None)
