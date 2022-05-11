"""
Base functions and classes used through the builder package

Functions
---------
    correct_coord: developer function
        check and auto-complete cell index, orbital coordinate,
        super-cell dimension and periodic condition

Classes
-------
    Orbital: developer class
        abstraction for orbitals in TB model
    Hopping: developer class
        abstraction for hopping terms in TB model
    LockableObject: developer class
        base class for all lockable classes
    HopDict: user class
        container for holding hopping terms
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


class Hopping:
    """
    Class for representing a hopping term in TB model.

    Attributes
    ----------
    index: (ra, rb, rc, orb_i, orb_j)
        cell and orbital indices of the hopping term
    energy: float or complex
        hopping energy in eV
    """
    def __init__(self, rn, orb_i, orb_j, energy=0.0) -> None:
        """
        :param rn: (ra, rb, rc)
            cell index of the hopping term
        :param orb_i: integer
            orbital index of bra of the hopping term
        :param orb_j: integer
            orbital index of ket of the hopping term
        :param energy: float or complex
            hopping energy of the hopping term in eV
        :return: None
        """
        assert len(rn) == 3
        self.index = rn + (orb_i, orb_j)
        self.energy = energy

    def __hash__(self):
        """Return hash value of this instance."""
        return hash(self.index + (self.energy,))


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
