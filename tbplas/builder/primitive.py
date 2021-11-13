"""
Functions and classes for orbitals, hopping terms and primitive cell.

Functions
---------
    correct_coord: developer API
        check and auto-complete cell index, orbital coordinate,
        super-cell dimension and periodic condition
    extend_prim_cell: user API
        extend primitive cell along a, b, and c directions
        reserved for compatibility with old version of TBPlaS

Classes
-------
    Orbital: developer class
        class for representing an orbital in TB model
    Hopping: developer class
        class for representing a hopping term in TB model
    LockableObject: developer class
        base class for all lockable classes
    PrimitiveCell: user class
        class for representing a primitive cell from which a super cell
        can be created
"""

import math

import numpy as np
import scipy.linalg.lapack as lapack
import matplotlib.pyplot as plt

from . import constants as consts
from . import lattice as lat
from . import kpoints as kpt
from . import exceptions as exc
from . import core


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
    """
    def __init__(self, position, energy=0.0) -> None:
        """
        :param position: tuple with 3 floats
            FRACTIONAL coordinate of the orbital
        :param energy: float
            on-site energy of the orbital in eV
        :return: None
        """
        assert len(position) == 3
        self.position = position
        self.energy = energy


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

    def check_lock_state(self):
        """
        Check and raise an error if the object is locked.

        :return: None.
        :raises LockError: is the the object is locked
        """
        if self.is_locked:
            raise exc.LockError()


class PrimitiveCell(LockableObject):
    """
    Class for representing a primitive cell, from which the super cell
    can be built.

    Attributes
    ----------
    lat_vec: (3, 3) float64 array
        Cartesian coordinates of lattice vectors in NANO METER
        Each ROW corresponds to one lattice vector.
    orbital_list: list of instances of 'Orbital' class
        list of orbitals in the primitive cell
    hopping_list: list of instances of 'Hopping' class
        list of hopping terms in the primitive cell
        Only half of the hopping terms is stored.
        Conjugate terms are added automatically when constructing
        the Hamiltonian.
    hash_dict: dictionary
        dictionary of hashes of tuple(orbital_list) and tuple(hopping_list)
        Keys should be either 'orb' or 'hop'.
        Method 'sync_array' will use this dictionary to update the arrays.
        Should only be accessed within that method!
    orb_pos: (num_orb, 3) float64 array
        collection of FRACTIONAL positions of all orbitals
    orb_eng: (num_orb,) float64 array
        collection of on-site energies of all orbitals in eV
    hop_ind: (num_hop, 5) int32 array
        collection of indices of all hopping terms
    hop_eng: (num_hop,) complex128 array
        collection of energies of all hopping terms in eV
    extended: integer
        number of times the primitive cell has been extended
        reserved for compatibility with old version of TBPlaS
    """
    def __init__(self, lat_vec: np.ndarray, unit=consts.ANG) -> None:
        """
        :param lat_vec: (3, 3) float64 array
            Cartesian coordinates of lattice vectors in arbitrary unit
        :param unit: float
            conversion coefficient from arbitrary unit to NM
        :return: None
        :raises LatVecError: if shape of lat_vec is not (3, 3)
        """
        super().__init__()

        # Setup lattice vectors
        if not isinstance(lat_vec, np.ndarray):
            lat_vec = np.array(lat_vec)
        if lat_vec.shape != (3, 3):
            raise exc.LatVecError()
        self.lat_vec = lat_vec * unit

        # Setup orbital and hopping lists
        self.orbital_list = []
        self.hopping_list = []
        self.hash_dict = {'orb': hash(tuple(self.orbital_list)),
                          'hop': hash(tuple(self.hopping_list))}

        # Setup arrays.
        self.orb_pos = None
        self.orb_eng = None
        self.hop_ind = None
        self.hop_eng = None

        # Setup misc. attributes.
        self.extended = 1

    def add_orbital(self, position, energy=0.0, sync_array=False, **kwargs):
        """
        Add a new orbital to the primitive cell.

        :param position: tuple with 2 or 3 floats
            FRACTIONAL coordinate of the orbital
        :param energy: float
            on-site energy of the orbital in eV
        :param sync_array: boolean
            whether to call 'sync_array' to update numpy arrays
            according to orbitals and hopping terms
        :return: None
            self.orbital_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        # Check arguments
        try:
            self.check_lock_state()
        except exc.LockError as err:
            raise exc.PCLockError() from err
        try:
            position = correct_coord(position)
        except exc.CoordLenError as err:
            raise exc.OrbPositionLenError(err.coord) from err

        # Add the orbital
        self.orbital_list.append(Orbital(position, energy))
        if sync_array:
            self.sync_array(**kwargs)

    def set_orbital(self, orb_i, position=None, energy=None, sync_array=False,
                    **kwargs):
        """
        Modify the position and energy of an existing orbital.

        If position or energy is None, then the corresponding attribute will
        not be modified.

        :param orb_i: integer
            index of the orbital to modify
        :param position: tuple with 2 or 3 floats
            new FRACTIONAL coordinate of the orbital
        :param energy: float
            new on-site energy of the orbital in eV
        :param sync_array: boolean
            whether to call sync_array to update numpy arrays
            according to orbitals and hopping terms
        :return: None
            self.orbital_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        # Check arguments
        try:
            self.check_lock_state()
        except exc.LockError as err:
            raise exc.PCLockError() from err
        if position is not None:
            try:
                position = correct_coord(position)
            except exc.CoordLenError as err:
                raise exc.OrbPositionLenError(err.coord) from err

        # Set orbital attributes
        try:
            orbital = self.orbital_list[orb_i]
        except IndexError as err:
            raise exc.PCOrbIndexError(orb_i) from err
        if position is not None:
            orbital.position = position
        if energy is not None:
            orbital.energy = energy
        if sync_array:
            self.sync_array(**kwargs)

    def get_orbital(self, orb_i):
        """
        Get given orbital instance.

        :param orb_i: integer
            index of the orbital
        :return: orbital: instance of 'Orbital' class
            orbital corresponding to the index 'orb_i'
        :raises PCOrbIndexError: if orb_i falls out of range
        """
        try:
            orbital = self.orbital_list[orb_i]
        except IndexError as err:
            raise exc.PCOrbIndexError(orb_i) from err
        return orbital

    def remove_orbital(self, orb_i, sync_array=False, **kwargs):
        """
        Remove given orbital and associated hopping terms, then update remaining
        hopping terms.

        :param orb_i: integer
            index of the orbital to remove
        :param sync_array: boolean
            whether to call sync_array to update numpy arrays
            according to orbitals and hopping terms
        :return: None
            self.orbital_list and self.hopping_list are modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        """
        # Check arguments
        try:
            self.check_lock_state()
        except exc.LockError as err:
            raise exc.PCLockError() from err

        # Delete the orbital.
        try:
            self.orbital_list.pop(orb_i)
        except IndexError as err:
            raise exc.PCOrbIndexError(orb_i) from err

        # Delete associated hopping terms.
        new_hopping = []
        for hopping in self.hopping_list:
            if orb_i not in hopping.index[3:]:
                new_hopping.append(hopping)

        # Update orbital indices in remaining hopping terms
        for hopping in new_hopping:
            index = list(hopping.index)
            if index[3] > orb_i:
                index[3] -= 1
            if index[4] > orb_i:
                index[4] -= 1
            hopping.index = tuple(index)

        # Finally, update hopping_list.
        self.hopping_list = new_hopping

        # Update arrays
        if sync_array:
            self.sync_array(**kwargs)

    def __assemble_hopping(self, rn, orb_i, orb_j, energy=0.0):
        """
        Check input and assemble a hopping item.

        :param rn: (ra, rb, rc)
            cell index of the hopping term, i.e. R
        :param orb_i: integer
            index of orbital i in <i,0|H|j,R>
        :param orb_j: integer
            index of orbital j in <i,0|H|j,R>
        :param energy: float
            hopping energy in eV
        :return: (ra, rb, rc, orb_i, orb_j)
            hopping item assembled from input
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        try:
            rn = correct_coord(rn)
        except exc.CoordLenError as err:
            raise exc.CellIndexLenError(err.coord) from err
        num_orbitals = self.num_orb
        if orb_i not in range(num_orbitals):
            raise exc.PCOrbIndexError(orb_i)
        if orb_j not in range(num_orbitals):
            raise exc.PCOrbIndexError(orb_j)
        if rn == (0, 0, 0) and orb_i == orb_j:
            raise exc.PCHopDiagonalError(rn, orb_i)
        hopping = Hopping(rn, orb_i, orb_j, energy)
        return hopping

    def __find_equiv_hopping(self, ref_hopping):
        """
        Find the indices of equivalent hopping term, i.e. the same term or
        its conjugate counterpart.

        :param ref_hopping: (ra, rb, rc, orb_i, orb_j, energy)
            reference hopping term
        :return: id_same: integer
            index of the same hopping term, none if not found
        :return: id_conj: integer
            index of the conjugate hopping term, none if not found
        """
        def __check_conj(index1, index2):
            return (index1[0] == -index2[0] and
                    index1[1] == -index2[1] and
                    index1[2] == -index2[2] and
                    index1[3] == index2[4] and
                    index1[4] == index2[3])

        id_same, id_conj = None, None
        for i_hop, hopping in enumerate(self.hopping_list):
            if ref_hopping.index == hopping.index:
                id_same = i_hop
                break
            elif __check_conj(ref_hopping.index, hopping.index):
                id_conj = i_hop
                break
            else:
                pass
        return id_same, id_conj

    def add_hopping(self, rn, orb_i, orb_j, energy, sync_array=False, **kwargs):
        """
        Add a new hopping term to the primitive cell, or update an existing
        hopping term.

        :param rn: (ra, rb, rc)
            cell index of the hopping term, i.e. R
        :param orb_i: integer
            index of orbital i in <i,0|H|j,R>
        :param orb_j:
            index of orbital j in <i,0|H|j,R>
        :param energy: float
            hopping integral in eV
        :param sync_array: boolean
            whether to call sync_array to update numpy arrays
            according to orbitals and hopping terms
        :return: None
            self.hopping_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        try:
            self.check_lock_state()
        except exc.LockError as err:
            raise exc.PCLockError() from err
        new_hopping = self.__assemble_hopping(rn, orb_i, orb_j, energy)
        id_same, id_conj = self.__find_equiv_hopping(new_hopping)

        # Update existing hopping term, or add the new term to hopping_list.
        if id_same is not None:
            exist_hopping = self.hopping_list[id_same]
            print("INFO: updating existing hopping term")
            print(" ", exist_hopping.index, exist_hopping.energy,
                  "->", new_hopping.energy)
            exist_hopping.energy = new_hopping.energy
        elif id_conj is not None:
            exist_hopping = self.hopping_list[id_conj]
            print("INFO: updating existing conjugate hopping term")
            print(" ", exist_hopping.index, exist_hopping.energy,
                  "->", new_hopping.energy.conjugate())
            exist_hopping.energy = new_hopping.energy.conjugate()
        else:
            self.hopping_list.append(new_hopping)
        if sync_array:
            self.sync_array(**kwargs)

    def get_hopping(self, rn, orb_i, orb_j):
        """
        Get given hopping term or its conjugate counterpart.

        :param rn: (ra, rb, rc)
            cell index of the hopping term, i.e. R
        :param orb_i: integer
            index of orbital i in <i,0|H|j,R>
        :param orb_j:
            index of orbital j in <i,0|H|j,R>
        :return: instance of 'Hopping' class
            hopping term or its conjugate counterpart
        :raises PCHopNotFoundError: if rn + (orb_i, orb_j) or its conjugate
            counterpart is not found in the hopping terms
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        new_hopping = self.__assemble_hopping(rn, orb_i, orb_j)
        id_same, id_conj = self.__find_equiv_hopping(new_hopping)

        # Return the hopping term directly, or its conjugate counterpart.
        if id_same is not None:
            return self.hopping_list[id_same]
        elif id_conj is not None:
            print("INFO: given hopping term not found."
                  " Returning conjugate counterpart instead.")
            return self.hopping_list[id_conj]
        else:
            raise exc.PCHopNotFoundError(new_hopping.index)

    def remove_hopping(self, rn, orb_i, orb_j, sync_array=False, **kwargs):
        """
        Remove given hopping term.

        :param rn: (ra, rb, rc)
            cell index of the hopping term, i.e. R
        :param orb_i: integer
            index of orbital i in <i,0|H|j,R>
        :param orb_j: integer
            index of orbital j in <i,0|H|j,R>
        :param sync_array: boolean
            whether to call sync_array to update numpy arrays
            according to orbitals and hopping terms
        :return: None
            self.hopping_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCHopNotFoundError: if rn + (orb_i, orb_j) or its conjugate
            counterpart is not found in the hopping terms
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        try:
            self.check_lock_state()
        except exc.LockError as err:
            raise exc.PCLockError() from err
        new_hopping = self.__assemble_hopping(rn, orb_i, orb_j)
        id_same, id_conj = self.__find_equiv_hopping(new_hopping)

        # Remove the given item or its conjugate counterpart.
        if id_same is not None:
            exist_hopping = self.hopping_list[id_same]
            print("INFO: removing hopping term")
            print(" ", exist_hopping.index, exist_hopping.energy)
            self.hopping_list.pop(id_same)
            if sync_array:
                self.sync_array(**kwargs)
        elif id_conj is not None:
            exist_hopping = self.hopping_list[id_conj]
            print("INFO: removing conjugate hopping term")
            print(" ", exist_hopping.index, exist_hopping.energy)
            self.hopping_list.pop(id_conj)
            if sync_array:
                self.sync_array(**kwargs)
        else:
            raise exc.PCHopNotFoundError(new_hopping.index)

    @property
    def num_orb(self):
        """
        Get the number of orbitals in the primitive cell.

        :return: integer, number of orbitals
        """
        return len(self.orbital_list)

    def get_lattice_area(self, direction="c"):
        """
        Get the area formed by lattice vectors normal to given direction.

        :param direction: string, should be in ("a", "b", "c")
            direction of area, e.g. "c" indicates the area formed by lattice
            vectors in the aOb plane.
        :return: float, area formed by lattice vectors in NM^2.
        """
        return lat.get_lattice_area(self.lat_vec, direction)

    def get_lattice_volume(self):
        """
        Get the volume formed by all three lattice vectors in NM^3.

        :return: float, volume in NM^3.
        """
        return lat.get_lattice_volume(self.lat_vec)

    def get_reciprocal_vectors(self):
        """
        Get the Cartesian coordinates of reciprocal lattice vectors in 1/NM.

        :return: (3, 3) float64 array, reciprocal vectors in 1/NM.
        """
        return lat.gen_reciprocal_vectors(self.lat_vec)

    def sync_array(self, verbose=False, force_sync=False):
        """
        Synchronize orb_pos, orb_eng, hop_ind and hop_eng according to
        orbital_list and hopping_list.

        :param verbose: boolean
            whether to output additional debugging information
        :param force_sync: boolean
            whether to force synchronizing the arrays even if orbital_list
            and hopping_list did not change
        :return: None
            self.orb_pos, self.orb_eng, self.hop_ind and self.hop_eng
            are modified.
        """
        new_orbital_hash = hash(tuple(self.orbital_list))
        if force_sync or new_orbital_hash != self.hash_dict['orb']:
            if verbose:
                print("INFO: updating pc orbital arrays")
            self.hash_dict['orb'] = new_orbital_hash
            # If orbital_list is not [], update as usual.
            if len(self.orbital_list) != 0:
                self.orb_pos = np.array(
                    [orb.position for orb in self.orbital_list], dtype=np.float64)
                self.orb_eng = np.array(
                    [orb.energy for orb in self.orbital_list], dtype=np.float64)
            # Otherwise, restore to default settings as in __init__.
            else:
                self.orb_pos = None
                self.orb_eng = None
        else:
            if verbose:
                print("INFO: no need to update pc orbital arrays")

        new_hopping_hash = hash(tuple(self.hopping_list))
        if force_sync or new_hopping_hash != self.hash_dict['hop']:
            if verbose:
                print("INFO: updating pc hopping arrays")
            self.hash_dict['hop'] = new_hopping_hash
            # if hopping_list is not [], update as usual.
            if len(self.hopping_list) != 0:
                self.hop_ind = np.array(
                    [hop.index for hop in self.hopping_list], dtype=np.int32)
                self.hop_eng = np.array(
                    [hop.energy for hop in self.hopping_list], dtype=np.complex128)
            # Otherwise, restore to default settings as in __init__.
            else:
                self.hop_ind = None
                self.hop_eng = None
        else:
            if verbose:
                print("INFO: no need to update pc hopping arrays")

    def __plot_cell(self, axes: plt.Axes, cell_index=(0, 0, 0)):
        """
        Plot lattice vectors, orbitals and hopping terms for given
        primitive cell.

        :param axes: instance of matplotlib 'Axes'
            axes on which the figure will be plot
        :param cell_index: (ia, ib, ic)
            index of the primitive cell to plot
        :return: None
        """
        # Shift lattice vectors according to cell index.
        center = np.matmul(cell_index, self.lat_vec)
        vec_0 = self.lat_vec[0] + center
        vec_1 = self.lat_vec[1] + center
        vec_2 = self.lat_vec[0] + self.lat_vec[1] + center

        # For (0, 0, 0) cell we draw lattice vectors as arrows.
        if cell_index == (0, 0, 0):
            axes.arrow(center[0], center[1], vec_0[0], vec_0[1],
                       color="k", length_includes_head=True, width=0.005,
                       head_width=0.02)
            axes.arrow(center[0], center[1], vec_1[0], vec_1[1],
                       color="k", length_includes_head=True, width=0.005,
                       head_width=0.02)
        # For other cells, draw lattice vectors as lines.
        else:
            axes.plot((center[0], vec_0[0]), (center[1], vec_0[1]),
                      color='k', ls=":")
            axes.plot((center[0], vec_1[0]), (center[1], vec_1[1]),
                      color='k', ls=":")
        # Draw lines parallel to lattice vectors.
        axes.plot((vec_0[0], vec_2[0]), (vec_0[1], vec_2[1]),
                  color='k', ls=":")
        axes.plot((vec_1[0], vec_2[0]), (vec_1[1], vec_2[1]),
                  color='k', ls=":")

        # Plot orbitals
        cart_pos = lat.frac2cart(self.lat_vec, self.orb_pos) + center
        axes.scatter(cart_pos[:, 0], cart_pos[:, 1], s=100, c=self.orb_eng)

        # Plot hopping terms
        for hopping in self.hop_ind:
            rn = hopping[:3]
            orb_i, orb_j = hopping[3], hopping[4]
            pos_i = cart_pos[orb_i]
            pos_j = cart_pos[orb_j] + np.matmul(rn, self.lat_vec)
            diff_pos = pos_j - pos_i
            axes.scatter(pos_j[0], pos_j[1], s=100, c=self.orb_eng[orb_j])
            axes.arrow(pos_i[0], pos_i[1], diff_pos[0], diff_pos[1], color='r',
                       length_includes_head=True, width=0.002, head_width=0.02,
                       fill=False)

    def plot(self, fig_name=None, fig_dpi=300):
        """
        Plot lattice vectors, orbitals, and hopping terms.

        If figure name is give, save the figure to file. Otherwise, show it on
        the screen.

        :param fig_name: string
            file name to which the figure will be saved
        :param fig_dpi: integer
            resolution of the figure file
        :returns: None
        """
        fig, axes = plt.subplots()
        axes.set_aspect('equal')

        # Determine the size of super cell in which the primitive cell resides.
        self.sync_array()
        sc_size = [1, 1, 1]
        for i_dim in range(3):
            sc_size[i_dim] = np.abs(self.hop_ind[:, i_dim]).max()

        # Plot cells one-by-one.
        for i_a in range(-sc_size[0], sc_size[0]+1):
            for i_b in range(-sc_size[1], sc_size[1]+1):
                for i_c in range(-sc_size[2], sc_size[2]+1):
                    self.__plot_cell(axes, (i_a, i_b, i_c))

        # Hide spines and ticks.
        for key in ("top", "bottom", "left", "right"):
            axes.spines[key].set_visible(False)
        axes.set_xticks([])
        axes.set_yticks([])
        fig.tight_layout()
        if fig_name is not None:
            plt.savefig(fig_name, dpi=fig_dpi)
        else:
            plt.show()

    def print(self):
        """
        Print orbital and hopping information.

        :return: None
        """
        print("Orbitals:")
        for orbital in self.orbital_list:
            pos = orbital.position
            pos_fmt = "%10.5f%10.5f%10.5f" % (pos[0], pos[1], pos[2])
            print(pos_fmt, orbital.energy)
        print("Hopping terms:")
        for hopping in self.hopping_list:
            print(" ", hopping.index, hopping.energy)

    def calc_bands(self, k_path: np.ndarray):
        """
        Calculate band structure along given k_path.

        :param k_path: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points along given path
        :return: k_len: (num_kpt,) float64 array in 1/NM
            length of k-path in reciprocal space, for plotting band structure
        :return: bands: (num_kpt, num_orb) float64 array
            Energies corresponding to k-points in eV
        """
        # Initialize working arrays.
        self.sync_array()
        num_k_points = k_path.shape[0]
        bands = np.zeros((num_k_points, self.num_orb), dtype=np.float64)
        ham_k = np.zeros((self.num_orb, self.num_orb), dtype=np.complex128)

        # Get length of k-path in reciprocal space
        k_len = kpt.gen_kdist(self.lat_vec, k_path)

        # Loop over k-points to evaluate the energies
        for i_k, k_point in enumerate(k_path):
            ham_k *= 0.0
            core.set_ham(self.orb_pos, self.orb_eng,
                         self.hop_ind, self.hop_eng,
                         k_point, ham_k)
            eigenvalues, eigenstates, info = lapack.zheev(ham_k)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            bands[i_k, :] = eigenvalues[:]
        return k_len, bands

    def calc_dos(self, k_points: np.ndarray, e_min=None, e_max=None,
                 e_step=0.05, sigma=0.05, basis="Gaussian"):
        """
        Calculate density of states for given energy range and step.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        :param e_min: float
            lower bound of the energy range in eV
        :param e_max: float
            upper hound of the energy range in eV
        :param e_step: float
            energy step in eV
        :param sigma: float
            broadening parameter in eV
        :param basis: string
            basis function to approximate the Delta function
            should be either "Gaussian" or "Lorentzian"
        :return: energies: (num_grid,) float64 array
            energy grid corresponding to e_min, e_max and e_step
        :return: dos: (num_grid,) float64 array
            density of states in states/eV
        :raises BasisError: if basis is neither Gaussian or Lorentzian
        """
        # Get the band energies
        k_len, bands = self.calc_bands(k_points)

        # Create energy grid
        if e_min is None:
            e_min = bands.min()
        if e_max is None:
            e_max = bands.max()
        num_grid = int((e_max - e_min) / e_step)
        energies = np.linspace(e_min, e_max, num_grid+1)

        # Define broadening functions
        def _gaussian(x, mu):
            part_a = 1.0 / (sigma * math.sqrt(2 * math.pi))
            part_b = np.exp(-(x - mu)**2 / (2 * sigma**2))
            return part_a * part_b

        def _lorentzian(x, mu):
            part_a = 1.0 / (math.pi * sigma)
            part_b = sigma**2 / ((x - mu)**2 + sigma**2)
            return part_a * part_b

        # Evaluate DOS by collecting contributions from all energies
        dos = np.zeros(energies.shape, dtype=np.float64)
        if basis == "Gaussian":
            basis_func = _gaussian
        elif basis == "Lorentzian":
            basis_func = _lorentzian
        else:
            raise exc.BasisError(basis)
        for eng_k in bands:
            for eng_i in eng_k:
                dos += basis_func(energies, eng_i)

        # Re-normalize dos
        # For each energy in bands, we use a normalized Gaussian or Lorentzian
        # basis function to approximate the Delta function. Totally, there are
        # bands.size basis functions. So we divide dos by this number.
        dos /= bands.size
        return energies, dos


def extend_prim_cell(prim_cell: PrimitiveCell, dim=(1, 1, 1)):
    """
    Extend primitive cell along a, b and c directions.

    :param prim_cell: instance of 'PrimitiveCell'
        primitive cell from which the extended cell is built
    :param dim: (na, nb, nc)
        dimension of the extended cell along 3 directions
    :return: extend_cell: instance of 'PrimitiveCell'
        extended cell created from primitive cell
    :raises CoordLenError: if len(dim) != 2 or 3
    :raises ValueError: if dimension along any direction is smaller than 1
    """
    # Check the dimension of extended cell
    dim = correct_coord(dim, complete_item=1)
    for i_dim in range(len(dim)):
        if dim[i_dim] < 1:
            raise ValueError(f"Dimension along direction {i_dim} should not"
                             f" be smaller than 1")

    # Extend lattice vectors
    lat_vec_ext = prim_cell.lat_vec.copy()
    for i_dim in range(3):
        lat_vec_ext[i_dim] *= dim[i_dim]

    # Create extended cell and add orbitals
    extend_cell = PrimitiveCell(lat_vec_ext, unit=consts.NM)
    extend_cell.extended *= np.prod(dim)
    orb_id_pc, orb_id_sc = [], {}
    id_sc = 0
    for i_a in range(dim[0]):
        for i_b in range(dim[1]):
            for i_c in range(dim[2]):
                for i_o, orbital in enumerate(prim_cell.orbital_list):
                    id_pc = (i_a, i_b, i_c, i_o)
                    orb_id_pc.append(id_pc)
                    orb_id_sc[id_pc] = id_sc
                    id_sc += 1
                    pos_ext = [(orbital.position[0] + i_a) / dim[0],
                               (orbital.position[1] + i_b) / dim[1],
                               (orbital.position[2] + i_c) / dim[2]]
                    extend_cell.add_orbital(pos_ext, orbital.energy)

    # Define periodic boundary condition.
    def _wrap_pbc(ji, ni):
        return ji % ni, ji // ni

    # Add hopping terms
    for id_sc_i in range(extend_cell.num_orb):
        id_pc_i = orb_id_pc[id_sc_i]
        for hopping in prim_cell.hopping_list:
            hop_ind = hopping.index
            if id_pc_i[3] == hop_ind[3]:
                ja, na = _wrap_pbc(id_pc_i[0] + hop_ind[0], dim[0])
                jb, nb = _wrap_pbc(id_pc_i[1] + hop_ind[1], dim[1])
                jc, nc = _wrap_pbc(id_pc_i[2] + hop_ind[2], dim[2])
                id_pc_j = (ja, jb, jc, hop_ind[4])
                id_sc_j = orb_id_sc[id_pc_j]
                rn = (na, nb, nc)
                extend_cell.add_hopping(rn, id_sc_i, id_sc_j, hopping.energy)

    extend_cell.sync_array()
    return extend_cell
