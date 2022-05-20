"""
Functions and classes for orbitals, hopping terms and primitive cell.

Functions
---------
    None

Classes
-------
    PrimitiveCell: user class
        abstraction for  primitive cell from which a super cell
        can be created
"""

import math
import time

import numpy as np
import scipy.linalg.lapack as lapack
import matplotlib.pyplot as plt

from . import constants as consts
from . import lattice as lat
from . import kpoints as kpt
from . import exceptions as exc
from . import core
from .base import (correct_coord, Orbital, LockableObject, PCIntraHopping,
                   HopDict)
from .utils import ModelViewer


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
    hopping_dict: instance of 'PCIntraHopping' class
        container of hopping terms in the primitive cell
        Only half of the hopping terms are stored.
        Conjugate terms are added automatically when constructing
        the Hamiltonian.
    orb_pos: (num_orb, 3) float64 array
        collection of FRACTIONAL positions of all orbitals
    orb_eng: (num_orb,) float64 array
        collection of on-site energies of all orbitals in eV
    hop_ind: (num_hop, 5) int32 array
        collection of indices of all hopping terms
    hop_eng: (num_hop,) complex128 array
        collection of energies of all hopping terms in eV
    time_stamp: dictionary
        time stamps of attributes
        Method 'sync_array' will use this dictionary to update the arrays.
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

        # Setup orbitals and hopping terms
        self.orbital_list = []
        self.hopping_dict = PCIntraHopping()

        # Setup arrays.
        self.orb_pos = None
        self.orb_eng = None
        self.hop_ind = None
        self.hop_eng = None

        # Setup timestamp
        # Orbital list and hopping terms should be newer than the arrays!
        current_time = time.time()
        self.time_stamp = {'orb_list': current_time,
                           'hop_dict': current_time,
                           'orb_array': current_time-0.1,
                           'hop_array': current_time-0.1}

        # Setup misc. attributes.
        self.extended = 1

    def _is_newer(self, key1, key2):
        """
        Check if the timestamp of key1 is newer than key2.

        :param key1: string, key of self.time_stamp
        :param key2: string, key of self.time_stamp
        :return: bool, whether timestamp of key1 is newer than key2
        """
        return self.time_stamp[key1] >= self.time_stamp[key2]

    def _update_time_stamp(self, key):
        """
        Update the timestamp of key.

        :param key: string, key of self.time_stamp
        :return: None
        """
        self.time_stamp[key] = time.time()

    def add_orbital(self, position, energy=0.0, label="X", sync_array=False,
                    **kwargs):
        """
        Add a new orbital to the primitive cell.

        :param position: tuple with 2 or 3 floats
            FRACTIONAL coordinate of the orbital
        :param energy: float
            on-site energy of the orbital in eV
        :param label: string
            orbital label
        :param sync_array: boolean
            whether to call 'sync_array' to update numpy arrays
            according to orbitals and hopping terms
        :param kwargs: dictionary
            arguments for method 'sync_array'
        :return: None
            self.orbital_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        # Check arguments
        if self.is_locked:
            raise exc.PCLockError()
        try:
            position = correct_coord(position)
        except exc.CoordLenError as err:
            raise exc.OrbPositionLenError(err.coord) from err

        # Add the orbital
        self.orbital_list.append(Orbital(position, energy, label))
        self._update_time_stamp('orb_list')
        if sync_array:
            self.sync_array(**kwargs)

    def add_orbital_cart(self, position, unit=consts.ANG, **kwargs):
        """
        Add a new orbital to the primitive cell with Cartesian coordinates.

        :param position: tuple of 2 or 3 floats
            Cartesian coordinate of orbital in arbitrary unit
        :param unit: float
            conversion coefficient from arbitrary unit to NM
        :param kwargs: dictionary
            keyword arguments for 'add_orbital'
        :return: None
            self.orbital_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        position_cart = np.array([correct_coord(position)])
        position_frac = lat.cart2frac(self.lat_vec, position_cart * unit)[0]
        self.add_orbital(position_frac, **kwargs)

    def set_orbital(self, orb_i, position=None, energy=None, label=None,
                    sync_array=False, **kwargs):
        """
        Modify the position, energy and label of an existing orbital.

        If position, energy or label is None, then the corresponding attribute
        will not be modified.

        :param orb_i: integer
            index of the orbital to modify
        :param position: tuple with 2 or 3 floats
            new FRACTIONAL coordinate of the orbital
        :param energy: float
            new on-site energy of the orbital in eV
        :param label: string
            orbital label
        :param sync_array: boolean
            whether to call sync_array to update numpy arrays
            according to orbitals and hopping terms
        :param kwargs: dictionary
            arguments for method 'sync_array'
        :return: None
            self.orbital_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        # Check arguments
        if self.is_locked:
            raise exc.PCLockError()
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
        if label is not None:
            orbital.label = label
        self._update_time_stamp('orb_list')
        if sync_array:
            self.sync_array(**kwargs)

    def set_orbital_cart(self, orb_i, position=None, unit=consts.ANG, **kwargs):
        """
        Modify the position,  energy and label of an existing orbital
        with Cartesian coordinates.

        If position, energy or label is None, then the corresponding attribute
        will not be modified.

        :param orb_i: integer
            index of the orbital to modify
        :param position: tuple of 2 or 3 floats
            Cartesian coordinate of orbital in arbitrary unit
        :param unit: float
            conversion coefficient from arbitrary unit to NM
        :param kwargs: dictionary
            keyword arguments for 'set_orbital'
        :return: None
            self.orbital_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        if position is not None:
            position = np.array([correct_coord(position)])
            position = lat.cart2frac(self.lat_vec, position * unit)[0]
        self.set_orbital(orb_i, position, **kwargs)

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
        :param kwargs: dictionary
            arguments for method 'sync_array'
        :return: None
            self.orbital_list and self.hopping_list are modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        """
        # Check arguments
        if self.is_locked:
            raise exc.PCLockError()

        # Delete the orbital
        try:
            self.orbital_list.pop(orb_i)
        except IndexError as err:
            raise exc.PCOrbIndexError(orb_i) from err

        # Delete associated hopping terms
        hop_dict = self.hopping_dict.dict
        for rn, hop_rn in hop_dict.items():
            for pair in list(hop_rn.keys()):
                if orb_i in pair:
                    hop_rn.pop(pair)
                else:
                    ii, jj = pair
                    if ii > orb_i:
                        ii -= 1
                    if jj > orb_i:
                        jj -= 1
                    new_pair = (ii, jj)
                    if new_pair != pair:
                        energy = hop_rn.pop(pair)
                        hop_rn[new_pair] = energy

        # Update timestamps
        self._update_time_stamp('orb_list')
        self._update_time_stamp('hop_dict')

        # Update arrays
        if sync_array:
            self.sync_array(**kwargs)

    def _check_hop_index(self, rn, orb_i, orb_j):
        """
        Check cell index and orbital pair of hopping term.

        :param rn: (ra, rb, rc)
            cell index of the hopping term, i.e. R
        :param orb_i: integer
            index of orbital i in <i,0|H|j,R>
        :param orb_j: integer
            index of orbital j in <i,0|H|j,R>
        :return: (rn, orb_i, orb_j)
            checked cell index and orbital pair
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        try:
            rn = correct_coord(rn)
        except exc.CoordLenError as err:
            raise exc.CellIndexLenError(err.coord) from err
        num_orbitals = len(self.orbital_list)
        if not (0 <= orb_i < num_orbitals):
            raise exc.PCOrbIndexError(orb_i)
        if not (0 <= orb_j < num_orbitals):
            raise exc.PCOrbIndexError(orb_j)
        if rn == (0, 0, 0) and orb_i == orb_j:
            raise exc.PCHopDiagonalError(rn, orb_i)
        return rn, orb_i, orb_j

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
        :param kwargs: dictionary
            arguments for method 'sync_array'
        :return: None
            self.hopping_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        if self.is_locked:
            raise exc.PCLockError()
        rn, orb_i, orb_j = self._check_hop_index(rn, orb_i, orb_j)
        self.hopping_dict.add_hopping(rn, orb_i, orb_j, energy)
        self._update_time_stamp('hop_dict')
        if sync_array:
            self.sync_array(**kwargs)

    def add_hopping_dict(self, hop_dict: HopDict, eng_cutoff=1e-5,
                         sync_array=False, **kwargs):
        """
        Add a matrix of hopping terms to the primitive cell, or update existing
        hopping terms.

        Reserved for compatibility with old version of TBPLaS.

        :param hop_dict: instance of 'HopDict' class
            hopping dictionary
        :param eng_cutoff: float
            energy cutoff for hopping terms in eV
            Hopping terms with energy below this threshold will be dropped.
        :param sync_array: boolean
            whether to call sync_array to update numpy arrays
            according to orbitals and hopping terms
        :param kwargs: dictionary
            arguments for method 'sync_array'
        :return: None
        :raises PCLockError: if the primitive cell is locked
        """
        for rn, hop_mat in hop_dict.dict.items():
            for orb_i in range(hop_mat.shape[0]):
                for orb_j in range(hop_mat.shape[1]):
                    hop_eng = hop_mat.item(orb_i, orb_j)
                    if abs(hop_eng) >= eng_cutoff:
                        self.add_hopping(rn, orb_i, orb_j, hop_eng,
                                         sync_array=False)
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
        :return: energy: complex
            hopping energy
        :raises PCHopNotFoundError: if rn + (orb_i, orb_j) or its conjugate
            counterpart is not found in the hopping terms
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        rn, orb_i, orb_j = self._check_hop_index(rn, orb_i, orb_j)
        energy, status = self.hopping_dict.get_hopping(rn, orb_i, orb_j)
        if status:
            return energy
        else:
            raise exc.PCHopNotFoundError(rn + (orb_i, orb_j))

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
        :param kwargs: dictionary
            arguments for method 'sync_array'
        :return: None
            self.hopping_list is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises PCHopNotFoundError: if rn + (orb_i, orb_j) or its conjugate
            counterpart is not found in the hopping terms
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        if self.is_locked:
            raise exc.PCLockError()
        rn, orb_i, orb_j = self._check_hop_index(rn, orb_i, orb_j)
        status = self.hopping_dict.remove_hopping(rn, orb_i, orb_j)
        if not status:
            raise exc.PCHopNotFoundError(rn + (orb_i, orb_j))
        self._update_time_stamp('hop_dict')
        if sync_array:
            self.sync_array(**kwargs)

    def trim(self):
        """
        Trim dangling orbitals and associated hopping terms.

        :return: None
            self.orbital_list and self.hopping_list is modified.
        :raises PCLockError: if the primitive cell is locked
        """
        # Count the number of hopping terms of each orbital
        hop_count = np.zeros(self.num_orb, dtype=np.int32)
        for rn, hop_rn in self.hopping_dict.dict.items():
            for pair, energy in hop_rn.items():
                hop_count[pair[0]] += 1
                hop_count[pair[1]] += 1

        # Get indices of orbitals to remove
        orb_id_trim = [i_o for i_o, count in enumerate(hop_count) if count <= 1]

        # Remove orbitals and hopping terms
        # Orbital indices should be sorted in increasing order.
        orb_id_trim = sorted(orb_id_trim)
        for i, orb_id in enumerate(orb_id_trim):
            self.remove_orbital(orb_id - i)
        self.sync_array()

    def apply_pbc(self, pbc=(True, True, True)):
        """
        Apply periodic boundary conditions by removing hopping terms between
        cells along non-periodic direction.

        :param pbc: tuple of 3 booleans
            whether pbc is enabled along 3 directions
        :return: None
            Incoming prim_cell is modified.
        :raises PCLockError: if the primitive cell is locked
        :raises ValueError: if len(pbc) != 3
        """
        if self.is_locked:
            raise exc.PCLockError()
        if len(pbc) != 3:
            raise ValueError("Length of pbc is not 3")

        # Get the list of hopping terms to keep
        hop_dict = self.hopping_dict.dict
        for rn in list(hop_dict.keys()):
            to_keep = True
            for i_dim in range(3):
                if not pbc[i_dim] and rn[i_dim] != 0:
                    to_keep = False
                    break
            if not to_keep:
                hop_dict.pop(rn)
        self._update_time_stamp('hop_dict')
        self.sync_array()

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
        # Update orbital arrays
        if force_sync or self._is_newer('orb_list', 'orb_array'):
            if verbose:
                print("INFO: updating pc orbital arrays")
            self._update_time_stamp('orb_array')
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

        # Update hopping arrays
        if force_sync or self._is_newer('hop_dict', 'hop_array'):
            if verbose:
                print("INFO: updating pc hopping arrays")
            self._update_time_stamp('hop_array')
            hop_ind, hop_eng = self.hopping_dict.to_array()
            # if hop_eng is not [], update as usual.
            if len(hop_eng) != 0:
                self.hop_ind = hop_ind
                self.hop_eng = hop_eng
            # Otherwise, restore to default settings as in __init__.
            else:
                self.hop_ind = None
                self.hop_eng = None
        else:
            if verbose:
                print("INFO: no need to update pc hopping arrays")

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

    def plot(self, fig_name=None, fig_size=None, fig_dpi=300,
             with_orbitals=True, with_cells=True, with_conj=True,
             hop_as_arrows=True, hop_eng_cutoff=1e-5, view="ab"):
        """
        Plot lattice vectors, orbitals, and hopping terms.

        If figure name is give, save the figure to file. Otherwise, show it on
        the screen.

        :param fig_name: string
            file name to which the figure will be saved
        :param fig_size: (width, height)
            size of the figure
        :param fig_dpi: integer
            resolution of the figure file
        :param with_orbitals: boolean
            whether to plot orbitals as filled circles
        :param with_cells: boolean
            whether to plot borders of primitive cells
        :param with_conj: boolean
            whether to plot conjugate hopping terms as well
        :param hop_as_arrows: boolean
            whether to plot hopping terms as arrows
            If true, hopping terms will be plotted as arrows using axes.arrow()
            method. Otherwise, they will be plotted as lines using
            LineCollection. The former is more intuitive but much slower.
        :param hop_eng_cutoff: float
            cutoff for showing hopping terms.
            Hopping terms with absolute energy below this value will not be
            shown in the plot.
        :param view: string
            kind of view point
            should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
        :returns: None
        :raises ValueError: if view is illegal
        """
        self.sync_array()
        fig, axes = plt.subplots(figsize=fig_size)
        axes.set_aspect('equal')
        viewer = ModelViewer(axes, self.lat_vec, view)

        # Restore conjugate hopping terms
        if with_conj:
            hop_ind_conj = np.zeros(self.hop_ind.shape, dtype=np.int32)
            hop_ind_conj[:, :3] = -self.hop_ind[:, :3]
            hop_ind_conj[:, 3] = self.hop_ind[:, 4]
            hop_ind_conj[:, 4] = self.hop_ind[:, 3]
            hop_ind_full = np.vstack((self.hop_ind, hop_ind_conj))
            hop_eng_full = np.vstack((self.hop_eng, self.hop_eng))
        else:
            hop_ind_full = self.hop_ind
            hop_eng_full = self.hop_eng

        # Determine the range of rn
        ra = hop_ind_full[:, 0]
        rb = hop_ind_full[:, 1]
        rc = hop_ind_full[:, 2]
        ra_min, ra_max = ra.min(), ra.max()
        rb_min, rb_max = rb.min(), rb.max()
        rc_min, rc_max = rc.min(), rc.max()

        # Get Cartesian coordinates of orbitals in (0, 0, 0) cell
        pos_r0 = np.matmul(self.orb_pos, self.lat_vec)

        # Plot orbitals
        if with_orbitals:
            for i_a in range(ra_min, ra_max+1):
                for i_b in range(rb_min, rb_max+1):
                    for i_c in range(rc_min, rc_max+1):
                        center = np.matmul((i_a, i_b, i_c), self.lat_vec)
                        pos_rn = pos_r0 + center
                        viewer.scatter(pos_rn, s=100, c=self.orb_eng)

        # Plot cells
        if with_cells:
            if view in ("ab", "ba"):
                viewer.add_grid(ra_min, ra_max + 1, rb_min, rb_max + 1)
            elif view in ("bc", "cb"):
                viewer.add_grid(rb_min, rb_max + 1, rc_min, rc_max + 1)
            else:
                viewer.add_grid(ra_min, ra_max + 1, rc_min, rc_max + 1)
            viewer.plot_grid(color="k", linestyle=":")
            viewer.plot_lat_vec(color="k", length_includes_head=True,
                                width=0.005, head_width=0.02)

        # Plot hopping terms
        for i_h, hop in enumerate(hop_ind_full):
            if abs(hop_eng_full.item(i_h)) >= hop_eng_cutoff:
                center = np.matmul(hop[:3], self.lat_vec)
                pos_i = pos_r0[hop.item(3)]
                pos_j = pos_r0[hop.item(4)] + center
                if hop_as_arrows:
                    viewer.plot_arrow(pos_i, pos_j, color="r",
                                      length_includes_head=True, width=0.002,
                                      head_width=0.02, fill=False)
                else:
                    viewer.add_line(pos_i, pos_j)
        if not hop_as_arrows:
            viewer.plot_line(color='r')

        # Hide spines and ticks.
        for key in ("top", "bottom", "left", "right"):
            axes.spines[key].set_visible(False)
        axes.set_xticks([])
        axes.set_yticks([])
        fig.tight_layout()
        plt.autoscale()
        if fig_name is not None:
            plt.savefig(fig_name, dpi=fig_dpi)
        else:
            plt.show()
        plt.close()

    def print(self):
        """
        Print orbital and hopping information.

        :return: None
        """
        print("Lattice vectors (nm):")
        for vec in self.lat_vec:
            vec_format = "%10.5f%10.5f%10.5f" % (vec[0], vec[1], vec[2])
            print(vec_format)
        print("Orbitals:")
        for orbital in self.orbital_list:
            pos = orbital.position
            pos_fmt = "%10.5f%10.5f%10.5f" % (pos[0], pos[1], pos[2])
            print(pos_fmt, orbital.energy)
        print("Hopping terms:")
        for rn, hop_rn in self.hopping_dict.dict.items():
            for pair, energy in hop_rn.items():
                print(" ", rn, pair, energy)

    def calc_bands(self, k_path: np.ndarray, enable_mpi=False):
        """
        Calculate band structure along given k_path.

        :param k_path: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points along given path
        :param enable_mpi: boolean
            whether to enable parallelization over k-points using mpi
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

        # Distribute k-points over processes
        if enable_mpi:
            from ..parallel import MPIEnv
            mpi_env = MPIEnv()
            k_index = mpi_env.dist_range(num_k_points)
        else:
            mpi_env = None
            k_index = range(num_k_points)

        # Loop over k-points to evaluate the energies
        for i_k in k_index:
            k_point = k_path[i_k]
            ham_k *= 0.0
            core.set_ham(self.orb_pos, self.orb_eng,
                         self.hop_ind, self.hop_eng,
                         k_point, ham_k)
            eigenvalues, eigenstates, info = lapack.zheev(ham_k)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            bands[i_k, :] = eigenvalues[:]

        # Collect energies
        if mpi_env is not None:
            bands = mpi_env.all_reduce(bands)
        return k_len, bands

    def calc_dos(self, k_points: np.ndarray, e_min=None, e_max=None,
                 e_step=0.05, sigma=0.05, basis="Gaussian", enable_mpi=False):
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
        :param enable_mpi: boolean
            whether to enable parallelization over k-points using mpi
        :return: energies: (num_grid,) float64 array
            energy grid corresponding to e_min, e_max and e_step
        :return: dos: (num_grid,) float64 array
            density of states in states/eV
        :raises BasisError: if basis is neither Gaussian nor Lorentzian
        """
        # Get the band energies
        k_len, bands = self.calc_bands(k_points, enable_mpi)

        # Create energy grid
        if e_min is None:
            e_min = np.min(bands)
        if e_max is None:
            e_max = np.max(bands)
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

        # Distribute k-points over processes
        num_kpt = bands.shape[0]
        if enable_mpi:
            from ..parallel import MPIEnv
            mpi_env = MPIEnv()
            k_index = mpi_env.dist_range(num_kpt)
        else:
            mpi_env = None
            k_index = range(num_kpt)

        # Collect contributions
        for i_k in k_index:
            for eng_i in bands[i_k]:
                dos += basis_func(energies, eng_i)
        if mpi_env is not None:
            dos = mpi_env.all_reduce(dos)

        # Re-normalize dos
        # For each energy in bands, we use a normalized Gaussian or Lorentzian
        # basis function to approximate the Delta function. Totally, there are
        # bands.size basis functions. So we divide dos by this number.
        dos /= bands.size
        return energies, dos

    @property
    def num_orb(self):
        """
        Get the number of orbitals in the primitive cell.

        :return: integer, number of orbitals
        """
        return len(self.orbital_list)

    @property
    def orb_pos_nm(self):
        """
        Get the Cartesian coordinates of orbitals in NANOMETER.

        :return: orb_pos_nm: (num_orb, 3) float64 array
            Cartesian coordinates of orbitals in NANOMETER
        """
        self.sync_array()
        orb_pos_nm = lat.frac2cart(self.lat_vec, self.orb_pos)
        return orb_pos_nm

    @property
    def orb_pos_ang(self):
        """
        Get the Cartesian coordinates of orbitals in ANGSTROM.

        :return: orb_pos_ang: (num_orb, 3) float64 array
            Cartesian coordinates of orbitals in arbitrary NANOMETER
        """
        orb_pos_ang = self.orb_pos_nm * 10
        return orb_pos_ang
