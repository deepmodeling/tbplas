"""Functions and classes for manipulating the primitive cell."""

from typing import List, Tuple, Dict, Hashable
import math

import numpy as np
from scipy.sparse import dia_matrix, dok_matrix, csr_matrix
import matplotlib.pyplot as plt

from ..base import constants as consts
from ..base import lattice as lat
from . import exceptions as exc
from . import core
from .base import (check_rn, check_pos, Orbital, Lockable, IntraHopping,
                   HopDict, pair_type, rn_type, rn3_type, pos_type, pos3_type)
from .utils import ModelViewer
from ..diagonal import DiagSolver


__all__ = ["PrimitiveCell"]


class PrimitiveCell(Lockable):
    """
    Class for representing the primitive cell.

    Attributes
    ----------
    _lat_vec: (3, 3) float64 array
        Cartesian coordinates of lattice vectors in NANO METER
        Each ROW corresponds to one lattice vector.
    _origin: (3,) float64 array
        Cartesian coordinates of origin of lattice vectors in NANO METER
    _orbital_list: List[Orbital]
        list of orbitals in the primitive cell
    _hopping_dict: 'IntraHopping' instance
        container of hopping terms in the primitive cell
        Only half of the hopping terms are stored. Conjugate terms are added
        automatically when constructing the Hamiltonian.
    _hash_dict: Dict[str, int]
        hashes of attributes to be used by 'sync_array' to update the arrays
    _orb_pos: (num_orb, 3) float64 array
        FRACTIONAL positions of all orbitals
    _orb_eng: (num_orb,) float64 array
        on-site energies of all orbitals in eV
    _hop_ind: (num_hop, 5) int32 array
        indices of all hopping terms
    _hop_eng: (num_hop,) complex128 array
        energies of all hopping terms in eV
    extended: float
        number of times the primitive cell has been extended
        Some response functions will be divided by this attribute to average
        them to the very primitive cell since the primitive cell itself may
        be formed by replicating another cell.
        In most cases it will be an integer. However, if the primitive cell
        has been created by reshaping another cell, it will become a float.
    """
    def __init__(self, lat_vec: np.ndarray = np.eye(3, dtype=np.float64),
                 origin: np.ndarray = np.zeros(3, dtype=np.float64),
                 unit: float = consts.ANG) -> None:
        """
        :param lat_vec: (3, 3) float64 array
            Cartesian coordinates of lattice vectors in arbitrary unit
        :param origin: (3,) float64 array
            Cartesian coordinate of lattice origin in arbitrary unit
        :param unit: conversion coefficient from arbitrary unit to NM
        :return: None
        :raises LatVecError: if shape of lat_vec is not (3, 3)
        :raise ValueError: if shape of origin is not (3,)
        """
        super().__init__()

        # Setup lattice vectors
        lat_vec = np.array(lat_vec, dtype=np.float64)
        if lat_vec.shape != (3, 3):
            raise exc.LatVecError()
        self._lat_vec = lat_vec * unit

        # Setup lattice origin
        origin = np.array(origin, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError(f"Length of origin is not 3")
        self._origin = origin * unit

        # Setup orbitals and hopping terms
        self._orbital_list = []
        self._hopping_dict = IntraHopping()

        # Setup hash values
        self._hash_dict = {'orb': self._get_hash('orb'),
                           'hop': self._get_hash('hop')}

        # Setup arrays.
        self._orb_pos = None
        self._orb_eng = None
        self._hop_ind = None
        self._hop_eng = None

        # Setup misc. attributes.
        self.extended = 1.0

    def __hash__(self) -> int:
        """
        Return the hash of this instance.

        NOTE: DO NOT use the hashes in self._hash_dict because they may be out
        of date.

        :return: hash of this instance
        """
        return hash((self._get_hash("orb"), self._get_hash("hop")))

    def _get_hash(self, attr: str) -> int:
        """
        Get the hash of given attribute.

        :param attr: name of the attribute
        :return: hash of the attribute
        :raises ValueError: if attr is illegal
        """
        if attr == "orb":
            new_hash = hash(tuple(self._orbital_list))
        elif attr == "hop":
            new_hash = hash(self._hopping_dict)
        else:
            raise ValueError(f"Illegal attribute name {attr}")
        return new_hash

    def _update_hash(self, attr: str) -> bool:
        """
        Compare and update the hash of given attribute.

        :param attr: name of the attribute
        :return: whether the hash has been updated.
        :raises ValueError: if attr is illegal
        """
        new_hash = self._get_hash(attr)
        if self._hash_dict[attr] != new_hash:
            self._hash_dict[attr] = new_hash
            status = True
        else:
            status = False
        return status

    @staticmethod
    def _check_position(position: pos_type) -> pos3_type:
        """
        Check and complete orbital position.

        :param position: orbital position to check
        :return position: completed orbital position
        :raises OrbPositionLenError: if len(position) is not 2 or 3
        """
        position, legal = check_pos(position)
        if not legal:
            raise exc.OrbPositionLenError(position)
        return position

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
        num_orbitals = self.num_orb
        if not (0 <= orb_i < num_orbitals):
            raise exc.PCOrbIndexError(orb_i)
        if not (0 <= orb_j < num_orbitals):
            raise exc.PCOrbIndexError(orb_j)
        if rn == (0, 0, 0) and orb_i == orb_j:
            raise exc.PCHopDiagonalError(rn, orb_i)
        return rn, orb_i, orb_j

    def add_orbital(self, position: pos_type,
                    energy: float = 0.0,
                    label: Hashable = "X") -> None:
        """
        Add a new orbital to the primitive cell.

        :param position: FRACTIONAL coordinate of the orbital
        :param energy: on-site energy of the orbital in eV
        :param label: orbital label
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        # Check arguments
        self.check_lock()
        position = self._check_position(position)

        # Add the orbital
        self._orbital_list.append(Orbital(position, energy, label))

    def add_orbital_cart(self, position: pos_type,
                         unit: float = consts.ANG,
                         **kwargs) -> None:
        """
        Add a new orbital to the primitive cell in CARTESIAN coordinates.

        :param position: Cartesian coordinate of orbital in arbitrary unit
        :param unit: conversion coefficient from arbitrary unit to NM
        :param kwargs: keyword arguments for 'add_orbital'
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        self.check_lock()
        position = np.array([self._check_position(position)]) * unit
        position = lat.cart2frac(self._lat_vec, position, self._origin)[0]
        self.add_orbital(position, **kwargs)

    def set_orbital(self, orb_i: int,
                    position: pos_type = None,
                    energy: float = None,
                    label: Hashable = None) -> None:
        """
        Modify the position, energy and label of an existing orbital.

        If position, energy or label is None, then the corresponding attribute
        will not be modified.

        :param orb_i: index of the orbital to modify
        :param position: new FRACTIONAL coordinate of the orbital
        :param energy: new on-site energy of the orbital in eV
        :param label: new orbital label
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        # Check arguments
        self.check_lock()
        if position is not None:
            position = self._check_position(position)

        # Set orbital attributes
        try:
            orbital = self._orbital_list[orb_i]
        except IndexError as err:
            raise exc.PCOrbIndexError(orb_i) from err
        if position is None:
            position = orbital.position
        if energy is None:
            energy = orbital.energy
        if label is None:
            label = orbital.label
        self._orbital_list[orb_i] = Orbital(position, energy, label)

    def set_orbital_cart(self, orb_i: int,
                         position: pos_type = None,
                         unit: float = consts.ANG,
                         **kwargs) -> None:
        """
        Modify the position, energy and label of an existing orbital with
        Cartesian coordinates.

        If position, energy or label is None, then the corresponding attribute
        will not be modified.

        :param orb_i: index of the orbital to modify
        :param position: Cartesian coordinate of orbital in arbitrary unit
        :param unit: conversion coefficient from arbitrary unit to NM
        :param kwargs: keyword arguments for 'set_orbital'
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        :raises OrbPositionLenError: if len(position) != 2 or 3
        """
        self.check_lock()
        if position is not None:
            position = np.array([self._check_position(position)]) * unit
            position = lat.cart2frac(self._lat_vec, position, self._origin)[0]
        self.set_orbital(orb_i, position, **kwargs)

    def get_orbital(self, orb_i: int) -> Orbital:
        """
        Get given orbital instance.

        :param orb_i: index of the orbital
        :return: the i-th orbital
        :raises PCOrbIndexError: if orb_i falls out of range
        """
        try:
            orbital = self._orbital_list[orb_i]
        except IndexError as err:
            raise exc.PCOrbIndexError(orb_i) from err
        return orbital

    def remove_orbital(self, orb_i: int) -> None:
        """
        Wrapper over 'remove_orbitals' to remove a single orbital.

        :param orb_i: index of the orbital to remove
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        """
        self.check_lock()
        self.remove_orbitals([orb_i])

    def remove_orbitals(self, indices: List[int]) -> None:
        """
        Remove given orbitals and associated hopping terms, then update
        remaining hopping terms.

        :param indices: indices of orbitals to remove
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i falls out of range
        """
        # Check arguments
        self.check_lock()

        # Delete the orbitals
        indices = sorted(indices)
        for i, orb_i in enumerate(indices):
            try:
                self._orbital_list.pop(orb_i - i)
            except IndexError as err:
                raise exc.PCOrbIndexError(orb_i) from err

        # Delete associated hopping terms
        self._hopping_dict.remove_orbitals(indices)

    def add_hopping(self, rn: rn_type,
                    orb_i: int,
                    orb_j: int,
                    energy: complex) -> None:
        """
        Add a new hopping term to the primitive cell, or update an existing
        hopping term.

        :param rn: cell index of the hopping term, i.e. R
        :param orb_i: index of orbital i in <i,0|H|j,R>
        :param orb_j: index of orbital j in <i,0|H|j,R>
        :param energy: hopping integral in eV
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        self.check_lock()
        rn, orb_i, orb_j = self._check_hop_index(rn, orb_i, orb_j)
        self._hopping_dict.add_hopping(rn, orb_i, orb_j, energy)

    def add_hopping_dict(self, hop_dict: HopDict) -> None:
        """
        Add a matrix of hopping terms to the primitive cell, or update existing
        hopping terms.

        Reserved for compatibility with old version of TBPLaS.

        :param hop_dict: hopping dictionary
        :return: None
        :raises LockError: if the primitive cell is locked
        """
        self.check_lock()
        hop_dict.to_spare()
        for rn, hop_mat in hop_dict.hoppings.items():
            for i in range(hop_mat.nnz):
                orb_i = hop_mat.row.item(i)
                orb_j = hop_mat.col.item(i)
                hop_eng = hop_mat.data.item(i)
                self.add_hopping(rn, orb_i, orb_j, hop_eng)

    def get_hopping(self, rn: rn_type,
                    orb_i: int,
                    orb_j: int) -> complex:
        """
        Get given hopping term or its conjugate counterpart.

        :param rn: cell index of the hopping term, i.e. R
        :param orb_i: index of orbital i in <i,0|H|j,R>
        :param orb_j: index of orbital j in <i,0|H|j,R>
        :return: hopping energy in eV
        :raises PCHopNotFoundError: if rn + (orb_i, orb_j) or its conjugate
            counterpart is not found in the hopping terms
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        rn, orb_i, orb_j = self._check_hop_index(rn, orb_i, orb_j)
        energy, status = self._hopping_dict.get_hopping(rn, orb_i, orb_j)
        if status:
            return energy
        else:
            raise exc.PCHopNotFoundError(rn + (orb_i, orb_j))

    def remove_hopping(self, rn: rn_type,
                       orb_i: int,
                       orb_j: int) -> None:
        """
        Remove given hopping term.

        :param rn: cell index of the hopping term, i.e. R
        :param orb_i: index of orbital i in <i,0|H|j,R>
        :param orb_j: index of orbital j in <i,0|H|j,R>
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises PCHopNotFoundError: if rn + (orb_i, orb_j) or its conjugate
            counterpart is not found in the hopping terms
        :raises PCOrbIndexError: if orb_i or orb_j falls out of range
        :raises PCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        self.check_lock()
        rn, orb_i, orb_j = self._check_hop_index(rn, orb_i, orb_j)
        status = self._hopping_dict.remove_hopping(rn, orb_i, orb_j)
        if not status:
            raise exc.PCHopNotFoundError(rn + (orb_i, orb_j))

    def trim(self) -> None:
        """
        Trim dangling orbitals and associated hopping terms.

        :return: None
        :raises LockError: if the primitive cell is locked
        """
        self.check_lock()

        # Count the number of hopping terms of each orbital
        hop_count = np.zeros(self.num_orb, dtype=np.int32)
        for rn, hop_rn in self.hoppings.items():
            for pair, energy in hop_rn.items():
                hop_count[pair[0]] += 1
                hop_count[pair[1]] += 1

        # Get indices of orbitals to remove
        orb_id_trim = [i_o for i_o, count in enumerate(hop_count) if count <= 1]

        # Remove orbitals and hopping terms
        self.remove_orbitals(orb_id_trim)

    def apply_pbc(self, pbc: Tuple[bool, bool, bool] = (True, True, True)) -> None:
        """
        Apply periodic boundary conditions by removing hopping terms between
        cells along non-periodic direction.

        :param pbc: whether pbc is enabled along 3 directions
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises ValueError: if len(pbc) != 3
        """
        self.check_lock()
        if len(pbc) != 3:
            raise ValueError("Length of pbc is not 3")

        # Get the list of hopping terms to keep
        for rn in self._hopping_dict.cell_indices:
            to_keep = True
            for i_dim in range(3):
                if not pbc[i_dim] and rn[i_dim] != 0:
                    to_keep = False
                    break
            if not to_keep:
                self._hopping_dict.remove_rn(rn)

    def reset_lattice(self, lat_vec: np.ndarray = None,
                      origin: np.ndarray = None,
                      unit: float = consts.ANG,
                      fix_orb: bool = True) -> None:
        """
        Reset lattice vectors and origin.

        :param lat_vec: (3, 3) float64 array
            Cartesian coordinates of lattice vectors in arbitrary unit
        :param origin: (3,) float64 array
            Cartesian coordinate of lattice origin in arbitrary unit
        :param unit: conversion coefficient from arbitrary unit to NM
        :param fix_orb: whether to keep Cartesian coordinates of orbitals
            unchanged after resetting lattice
        :return: None
        :raises LockError: if the primitive cell is locked
        :raises LatVecError: if shape of lat_vec is not (3, 3)
        :raise ValueError: if shape of origin is not (3,)
        """
        self.check_lock()

        # Check lattice vectors and convert unit
        if lat_vec is None:
            lat_vec = self._lat_vec
        else:
            lat_vec = np.array(lat_vec, dtype=np.float64)
            if lat_vec.shape != (3, 3):
                raise exc.LatVecError()
            lat_vec = lat_vec * unit

        # Check lattice origin and convert unit
        if origin is None:
            origin = self._origin
        else:
            origin = np.array(origin, dtype=np.float64)
            if origin.shape != (3,):
                raise ValueError(f"Length of origin is not 3")
            origin = origin * unit

        # Update orbital positions if needed
        if fix_orb and self.num_orb > 0:
            orb_pos = lat.cart2frac(lat_vec, self.orb_pos_nm, origin)
            for i, pos in enumerate(orb_pos):
                orbital = self._orbital_list[i]
                self._orbital_list[i] = Orbital(tuple(pos), orbital.energy,
                                                orbital.label)

        # Finally update the attributes
        self._lat_vec = lat_vec
        self._origin = origin

    def sync_orb(self, verbose: bool = False,
                 force_sync: bool = False) -> None:
        """
        Synchronize orb_pos and orb_eng according to the orbitals.

        :param verbose: whether to output additional debugging information
        :param force_sync: whether to force synchronizing the arrays even if the
            orbitals did not change
        :return: None
        """
        to_update = self._update_hash('orb')
        if force_sync or to_update:
            if verbose:
                print("INFO: updating pc orbital arrays")
            # If orbital_list is not [], update as usual.
            if len(self._orbital_list) != 0:
                self._orb_pos = np.array(
                    [orb.position for orb in self._orbital_list], dtype=np.float64)
                self._orb_eng = np.array(
                    [orb.energy for orb in self._orbital_list], dtype=np.float64)
            # Otherwise, restore to default settings as in __init__.
            else:
                self._orb_pos = None
                self._orb_eng = None
        else:
            if verbose:
                print("INFO: no need to update pc orbital arrays")

    def sync_hop(self, verbose: bool = False,
                 force_sync: bool = False):
        """
        Synchronize hop_ind and hop_eng according to the hopping terms.

        :param verbose: whether to output additional debugging information
        :param force_sync: whether to force synchronizing the arrays even if the
            hopping terms did not change
        :return: None
        """
        to_update = self._update_hash('hop')
        if force_sync or to_update:
            if verbose:
                print("INFO: updating pc hopping arrays")
            hop_ind, hop_eng = self._hopping_dict.to_array(use_int64=False)
            # if hop_eng is not [], update as usual.
            if len(hop_eng) != 0:
                self._hop_ind = hop_ind
                self._hop_eng = hop_eng
            # Otherwise, restore to default settings as in __init__.
            else:
                self._hop_ind = None
                self._hop_eng = None
        else:
            if verbose:
                print("INFO: no need to update pc hopping arrays")

    def sync_array(self, **kwargs) -> None:
        """
        Synchronize orb_pos, orb_eng, hop_ind and hop_eng according to orbitals
        and hopping terms.

        :param kwargs: arguments for 'sync_orb' and sync_'hop'
        :return: None
        """
        self.sync_orb(**kwargs)
        self.sync_hop(**kwargs)

    def get_lattice_area(self, direction: str = "c") -> float:
        """
        Get the area formed by lattice vectors normal to given direction.

        :param direction: direction of area, e.g. "c" indicates the area formed
            by lattice vectors in the aOb plane.
        :return: area formed by lattice vectors in NM^2.
        """
        return lat.get_lattice_area(self._lat_vec, direction)

    def get_lattice_volume(self) -> float:
        """
        Get the volume formed by all three lattice vectors in NM^3.

        :return: volume in NM^3.
        """
        return lat.get_lattice_volume(self._lat_vec)

    def get_reciprocal_vectors(self) -> np.ndarray:
        """
        Get the Cartesian coordinates of reciprocal lattice vectors in 1/NM.

        :return: (3, 3) float64 array
            reciprocal vectors in 1/NM.
        """
        return lat.gen_reciprocal_vectors(self._lat_vec)

    def plot(self, fig_name: str = None,
             fig_size: Tuple[float, float] = None,
             fig_dpi: int = 300,
             with_orbitals: bool = True,
             with_cells: bool = True,
             with_conj: bool = True,
             hop_as_arrows: bool = True,
             hop_eng_cutoff: float = 1e-5,
             view: str = "ab") -> None:
        """
        Plot lattice vectors, orbitals, and hopping terms.

        If figure name is given, save the figure to file. Otherwise, show it on
        the screen.

        :param fig_name: file name to which the figure will be saved
        :param fig_size: width and height of the figure
        :param fig_dpi: resolution of the figure file
        :param with_orbitals: whether to plot orbitals as filled circles
        :param with_cells: whether to plot borders of primitive cells
        :param with_conj: whether to plot conjugate hopping terms as well
        :param hop_as_arrows: whether to plot hopping terms as arrows
            If true, hopping terms will be plotted as arrows using axes.arrow()
            method. Otherwise, they will be plotted as lines using
            LineCollection. The former is more intuitive but much slower.
        :param hop_eng_cutoff: cutoff for showing hopping terms
        :param view: kind of view point
        :returns: None
        :raises ValueError: if view is illegal
        """
        self.sync_array()
        fig, axes = plt.subplots(figsize=fig_size)
        axes.set_aspect('equal')
        viewer = ModelViewer(axes, self._lat_vec, self._origin, view)

        # Prepare hopping terms and plot ranges
        if self.num_hop > 0:
            # Restore conjugate hopping terms
            if with_conj:
                hop_ind_conj = np.zeros(self._hop_ind.shape, dtype=np.int32)
                hop_ind_conj[:, :3] = -self._hop_ind[:, :3]
                hop_ind_conj[:, 3] = self._hop_ind[:, 4]
                hop_ind_conj[:, 4] = self._hop_ind[:, 3]
                hop_ind_full = np.vstack((self._hop_ind, hop_ind_conj))
                hop_eng_full = np.vstack((self._hop_eng, self._hop_eng))
            else:
                hop_ind_full = self._hop_ind
                hop_eng_full = self._hop_eng

            # Determine the range of rn
            ra = hop_ind_full[:, 0]
            rb = hop_ind_full[:, 1]
            rc = hop_ind_full[:, 2]
            ra_min, ra_max = ra.min(), ra.max()
            rb_min, rb_max = rb.min(), rb.max()
            rc_min, rc_max = rc.min(), rc.max()
        else:
            hop_ind_full = None
            hop_eng_full = None
            ra_min, ra_max = 0, 0
            rb_min, rb_max = 0, 0
            rc_min, rc_max = 0, 0

        # Plot orbitals
        pos_r0 = self.orb_pos_nm
        if with_orbitals:
            for i_a in range(ra_min, ra_max+1):
                for i_b in range(rb_min, rb_max+1):
                    for i_c in range(rc_min, rc_max+1):
                        center = np.matmul((i_a, i_b, i_c), self._lat_vec)
                        pos_rn = pos_r0 + center
                        viewer.scatter(pos_rn, s=100, c=self._orb_eng)

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
                center = np.matmul(hop[:3], self._lat_vec)
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

    def print(self) -> None:
        """
        Print orbital and hopping information.

        :return: None
        """
        print("Lattice origin (nm):")
        for v in self._origin:
            print("%10.5f" % v, end="")
        print()
        print("Lattice vectors (nm):")
        for vec in self._lat_vec:
            vec_format = "%10.5f%10.5f%10.5f" % (vec[0], vec[1], vec[2])
            print(vec_format)
        print("Orbitals:")
        for orbital in self._orbital_list:
            pos = orbital.position
            pos_fmt = "%10.5f%10.5f%10.5f" % (pos[0], pos[1], pos[2])
            print(pos_fmt, orbital.energy)
        print("Hopping terms:")
        for rn, hop_rn in self.hoppings.items():
            for pair, energy in hop_rn.items():
                print(" ", rn, pair, energy)

    def set_ham_dense(self, k_point: np.ndarray,
                      ham_dense: np.ndarray,
                      convention: int = 1) -> None:
        """
        Set up dense Hamiltonian for given k-point.

        This is the interface to be called by external exact solvers. The
        callers are responsible to call the 'sync_array' method.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param ham_dense: (num_orb, num_orb) complex128 array
            incoming Hamiltonian
        :param convention: convention for setting up the Hamiltonian
        :return: None
        :raises ValueError: if convention not in (1, 2)
        """
        if convention not in (1, 2):
            raise ValueError(f"Illegal convention {convention}")
        ham_dense *= 0.0
        core.set_ham(self._orb_pos, self._orb_eng, self._hop_ind, self._hop_eng,
                     convention, k_point, ham_dense)

    def set_ham_csr(self, k_point: np.ndarray,
                    convention: int = 1) -> csr_matrix:
        """
        Set up sparse Hamiltonian in csr format for given k-point.

        This is the interface to be called by external exact solvers. The
        callers are responsible to call the 'sync_array' method.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param convention: convention for setting up the Hamiltonian
        :return: sparse Hamiltonian
        :raises ValueError: if convention not in (1, 2)
        """
        if convention not in (1, 2):
            raise ValueError(f"Illegal convention {convention}")

        # Diagonal terms
        ham_shape = (self.num_orb, self.num_orb)
        ham_dia = dia_matrix((self._orb_eng, 0), shape=ham_shape)

        # Off-diagonal terms
        ham_half = dok_matrix(ham_shape, dtype=np.complex128)
        for rn, hop_rn in self.hoppings.items():
            for pair, energy in hop_rn.items():
                ii, jj = pair
                if convention == 1:
                    dr = self._orb_pos[jj] - self._orb_pos[ii] + rn
                else:
                    dr = rn
                phase = 2 * math.pi * np.dot(k_point, dr).item()
                factor = math.cos(phase) + 1j * math.sin(phase)
                ham_half[ii, jj] += factor * energy

        # Sum up the terms
        ham_dok = ham_dia + ham_half + ham_half.getH()
        ham_csr = ham_dok.tocsr()
        return ham_csr

    def calc_bands(self, k_points: np.ndarray,
                   enable_mpi: bool = False,
                   echo_details: bool = True,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate band structure along given k_path.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of the k-points
        :param enable_mpi: whether to enable parallelization over k-points
            using mpi
        :param echo_details: whether to output parallelization details
        :param kwargs: arguments for 'calc_bands' of 'DiagSolver' class
        :return: (k_len, bands)
            k_len: (num_kpt,) float64 array in 1/NM
            length of k-path in reciprocal space, for plotting band structure
            bands: (num_kpt, num_orb) float64 array
            Energies corresponding to k-points in eV
        """
        diag_solver = DiagSolver(self, enable_mpi=enable_mpi,
                                 echo_details=echo_details)
        k_len, bands = diag_solver.calc_bands(k_points, **kwargs)[:2]
        return k_len, bands

    def calc_dos(self, k_points: np.ndarray,
                 enable_mpi: bool = False,
                 echo_details: bool = True,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate density of states for given energy range and step.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of the k-points
        :param enable_mpi: whether to enable parallelization over k-points
            using mpi
        :param echo_details: whether to output parallelization details
        :param kwargs: arguments for 'calc_dos' of 'DiagSolver' class
        :return: (energies, dos)
            energies: (num_grid,) float64 array
            energy grid corresponding to e_min, e_max and e_step
            dos: (num_grid,) float64 array
            density of states in states/eV
        """
        diag_solver = DiagSolver(self, enable_mpi=enable_mpi,
                                 echo_details=echo_details)
        energies, dos = diag_solver.calc_dos(k_points, **kwargs)
        return energies, dos

    @property
    def lat_vec(self) -> np.ndarray:
        """
        Interface for the lattice vectors.

        :return: (3, 3) float64 array
            Cartesian coordinates of lattice vectors in NM
        """
        return self._lat_vec

    @property
    def origin(self) -> np.ndarray:
        """
        Interface for the origin.

        :return: (3,) float64 array
            Cartesian coordinate of origin in NM
        """
        return self._origin

    @property
    def orbitals(self) -> List[Orbital]:
        """
        Interface for the '_orbital_list' attribute.

        :return: list of orbitals
        """
        return self._orbital_list

    @property
    def hoppings(self) -> Dict[rn3_type, Dict[pair_type, complex]]:
        """
        Interface for the hopping terms.

        :return: hopping terms
        """
        return self._hopping_dict.hoppings

    @property
    def num_orb(self) -> int:
        """
        Get the number of orbitals in the primitive cell.

        :return: number of orbitals
        """
        return len(self._orbital_list)

    @property
    def num_hop(self) -> int:
        """
        Get the number of hopping terms WITHOUT considering conjugate relation.

        :return: number of hopping terms
        """
        return self._hopping_dict.num_hop

    @property
    def orb_pos(self) -> np.ndarray:
        """
        Interface for the '_orb_pos' attribute.

        :return: (num_orb, 3) float64 array
            fractional coordinates or orbitals
        """
        self.sync_orb()
        return self._orb_pos

    @property
    def orb_eng(self) -> np.ndarray:
        """
        Interface for the '_orb_eng' attribute.

        :return: (num_orb,) float64 array
            on-site energies in eV
        """
        self.sync_orb()
        return self._orb_eng

    @property
    def hop_ind(self) -> np.ndarray:
        """
        Interface for the '_hop_ind' attribute.

        :return: (num_hop, 5) int32 array
            hopping indices
        """
        self.sync_hop()
        return self._hop_ind

    @property
    def hop_eng(self) -> np.ndarray:
        """
        Interface for the '_hop_eng' attribute.

        :return: (num_hop,) complex128 array
            hopping energies
        """
        self.sync_hop()
        return self._hop_eng

    @property
    def orb_pos_nm(self) -> np.ndarray:
        """
        Get the Cartesian coordinates of orbitals in NANOMETER.

        :return: (num_orb, 3) float64 array
            Cartesian coordinates of orbitals in NANOMETER
        """
        self.sync_orb()
        orb_pos_nm = lat.frac2cart(self._lat_vec, self._orb_pos, self._origin)
        return orb_pos_nm

    @property
    def orb_pos_ang(self) -> np.ndarray:
        """
        Get the Cartesian coordinates of orbitals in ANGSTROM.

        :return: (num_orb, 3) float64 array
            Cartesian coordinates of orbitals in arbitrary NANOMETER
        """
        orb_pos_ang = self.orb_pos_nm * 10
        return orb_pos_ang

    @property
    def hop_dr(self) -> np.ndarray:
        """
        Get the hopping distances in FRACTIONAL coordinates.

        :return: (num_hop, 3) float64 array
            hopping distances in FRACTIONAL coordinates
        """
        self.sync_hop()
        hop_dr = np.zeros((self.num_hop, 3), dtype=np.float64)
        for i_h, ind in enumerate(self._hop_ind):
            orb_i, orb_j = ind.item(3), ind.item(4)
            hop_dr[i_h] = self._orb_pos[orb_j] + ind[0:3] - self._orb_pos[orb_i]
        return hop_dr

    @property
    def hop_dr_nm(self) -> np.ndarray:
        """
        Get the hopping distances in NM.

        :return: (num_hop, 3) float64 array
            hopping distances in NM
        """
        return lat.frac2cart(self._lat_vec, self.hop_dr)

    @property
    def hop_dr_ang(self) -> np.ndarray:
        """
        Get the hopping distances in ANGSTROM.

        :return: (num_hop, 3) float64 array
            hopping distances in ANGSTROM
        """
        return self.hop_dr_nm * 10
