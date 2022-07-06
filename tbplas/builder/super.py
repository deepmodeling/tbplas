"""
Functions and classes for super cell.

Functions
---------
    None.

Classes
-------
    OrbitalSet: developer class
        container class for orbitals and vacancies in the super cell
    IntraHopping: user class
        container class for modifications to hopping terms in the super cell
    SuperCell: user class
        abstraction for a super cell from which the sample is constructed
"""

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from . import exceptions as exc
from . import core
from . import lattice as lat
from .base import correct_coord, LockableObject
from .primitive import PrimitiveCell
from .utils import ModelViewer


class OrbitalSet(LockableObject):
    """
    Container class for orbitals and vacancies in the super cell.

    Attributes
    ----------
    prim_cell: instance of 'PrimitiveCell' class
        primitive cell from which the super cell is constructed
    dim: (3,) int32 array
        dimension of the super cell along a, b, and c directions
    pbc: (3,) int32 array
        whether to enable periodic condition along a, b, and c directions
        0 for False, 1 for True.
    vacancy_list: list of (ia, ib, ic, io)
        indices of vacancies in primitive cell representation
        None if there are no vacancies.
    hash_dict: dictionary
        dictionary of hash of tuple(vacancy_list)
        Key should be 'vac'.
        Method 'sync_array' will use this flag to update the arrays.
        Should only be accessed within that method!
    vac_id_pc: (num_vac, 4) int32 array
        indices of vacancies in primitive cell representation
    vac_id_sc: (num_vac,) int64 array
        indices of vacancies in super cell representation
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals in primitive cell representation

    NOTES
    -----
    1. Minimal super cell dimension

    Assume that we have a primitive cell located at R=0. The furthest primitive
    cell between which hopping terms exist is located at R=N. It can be proved
    that if the dimension of super cell along that direction is less than N,
    the same matrix element hij will appear more than one time in hop_i, hop_j
    and hop_v of 'SuperCell' class, as well as its conjugate counterpart. This
    will complicate the program and significantly slow it down, which situation
    we must avoid.

    Further, if the dimension of super cell falls in [N, 2*N], hij will appear
    only one time, yet its conjugate counterpart still exists. Although no
    problems have been found so far, we still need to avoid this situation.

    So the minimal dimension of super cell is 2*N+1, where N is the index of
    the furthest primitive cell between which hopping terms exists. Otherwise,
    the 'SuperCell' class, as well as the core functions of '_get_num_hop_sc',
    'build_hop', 'build_hop_k' and 'fill_ham' will not work properly.

    In the hr.dat file produced by Wannier90, there is an N_min and an N_max
    for the furthest primitive cell index. In that case, N should be the
    maximum of |N_max| and |N_min| as the result of translational symmetry.

    2. Why not orb_id_sc

    It's unnecessary to have the orb_id_sc array, as it can be generated from
    orb_id_pc on-the-fly. Actually, the vac_id_sc array is also unnecessary,
    as it can also be generated from vac_id_pc. We keep it just to accelerate
    some operations. For orb_id_sc, there is no such need, and we do not keep
    it for reduce memory usage.

    However, it should be noted that vac_id_sc and orb_id_sc are generated via
    different approaches. We show it by an example of 2*2 super cell with 2
    orbitals per primitive cell. The indices of orbitals as well as vacancies
    in primitive cell representation are
               id_pc    id_sc    type
        (0, 0, 0, 0)        0     orb
        (0, 0, 0, 1)        1     vac
        (0, 1, 0, 0)        2     vac
        (0, 1, 0, 1)        3     orb
        (1, 0, 0, 0)        4     orb
        (1, 0, 0, 1)        5     vac
        (1, 0, 0, 0)        6     orb
        (1, 0, 0, 1)        7     orb

    The indices for vacancies in sc representation are the original id_sc, i.e.
    1, 2, and 5. However, the indices for orbitals are re-ordered to be
               id_pc    id_sc    type
        (0, 0, 0, 0)        0     orb
        (0, 1, 0, 1)        1     orb
        (1, 0, 0, 0)        2     orb
        (1, 0, 0, 0)        3     orb
        (1, 0, 0, 1)        4     orb
    In the core module, indices of vacancies are generated by _id_pc2sc while
    indices of orbitals are generated by _id_pc2sc_vac. The latter excludes
    vacancies when generating the indices.
    """
    def __init__(self, prim_cell: PrimitiveCell, dim, pbc=(False, False, False),
                 vacancies=None) -> None:
        """
        :param prim_cell: instance of 'PrimitiveCell'
            primitive cell from which the super cell is constructed
        :param dim: (na, nb, nc)
            dimension of the super cell along a, b and c directions
        :param pbc: tuple consisting of 3 booleans
            whether to enable periodic boundary condition along a, b, and c
            directions
        :param vacancies: list of (ia, ib, ic, io) or equivalent int32 arrays
            list of indices of vacancies in primitive cell representation
        :raises SCDimLenError: if len(dim) != 2 or 3
        :raises SCDimSizeError: if dimension is smaller than minimal value
        :raises PBCLenError: if len(pbc) != 2 or 3
        :raises VacIDPCLenError: if any vacancy does not have right length
        :raises VacIDPCIndexError: if cell or orbital index of any vacancy is
            out of range
        """
        super().__init__()

        # Synchronize and lock primitive cell
        self.prim_cell = prim_cell
        self.prim_cell.sync_array()
        self.prim_cell.lock()

        # Check and set dimension
        try:
            dim = correct_coord(dim, complete_item=1)
        except exc.CoordLenError as err:
            raise exc.SCDimLenError(dim) from err
        for i in range(3):
            rn_min = self.prim_cell.hop_ind[:, i].min()
            rn_max = self.prim_cell.hop_ind[:, i].max()
            dim_min = max(abs(rn_min), abs(rn_max))
            if dim[i] < dim_min:
                raise exc.SCDimSizeError(i, dim_min)
        self.dim = np.array(dim, dtype=np.int32)

        # Check and set periodic boundary condition
        try:
            pbc = correct_coord([0 if _ is False else 1 for _ in pbc])
        except exc.CoordLenError as err:
            raise exc.PBCLenError(err.coord) from err
        self.pbc = np.array(pbc, dtype=np.int32)

        # Initialize lists and arrays assuming no vacancies
        self.vacancy_list = []
        self.hash_dict = {'vac': hash(tuple(self.vacancy_list))}
        self.vac_id_pc = None
        self.vac_id_sc = None
        self.orb_id_pc = core.build_orb_id_pc(self.dim, self.num_orb_pc,
                                              self.vac_id_pc)

        # Set vacancies if any
        if vacancies is not None:
            self.set_vacancies(vacancies)

    def check_id_pc(self, id_pc):
        """
        Checks if orbital or vacancy index in primitive cell representation
        is legal.

        A legal id_pc should have cell indices falling within [0, dim) and
        orbital index falling within [0, num_orb_pc).

        :param id_pc: (ia, ib, ic, io) or equivalent int32 array
            orbital or vacancy index in primitive cell representation
        :return: None
        :raises IDPCLenError: if len(id_pc) != 4
        :raises IDPCIndexError: if cell or orbital index of id_pc is
            out of range
        :raises IDPCTypeError: if id_pc is not tuple or numpy array
        """
        if len(id_pc) != 4:
            raise exc.IDPCLenError(id_pc)
        if isinstance(id_pc, tuple):
            for i in range(3):
                if id_pc[i] not in range(self.dim.item(i)):
                    raise exc.IDPCIndexError(i, id_pc)
            if id_pc[3] not in range(self.num_orb_pc):
                raise exc.IDPCIndexError(3, id_pc)
        elif isinstance(id_pc, np.ndarray):
            for i in range(3):
                if id_pc.item(i) not in range(self.dim.item(i)):
                    raise exc.IDPCIndexError(i, id_pc)
            if id_pc.item(3) not in range(self.num_orb_pc):
                raise exc.IDPCIndexError(3, id_pc)
        else:
            raise exc.IDPCTypeError(id_pc)

    def add_vacancy(self, vacancy, sync_array=False, **kwargs):
        """
        Add a vacancy to existing list of vacancies.

        :param vacancy: (ia, ib, ic, io) or equivalent int32 array
            vacancy index in primitive cell representation
        :param sync_array: boolean
            whether to call 'sync_array' to update the arrays
        :param kwargs: dictionary
            arguments for method 'sync_array'
        :return: None
            self.vacancy_list is modified.
            self.vac_id_pc, self.vac_id_sc and self.orb_id_pc will also
            be modified if sync_array is True.
        :raises OrbSetLockError: if the object is locked
        :raises VacIDPCLenError: if length of vacancy index is not 4
        :raises VacIDPCIndexError: if cell or orbital index of vacancy is
            out of range
        """
        if self.is_locked:
            raise exc.OrbSetLockError()

        # Convert and check vacancy
        if not isinstance(vacancy, tuple):
            vacancy = tuple(vacancy)
        try:
            self.check_id_pc(vacancy)
        except exc.IDPCLenError as err:
            raise exc.VacIDPCLenError(err.id_pc) from err
        except exc.IDPCIndexError as err:
            raise exc.VacIDPCIndexError(err.i_dim, err.id_pc) from err

        # Add vacancy
        if vacancy not in self.vacancy_list:
            self.vacancy_list.append(vacancy)
            if sync_array:
                self.sync_array(**kwargs)

    def set_vacancies(self, vacancies=None, sync_array=True, **kwargs):
        """
        Reset the list of vacancies to given ones.

        :param vacancies: list of (ia, ib, ic, io) or equivalent int32 arrays
            list of indices of vacancies in primitive cell representation
        :param sync_array: boolean
            whether to call 'sync_array' to update the arrays
        :param kwargs: dictionary
            arguments for method 'sync_array'
        :return: None
            self.vacancy_list is modified.
            self.vac_id_pc, self.vac_id_sc and self.orb_id_pc will also
            be modified if sync_array is True.
        :raises OrbSetLockError: if the object is locked
        :raises VacIDPCLenError: if length of vacancy index is not 4
        :raises VacIDPCIndexError: if cell or orbital index of vacancy is
            out of range
        """
        self.vacancy_list = []
        for vacancy in vacancies:
            self.add_vacancy(vacancy)
        if sync_array:
            self.sync_array(**kwargs)

    def sync_array(self, verbose=False, force_sync=False):
        """
        Synchronize vac_id_pc, vac_id_sc and orb_id_pc according to
        vacancy_list.

        NOTE: The core function '_id_pc2sc_vac' requires vac_id_sc to be sorted
        in increasing order. Otherwise, it won't work properly! So we must sort
        it here. We also re-order vac_id_pc accordingly to avoid potential bugs.

        :param verbose: boolean
            whether to output additional debugging information
        :param force_sync: boolean
            whether to force synchronizing the arrays even if vacancy_list did
            not change
        :return: None
            self.vac_id_pc, self.vac_id_sc and self.orb_id_pc are modified.
        """
        new_hash = hash(tuple(self.vacancy_list))
        if force_sync or new_hash != self.hash_dict['vac']:
            if verbose:
                print("INFO: updating sc vacancy and orbital arrays")
            self.hash_dict['vac'] = new_hash
            # If vacancy list is not [], update arrays as usual.
            if len(self.vacancy_list) != 0:
                vac_id_pc = np.array(self.vacancy_list, dtype=np.int32)
                vac_id_sc = core.build_vac_id_sc(self.dim, self.num_orb_pc,
                                                 vac_id_pc)
                sorted_idx = np.argsort(vac_id_sc, axis=0)
                self.vac_id_pc = vac_id_pc[sorted_idx]
                self.vac_id_sc = vac_id_sc[sorted_idx]
                self.orb_id_pc = core.build_orb_id_pc(self.dim, self.num_orb_pc,
                                                      self.vac_id_pc)
            # Otherwise, restore to default settings as in __ini__.
            else:
                self.vac_id_pc = None
                self.vac_id_sc = None
                self.orb_id_pc = core.build_orb_id_pc(self.dim, self.num_orb_pc,
                                                      self.vac_id_pc)
        else:
            if verbose:
                print("INFO: no need to update sc vacancy and orbital arrays")

    def orb_id_sc2pc(self, id_sc):
        """
        Convert orbital (NOT VACANCY) index from sc representation to pc
        representation.

        NOTE: This method is safe, but EXTREMELY SLOW. If you are going to
        call this method many times, use orb_id_sc2pc_array instead.

        :param id_sc: integer
            index of orbital in super cell representation
        :return: id_pc: (4,) int32 array
            index of orbital in primitive cell representation
        :raises IDSCIndexError: if id_sc is out of range
        """
        self.sync_array()
        try:
            id_pc = self.orb_id_pc[id_sc]
        except IndexError as err:
            raise exc.IDSCIndexError(id_sc) from err
        return id_pc

    def orb_id_pc2sc(self, id_pc):
        """
        Convert orbital (NOT VACANCY) index from pc representation to sc
        representation.

        NOTE: This method is safe, but EXTREMELY SLOW. If you are going to
        call this method many times, use orb_id_pc2sc_array instead.

        :param id_pc: (ia, ib, ic, io), or equivalent int32 array
            index of orbital in primitive cell representation
        :return: id_sc: integer
            index of orbital in super cell representation
        :raises IDPCLenError: if len(id_pc) != 4
        :raises IDPCIndexError: if cell or orbital index of id_pc is
            out of range
        :raises IDPCTypeError: if id_pc is not tuple or numpy array
        :raises IDPCVacError: if id_pc corresponds to a vacancy
        """
        self.sync_array()
        self.check_id_pc(id_pc)
        if not isinstance(id_pc, np.ndarray):
            id_pc = np.array(id_pc, dtype=np.int32)
        orb_id_sc = core.id_pc2sc(self.dim, self.num_orb_pc,
                                  id_pc, self.vac_id_sc)
        if orb_id_sc == -1:
            raise exc.IDPCVacError(id_pc)
        return orb_id_sc

    def orb_id_sc2pc_array(self, id_sc_array):
        """
        Convert an array of orbital (NOT VACANCY) indices from sc
        representation to pc representation.

        :param id_sc_array: (num_orb,) int64 array
            orbital indices in super cell representation
        :return: id_pc_array: (num_orb, 4) int32 array
            orbital indices in primitive cell representation
        :raises IDSCIndexError: if any id_sc in id_sc_array is out of range
        """
        self.sync_array()
        status = core.check_id_sc_array(self.num_orb_sc, id_sc_array)
        if status[0] == -1:
            raise exc.IDSCIndexError(id_sc_array[status[1]])
        id_pc_array = core.id_sc2pc_array(self.orb_id_pc, id_sc_array)
        return id_pc_array

    def orb_id_pc2sc_array(self, id_pc_array):
        """
        Convert an array of orbital (NOT VACANCY) indices from pc
        representation to sc representation.

        :param id_pc_array: (num_orb, 4) int32 array
            orbital indices in primitive cell representation
        :return: id_sc_array: (num_orb,) int64 array
            orbital indices in super cell representation
        :raises IDPCIndexError: if any id_pc in id_pc_array is out of range
        :raises IDPCVacError: if any id_pc in id_pc_array is a vacancy
        """
        self.sync_array()
        status = core.check_id_pc_array(self.dim, self.num_orb_pc,
                                        id_pc_array, self.vac_id_pc)
        if status[0] == -2:
            raise exc.IDPCIndexError(status[2], id_pc_array[status[1]])
        if status[0] == -1:
            raise exc.IDPCVacError(id_pc_array[status[1]])
        id_sc_array = core.id_pc2sc_array(self.dim, self.num_orb_pc,
                                          id_pc_array, self.vac_id_sc)
        return id_sc_array

    def wrap_id_pc_array(self, id_pc_array):
        """
        Wrap orbital or vacancy indices im primitive cell representation
        according to boundary conditions.

        :param id_pc_array: (num_orb, 4) int32 array
            orbital indices in primitive cell representation
        :return: None
            Incoming id_pc_array is modified.
        """
        core.wrap_id_pc_array(id_pc_array, self.dim, self.pbc)

    @property
    def num_orb_pc(self):
        """
        Get the number of orbitals of primitive cell.

        :return: integer
            number of orbitals in primitive cell.
        """
        return self.prim_cell.num_orb

    @property
    def num_orb_sc(self):
        """
        Get the number of orbitals of super cell.

        :return: integer
            number of orbitals in super cell
        """
        num_orb_sc = self.num_orb_pc * np.prod(self.dim).item()
        num_orb_sc -= len(self.vacancy_list)
        return num_orb_sc


class IntraHopping(LockableObject):
    """
    Container class for modifications to hopping terms in the super cell.

    Attributes
    ----------
    indices: list of ((ia, ib, ic, io), (ia', ib', ic', io'))
        where (ia, ib, ic, io) is the index of bra in primitive cell
        representation and (ia', ib', ic', io') is the index of ket
    energies: list of complex numbers
        hopping energies corresponding to indices in eV

    NOTES
    -----
    1. Sanity check

    This class is intended to constitute the 'hop_modifier' attribute of the
    'SuperCell' class and is closely coupled to it. So we do not check if the
    hopping indices are out of range here. This will be done in the 'SuperCell'
    class via the 'get_hop' method.

    2. Reduction

    We reduce hopping terms according to the conjugate relation
        <bra|H|ket> = <ket|H|bra>*.
    So actually only half of hopping terms are stored.

    3. Rules

    If the hopping terms claimed here are already included in 'SuperCell', they
    will overwrite the existing terms. If the hopping terms or their conjugate
    counterparts are new to 'SuperCell', they will be appended to hop_* arrays.
    The dr array will also be updated accordingly.

    We restrict hopping terms to reside within the (0, 0, 0) super cell even if
    periodic conditions are enabled. Other hopping terms will be treated as
    illegal.
    """
    def __init__(self):
        super().__init__()
        self.indices = []
        self.energies = []

    def __find_equiv_hopping(self, bra, ket):
        """
        Find the index of equivalent hopping term of <bra|H|ket>, i.e. the same
        term or its conjugate counterpart.

        :param bra: (ia, ib, ic, io)
            index of bra of the hopping term in primitive cell representation
        :param ket: (ia', ib', ic', io')
            index of ket of the hopping term in primitive cell representation
        :return: id_same: integer
            index of the same hopping term, none if not found
        :return: id_conj: integer
            index of the conjugate hopping term, none if not found
        """
        assert len(bra) == len(ket) == 4
        id_same, id_conj = None, None
        hop_same = (bra, ket)
        hop_conj = (ket, bra)
        try:
            id_same = self.indices.index(hop_same)
        except ValueError:
            pass
        if id_same is None:
            try:
                id_conj = self.indices.index(hop_conj)
            except ValueError:
                pass
        return id_same, id_conj

    def add_hopping(self, rn_i, orb_i, rn_j, orb_j, energy=0.0):
        """
        Add a hopping term.

        :param rn_i: (ia, ib, ic)
            cell index of bra in primitive cell representation
        :param orb_i: integer
            orbital index of bra in primitive cell representation
        :param rn_j: (ia', ib', ic')
            cell index of ket in primitive cell representation
        :param orb_j: integer
            orbital index of ket in primitive cell representation
        :param energy: complex
            hopping energy in eV
        :return: None
            self.indices and self.energies are modified.
        :raises IntraHopLockError: if the object is locked
        :raises IDPCLenError: if len(rn_i) or len(rn_j) not in (2, 3)
        :raises SCHopDiagonalError: if rn_i + (orb_i,) == rn_j + (orb_j,)
        """
        if self.is_locked:
            raise exc.IntraHopLockError()

        # Assemble and check index of bra
        try:
            rn_i = correct_coord(rn_i)
        except exc.CoordLenError as err:
            raise exc.IDPCLenError(rn_i + (orb_i,)) from err
        bra = rn_i + (orb_i,)

        # Assemble and check index of ket
        try:
            rn_j = correct_coord(rn_j)
        except exc.CoordLenError as err:
            raise exc.IDPCLenError(rn_j + (orb_j,)) from err
        ket = rn_j + (orb_j,)

        # Check if the hopping term is a diagonal term
        if bra == ket:
            raise exc.SCHopDiagonalError(bra, ket)

        # Update existing hopping term, or add the new term to hopping_list.
        id_same, id_conj = self.__find_equiv_hopping(bra, ket)
        if id_same is not None:
            self.energies[id_same] = energy
        elif id_conj is not None:
            self.energies[id_conj] = energy.conjugate()
        else:
            self.indices.append((bra, ket))
            self.energies.append(energy)

    def get_orb_ind(self):
        """
        Get orbital indices of bra and ket in hopping terms in primitive cell
        presentation.

        :return: id_bra: (num_hop,) int32 array
            indices of bra
        :return id_ket: (num_hop,) int32 array
            indices of ket
        """
        id_bra = np.array([_[0] for _ in self.indices], dtype=np.int32)
        id_ket = np.array([_[1] for _ in self.indices], dtype=np.int32)
        return id_bra, id_ket

    def trim(self, orb_id_trim):
        """
        Remove hopping terms associated to dangling orbitals.

        This method is intended to be called by 'trim' method of 'SuperCell'
        class.

        :param orb_id_trim: (num_orb_trim, 4) int32 array
            indices of orbitals to trim in primitive cell representation
        :return: None.
            self.energies and self.indices are modified.
        :raises IntraHopLockError: if the object is locked
        """
        if self.is_locked:
            raise exc.IntraHopLockError()

        remain_terms = []
        orb_id_trim = [tuple(orb_id) for orb_id in orb_id_trim]
        for i, ind in enumerate(self.indices):
            to_trim = False
            for orb_id in orb_id_trim:
                if orb_id == ind[0] or orb_id == ind[1]:
                    to_trim = True
                    break
            if not to_trim:
                remain_terms.append(i)
        new_indices = [self.indices[i] for i in remain_terms]
        new_energies = [self.energies[i] for i in remain_terms]
        self.indices = new_indices
        self.energies = new_energies


class SuperCell(OrbitalSet):
    """
    Class for representing a super cell from which the sample is constructed.

    Attributes
    ----------
    hop_modifier: instance of 'IntraHopping' class
        modification to hopping terms in the super cell
    orb_pos_modifier: function
        modification to orbital positions in the super cell
    """
    def __init__(self, prim_cell: PrimitiveCell, dim, pbc=(False, False, False),
                 vacancies=None, hop_modifier: IntraHopping = None,
                 orb_pos_modifier: Callable[[np.ndarray], None] = None) -> None:
        """
        :param prim_cell: instance of 'PrimitiveCell' class
            primitive cell from which the super cell is constructed
        :param dim: (na, nb, nc)
            dimension of the super cell along a, b, and c directions
        :param pbc: tuple consisting of 3 booleans
            whether to enable periodic boundary condition along a, b and c
            directions
        :param vacancies: list of (ia, ib, ic, io)
            indices of vacancies in primitive cell representation
        :param hop_modifier: instance of 'IntraHopping' class
            modification to hopping terms
        :param orb_pos_modifier: function
            modification to orbital positions in the super cell
        :return: None
        :raises SCDimLenError: if len(dim) != 2 or 3
        :raises SCDimSizeError: if dimension is smaller than minimal value
        :raises PBCLenError: if len(pbc) != 2 or 3
        :raises VacIDPCLenError: if any vacancy does not have right length
        :raises VacIDPCIndexError: if cell or orbital index of any vacancy is
            out of range
        """
        # Build and lock orbital set
        super().__init__(prim_cell, dim, pbc=pbc, vacancies=vacancies)
        self.lock()

        # Assign and Lock hop_modifier
        self.hop_modifier = hop_modifier
        if self.hop_modifier is not None:
            self.hop_modifier.lock()

        # Assign orb_pos_modifier
        self.orb_pos_modifier = orb_pos_modifier

    def set_hop_modifier(self, hop_modifier: IntraHopping = None):
        """
        Reset hop_modifier.

        :param hop_modifier: instance of 'IntraHopping' class
            hopping modifier
        :return: None
        """
        # Release old hop_modifier
        if self.hop_modifier is not None:
            self.hop_modifier.unlock()

        # Assign and lock new hop_modifier
        self.hop_modifier = hop_modifier
        if self.hop_modifier is not None:
            self.hop_modifier.lock()

    def set_orb_pos_modifier(self, orb_pos_modifier: Callable = None):
        """
        Reset orb_pos_modifier.

        :param orb_pos_modifier: function
            modifier to orbital positions
        :return: None
        """
        self.orb_pos_modifier = orb_pos_modifier

    def get_orb_eng(self):
        """
        Get energies of all orbitals in the super cell.

        :return: orb_eng: (num_orb_sc,) float64 array
            on-site energies of orbitals in the super cell in eV
        """
        self.sync_array()
        return core.build_orb_eng(self.pc_orb_eng, self.orb_id_pc)

    def get_orb_pos(self):
        """
        Get positions of all orbitals in the super cell.

        :return: orb_pos: (num_orb_sc, 3) float64 array
            Cartesian coordinates of orbitals in the super cell in nm
        """
        self.sync_array()
        orb_pos = core.build_orb_pos(self.pc_lat_vec, self.pc_orb_pos,
                                     self.orb_id_pc)
        if self.orb_pos_modifier is not None:
            self.orb_pos_modifier(orb_pos)
        return orb_pos

    def get_hop(self):
        """
        Get indices and energies of all hopping terms in the super cell.

        NOTE: The hopping terms will be reduced by conjugate relation.
        So only half of them will be returned as results.

        :return: hop_i: (num_hop_sc,) int64 array
            row indices of hopping terms reduced by conjugate relation
        :return: hop_j: (num_hop_sc,) int64 array
            column indices of hopping terms reduced by conjugate relation
        :return: hop_v: (num_hop_sc,) complex128 array
            energies of hopping terms in accordance with hop_i and hop_j in eV
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier is out of range
        :raises IDPCVacError: if bra or ket in hop_modifier corresponds
            to a vacancy
        """
        self.sync_array()

        # Get initial hopping terms
        hop_i, hop_j, hop_v =  \
            core.build_hop(self.pc_hop_ind, self.pc_hop_eng,
                           self.dim, self.pbc, self.num_orb_pc,
                           self.orb_id_pc, self.vac_id_sc,
                           self.sc_lat_vec, None,
                           data_kind=0)

        # Apply hopping modifier
        if self.hop_modifier is not None \
                and len(self.hop_modifier.indices) != 0:
            # Convert hopping indices to sc representation
            id_bra_pc, id_ket_pc = self.hop_modifier.get_orb_ind()
            id_bra_sc = self.orb_id_pc2sc_array(id_bra_pc)
            id_ket_sc = self.orb_id_pc2sc_array(id_ket_pc)

            # Parse hopping terms
            hop_i_new, hop_j_new, hop_v_new = [], [], []
            for ih in range(id_bra_sc.shape[0]):
                id_bra = id_bra_sc.item(ih)
                id_ket = id_ket_sc.item(ih)
                id_same, id_conj = \
                    core.find_equiv_hopping(hop_i, hop_j, id_bra, id_ket)

                # If the hopping term already exists, update it.
                hop_energy = self.hop_modifier.energies[ih]
                if id_same != -1:
                    hop_v[id_same] = hop_energy

                # If the conjugate counterpart of the hopping term exists,
                # update it.
                elif id_conj != -1:
                    hop_v[id_conj] = hop_energy.conjugate()

                # If neither the hopping term nor its conjugate counterpart
                # exist, append it to hop_i, hop_j and hop_v.
                else:
                    hop_i_new.append(id_bra)
                    hop_j_new.append(id_ket)
                    hop_v_new.append(hop_energy)

            # Append additional hopping terms
            hop_i = np.append(hop_i, hop_i_new)
            hop_j = np.append(hop_j, hop_j_new)
            hop_v = np.append(hop_v, hop_v_new)

        # Check for diagonal, duplicate or conjugate terms in hopping terms
        # NOTE: the checking procedure is EXTREMELY SLOW for large models even
        # though it is implemented in Cython. So it is disabled by default.
        # Hopefully, the limitation on super cell dimension will prohibit the
        # ill situations.
        # status = core.check_hop(hop_i, hop_j)
        # if status[0] == -3:
        #     raise ValueError(f"Diagonal term detected {status[1]}")
        # elif status[0] == -2:
        #     raise ValueError(f"Conjugate terms detected {status[1]} "
        #                      f"{status[2]}")
        # elif status[0] == -1:
        #     raise ValueError(f"Duplicate terms detected {status[1]} "
        #                      f"{status[2]}")
        # else:
        #     pass
        return hop_i, hop_j, hop_v

    def get_dr(self):
        """
        Get distances of all hopping terms in the super cell.

        NOTE: The hopping distances will be reduced by conjugate relation.
        So only half of them will be returned as results.

        NOTE: If periodic conditions are enabled, orbital indices in hop_j may
        be wrapped back if it falls out of the super cell. Nevertheless, the
        distances in dr are still the ones before wrapping. This is essential
        for adding magnetic field, calculating band structure and many
        properties involving dx and dy.

        :return: dr: (num_hop_sc, 3) float64 array
            distances of hopping terms in accordance with hop_i and hop_j in nm
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier is out of range
        :raises IDPCVacError: if bra or ket in hop_modifier corresponds
            to a vacancy
        """
        self.sync_array()

        # Get initial dr
        orb_pos = self.get_orb_pos()
        dr = core.build_hop(self.pc_hop_ind, self.pc_hop_eng,
                            self.dim, self.pbc, self.num_orb_pc,
                            self.orb_id_pc, self.vac_id_sc,
                            self.sc_lat_vec, orb_pos,
                            data_kind=1)

        # Append additional terms from hop_modifier
        dr_new = []
        hop_i, hop_j, hop_v = self.get_hop()
        for i in range(dr.shape[0], hop_i.shape[0]):
            id_bra, id_ket = hop_i.item(i), hop_j.item(i)
            dr_new.append(orb_pos[id_ket] - orb_pos[id_bra])
        if len(dr_new) != 0:
            dr = np.vstack((dr, dr_new))
        return dr

    def trim(self):
        """
        Trim dangling orbitals and associated hopping terms.

        :return: None.
            self.vacancy_list, self.vac_id_pc, self.vac_id_sc and self.orb_id_pc
            are modified.
        :raises OrbSetLockError: if the object is locked
        """
        # Get indices of dangling orbitals in primitive cell representation
        hop_i, hop_j, hop_v = self.get_hop()
        orb_id_trim = core.get_orb_id_trim(self.orb_id_pc, hop_i, hop_j)

        # Add vacancies
        for orb_id in orb_id_trim:
            self.add_vacancy(orb_id)
        self.sync_array(force_sync=True)

        # Also trim hop_modifier
        if self.hop_modifier is not None:
            self.hop_modifier.unlock()
            self.hop_modifier.trim(orb_id_trim)
            self.hop_modifier.lock()

    def get_reciprocal_vectors(self):
        """
        Get the Cartesian coordinates of reciprocal lattice vectors in 1/NM.

        :return: (3, 3) float64 array, reciprocal vectors in 1/NM.
        """
        return lat.gen_reciprocal_vectors(self.sc_lat_vec)

    def get_lattice_area(self, direction="c"):
        """
        Get the area formed by lattice vectors normal to given direction.

        :param direction: string, should be in ("a", "b", "c")
            direction of area, e.g. "c" indicates the area formed by lattice
            vectors in the aOb plane.
        :return: float, area formed by lattice vectors in NM^2.
        """
        return lat.get_lattice_area(self.sc_lat_vec, direction)

    def get_lattice_volume(self):
        """
        Get the volume formed by all three lattice vectors in NM^3.

        :return: float, volume in NM^3.
        """
        return lat.get_lattice_volume(self.sc_lat_vec)

    def plot(self, axes: plt.Axes, with_orbitals=True, with_cells=True,
             hop_as_arrows=True, hop_eng_cutoff=1e-5, view="ab"):
        """
        Plot lattice vectors, orbitals, and hopping terms to axes.

        :param axes: instance of matplotlib 'Axes' class
            axes on which the figure will be plotted
        :param with_orbitals: boolean
            whether to plot orbitals as filled circles
        :param with_cells: boolean
            whether to plot borders of primitive cells
        :param hop_as_arrows: boolean
            whether to plot hopping terms as arrows
        :param hop_eng_cutoff: float
            cutoff for showing hopping terms.
            Hopping terms with absolute energy below this value will not be
            shown in the plot.
        :param view: string
            kind of view point
            should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
        :return: None.
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier is out of range
        :raises IDPCVacError: if bra or ket in hop_modifier corresponds
            to a vacancy
        :raises ValueError: if view is illegal
        """
        viewer = ModelViewer(axes, self.pc_lat_vec, view)

        # Plot orbitals
        orb_pos = self.get_orb_pos()
        orb_eng = self.get_orb_eng()
        if with_orbitals:
            viewer.scatter(orb_pos, c=orb_eng)

        # Plot hopping terms
        hop_i, hop_j, hop_v = self.get_hop()
        dr = self.get_dr()
        for i_h in range(hop_i.shape[0]):
            if abs(hop_v.item(i_h)) >= hop_eng_cutoff:
                pos_i = orb_pos[hop_i.item(i_h)]
                # pos_j = orb_pos[hop_j.item(i_h)]
                pos_j = pos_i + dr[i_h]
                if hop_as_arrows:
                    viewer.plot_arrow(pos_i, pos_j, color='r',
                                      length_includes_head=True,
                                      width=0.002, head_width=0.02, fill=False)
                else:
                    viewer.add_line(pos_i, pos_j)
        if not hop_as_arrows:
            viewer.plot_line(color='r')

        # Plot cells
        if with_cells:
            if view in ("ab", "ba"):
                viewer.add_grid(0, self.dim.item(0), 0, self.dim.item(1))
            elif view in ("bc", "cb"):
                viewer.add_grid(0, self.dim.item(1), 0, self.dim.item(2))
            else:
                viewer.add_grid(0, self.dim.item(0), 0, self.dim.item(2))
            viewer.plot_grid(color="k", linestyle=":")
            viewer.plot_lat_vec(color="k", length_includes_head=True,
                                width=0.005, head_width=0.02)

    @property
    def pc_lat_vec(self):
        """
        Get the lattice vectors of primitive cell.

        :return: (3, 3) float64 array
            lattice vectors of primitive cell in nm.
        """
        return self.prim_cell.lat_vec

    @property
    def sc_lat_vec(self):
        """
        Get the lattice vectors of super cell.

        :return: (3, 3) float64 array
            lattice vectors of primitive cell in nm.
        """
        sc_lat_vec = self.pc_lat_vec.copy()
        for i in range(3):
            sc_lat_vec[i] *= self.dim.item(i)
        return sc_lat_vec

    @property
    def pc_orb_pos(self):
        """
        Get the orbital positions of primitive cell.

        :return: (num_orb_pc, 3) float64 array
            fractional positions of primitive cell
        """
        self.prim_cell.sync_array()
        return self.prim_cell.orb_pos

    @property
    def pc_orb_eng(self):
        """
        Get the energies of orbitals of primitive cell.

        :return: (num_orb_pc,) float64 array
            energies of orbitals of primitive cell in eV.
        """
        self.prim_cell.sync_array()
        return self.prim_cell.orb_eng

    @property
    def pc_hop_ind(self):
        """
        Get the indices of hopping terms of primitive cell.

        :return: (num_hop_pc, 5) int32 array
            indices of hopping terms of primitive cell
        """
        self.prim_cell.sync_array()
        return self.prim_cell.hop_ind

    @property
    def pc_hop_eng(self):
        """
        Get the energies of hopping terms of primitive cell.

        :return: (num_hop_pc,) complex128 array
            hopping energies of primitive cell in eV
        """
        self.prim_cell.sync_array()
        return self.prim_cell.hop_eng
