"""Functions and classes for supercell."""

from typing import Callable, Iterable, Tuple, Union, Set

import numpy as np
import matplotlib.pyplot as plt

from ..base import lattice as lat
from ..cython import super as core
from . import exceptions as exc
from .base import (check_rn, check_pbc, Observable, IntraHopping, rn_type,
                   pbc_type, id_pc_type)
from .primitive import PrimitiveCell
from .visual import ModelViewer


hop_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class OrbitalSet(Observable):
    """
    Container class for orbitals and vacancies in the supercell.

    Attributes
    ----------
    _prim_cell: 'PrimitiveCell' instance
        primitive cell from which the supercell is constructed
    _dim: (3,) int32 array
        dimension of the supercell along a, b, and c directions
    _pbc: (3,) int32 array
        whether to enable periodic condition along a, b, and c directions
        0 for False, 1 for True.
    _vacancy_set: Set[Tuple[int, int, int, int]]
        indices of vacancies in primitive cell representation (ia, ib, ic, io)
    _hash_dict: Dict[str, int]
        hashes of attributes to be used by 'sync_array' to update the arrays
    _vac_id_sc: (num_vac,) int64 array
        indices of vacancies in supercell representation
    _orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals in primitive cell representation

    NOTES
    -----
    1. Minimal supercell dimension

    Assume that we have a primitive cell located at R=0. The furthest primitive
    cell between which hopping terms exist is located at R=N. It can be proved
    that if the dimension of supercell along that direction is less than N,
    the same matrix element hij will appear more than one time in hop_i, hop_j
    and hop_v of 'SuperCell' class, as well as its conjugate counterpart. This
    will complicate the program and significantly slow it down, which situation
    we must avoid.

    Further, if the dimension of supercell falls in [N, 2*N], hij will appear
    only one time, yet its conjugate counterpart still exists. Although no
    problems have been found so far, we still need to avoid this situation.

    So the minimal dimension of supercell is 2*N+1, where N is the index of
    the furthest primitive cell between which hopping terms exists. Otherwise,
    the 'SuperCell' class, as well as the core functions of '_get_num_hop_sc',
    'build_hop', 'build_hop_k' and 'fill_ham' will not work properly.

    In the hr.dat file produced by Wannier90, there is an N_min and an N_max
    for the furthest primitive cell index. In that case, N should be the
    maximum of |N_max| and |N_min| as the result of translational symmetry.

    2. Why no vac_id_pc and orb_id_sc

    It's unnecessary to have the vac_id_pc array, as it can be generated from
    vacancy_set on-the-fly. Similarly, orb_id_sc can be generated from
    orb_id_pc. So we do not keep it in memory for reducing RAM usage.

    However, it should be noted that vac_id_sc and orb_id_sc are generated via
    different approaches. We show it by an example of 2*2 supercell with 2
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
    def __init__(self, prim_cell: PrimitiveCell,
                 dim: rn_type,
                 pbc: pbc_type = (False, False, False),
                 vacancies: Union[Iterable[id_pc_type], np.ndarray] = None) -> None:
        """
        :param prim_cell: primitive cell from which the supercell is constructed
        :param dim: dimension of the supercell along a, b and c directions
        :param pbc: whether to enable periodic boundary condition along a, b,
            and c directions
        :param vacancies: list of indices of vacancies in primitive cell
            representation
        :raises SCDimLenError: if len(dim) != 2 or 3
        :raises SCDimSizeError: if dimension is smaller than minimal value
        :raises PBCLenError: if len(pbc) != 2 or 3
        :raises IDPCLenError: if any vacancy does not have right length
        :raises IDPCIndexError: if cell or orbital index of any vacancy is
            out of range
        """
        super().__init__()

        # Set and lock the primitive cell
        self._prim_cell = prim_cell
        self._prim_cell.add_subscriber(f"supercell #{id(self)}", self)
        self._prim_cell.lock(f"supercell #{id(self)}")

        # Check and set dimension
        dim, legal = check_rn(dim, complete_item=1)
        if not legal:
            raise exc.SCDimLenError(dim)
        hop_ind = self._prim_cell.hop_ind
        for i in range(3):
            rn_min = hop_ind[:, i].min()
            rn_max = hop_ind[:, i].max()
            dim_min = max(abs(rn_min), abs(rn_max))
            if dim[i] < dim_min:
                raise exc.SCDimSizeError(i, dim_min)
        self._dim = np.array(dim, dtype=np.int32)

        # Check and set periodic boundary condition
        pbc, legal = check_pbc(pbc, complete_item=False)
        if not legal:
            raise exc.PBCLenError(pbc)
        self._pbc = np.array([1 if _ else 0 for _ in pbc], dtype=np.int32)

        # Initialize arrays assuming no vacancies
        self._vacancy_set = set()
        self._hash_dict = {'pc': self._get_hash('pc'),
                           'vac': self._get_hash('vac')}
        self._vac_id_sc = None
        self._orb_id_pc = core.build_orb_id_pc(self._dim, self.num_orb_pc,
                                               self._vac_id_sc)

        # Set vacancies if any
        if vacancies is not None:
            self.add_vacancies(vacancies)

    def __hash__(self) -> int:
        """Return the hash of this instance."""
        fp = (self._prim_cell, tuple(self._dim), tuple(self._pbc),
              tuple(self._vacancy_set))
        return hash(fp)

    def _get_hash(self, attr: str) -> int:
        """
        Get hash of given attribute.

        For the primitive cell, we can use both its actual hash or the number
        of orbitals as the fingerprint, while the latter is much faster.

        :param attr: name of the attribute
        :return: hash of the attribute
        :raises ValueError: if attr is illegal
        """
        if attr == "pc":
            new_hash = hash(self._prim_cell)
            # new_hash = self.num_orb_pc
        elif attr == "vac":
            new_hash = hash(tuple(self._vacancy_set))
        else:
            raise ValueError(f"Illegal attribute name {attr}")
        return new_hash

    def _update_hash(self, attr: str) -> bool:
        """
        Compare and update hash of given attribute.

        :param attr: name of the attribute
        :return: whether the hash has been updated
        :raises ValueError: if attr is illegal
        """
        new_hash = self._get_hash(attr)
        if self._hash_dict[attr] != new_hash:
            self._hash_dict[attr] = new_hash
            status = True
        else:
            status = False
        return status

    def _check_id_pc(self, id_pc: Union[id_pc_type, np.ndarray]) -> None:
        """
        Checks if orbital or vacancy index in primitive cell representation
        is legal.

        A legal id_pc should have cell indices falling within 0 <= rn < dim and
        orbital index falling within 0 <= ib < num_orb_pc.

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
                if id_pc[i] not in range(self._dim.item(i)):
                    raise exc.IDPCIndexError(i, id_pc)
            if id_pc[3] not in range(self.num_orb_pc):
                raise exc.IDPCIndexError(3, id_pc)
        elif isinstance(id_pc, np.ndarray):
            for i in range(3):
                if id_pc.item(i) not in range(self._dim.item(i)):
                    raise exc.IDPCIndexError(i, id_pc)
            if id_pc.item(3) not in range(self.num_orb_pc):
                raise exc.IDPCIndexError(3, id_pc)
        else:
            raise exc.IDPCTypeError(id_pc)

    def add_vacancy(self, vacancy: Union[id_pc_type, np.ndarray]) -> None:
        """
        Wrapper over 'add_vacancies' to add a single vacancy to the orbital set.

        :param vacancy: (ia, ib, ic, io) or equivalent int32 array
            vacancy index in primitive cell representation
        :return: None
        :raises LockError: if the object is locked
        :raises IDPCLenError: if length of vacancy index is not 4
        :raises IDPCIndexError: if cell or orbital index of vacancy is
            out of range
        """
        self.check_lock()
        self.add_vacancies([vacancy])

    def add_vacancies(self, vacancies: Union[Iterable[id_pc_type], np.ndarray]) -> None:
        """
        Add a list of vacancies to the orbital set.

        :param vacancies: list of (ia, ib, ic, io) or equivalent int32 arrays
            list of indices of vacancies in primitive cell representation
        :return: None
        :raises LockError: if the object is locked
        :raises IDPCLenError: if length of vacancy index is not 4
        :raises IDPCIndexError: if cell or orbital index of vacancy is
            out of range
        """
        self.check_lock()
        for vacancy in vacancies:
            if not isinstance(vacancy, tuple):
                vacancy = tuple(vacancy)
            self._check_id_pc(vacancy)
            self._vacancy_set.add(vacancy)

    def set_vacancies(self, vacancies: Union[Iterable[id_pc_type], np.ndarray] = None) -> None:
        """
        Reset the set of vacancies.

        :param vacancies: list of (ia, ib, ic, io) or equivalent int32 arrays
            list of indices of vacancies in primitive cell representation
        :return: None
        :raises LockError: if the object is locked
        :raises IDPCLenError: if length of vacancy index is not 4
        :raises IDPCIndexError: if cell or orbital index of vacancy is
            out of range
        """
        self.check_lock()
        self._vacancy_set = set()
        self.add_vacancies(vacancies)

    def sync_array(self, verbose: bool = False,
                   force_sync: bool = False) -> None:
        """
        Synchronize vac_id_sc and orb_id_pc according to primitive cell and
        vacancies.

        NOTE: The core function '_id_pc2sc_vac' requires vac_id_sc to be sorted
        in increasing order. Otherwise, it won't work properly! So we must sort
        it here.

        :param verbose: whether to output additional debugging information
        :param force_sync: whether to force synchronizing the arrays even if
            primitive cell and vacancy_set did not change
        :return: None
        """
        to_update = self._update_hash("pc") or self._update_hash("vac")
        if force_sync or to_update:
            if verbose:
                print("INFO: updating sc vacancy and orbital arrays")

            # If vacancy set is not empty, update arrays as usual.
            if len(self._vacancy_set) != 0:
                vac_id_pc = np.array(sorted(self._vacancy_set), dtype=np.int32)
                vac_id_sc = core.build_vac_id_sc(self._dim, self.num_orb_pc,
                                                 vac_id_pc)
                sorted_idx = np.argsort(vac_id_sc, axis=0)
                self._vac_id_sc = vac_id_sc[sorted_idx]
                self._orb_id_pc = core.build_orb_id_pc(self._dim, self.num_orb_pc,
                                                       self._vac_id_sc)

            # Otherwise, restore to default settings as in __init__.
            else:
                self._vac_id_sc = None
                self._orb_id_pc = core.build_orb_id_pc(self._dim, self.num_orb_pc,
                                                       self._vac_id_sc)
            self._prim_cell.lock(f"supercell #{id(self)}")
        else:
            if verbose:
                print("INFO: no need to update sc vacancy and orbital arrays")

    def orb_id_sc2pc(self, id_sc: int) -> np.ndarray:
        """
        Convert orbital (NOT VACANCY) index from sc representation to pc
        representation.

        NOTE: This method is safe, but EXTREMELY SLOW. If you are going to
        call this method many times, use orb_id_sc2pc_array instead.

        :param id_sc: index of orbital in supercell representation
        :return: (4,) int32 array
            index of orbital in primitive cell representation
        :raises IDSCIndexError: if id_sc is out of range
        """
        self.sync_array()
        try:
            id_pc = self._orb_id_pc[id_sc]
        except IndexError as err:
            raise exc.IDSCIndexError(id_sc) from err
        return id_pc

    def orb_id_pc2sc(self, id_pc: Union[id_pc_type, np.ndarray]) -> int:
        """
        Convert orbital (NOT VACANCY) index from pc representation to sc
        representation.

        NOTE: This method is safe, but EXTREMELY SLOW. If you are going to
        call this method many times, use orb_id_pc2sc_array instead.

        :param id_pc: (ia, ib, ic, io), or equivalent int32 array
            index of orbital in primitive cell representation
        :return: index of orbital in supercell representation
        :raises IDPCLenError: if len(id_pc) != 4
        :raises IDPCIndexError: if cell or orbital index of id_pc is
            out of range
        :raises IDPCTypeError: if id_pc is not tuple or numpy array
        :raises IDPCVacError: if id_pc corresponds to a vacancy
        """
        self.sync_array()
        self._check_id_pc(id_pc)
        if not isinstance(id_pc, np.ndarray) or id_pc.dtype != np.int32:
            id_pc = np.array(id_pc, dtype=np.int32)
        orb_id_sc = core.id_pc2sc(self._dim, self.num_orb_pc, id_pc,
                                  self._vac_id_sc)
        if orb_id_sc == -1:
            raise exc.IDPCVacError(id_pc)
        return orb_id_sc

    def orb_id_sc2pc_array(self, id_sc_array: np.ndarray) -> np.ndarray:
        """
        Convert an array of orbital (NOT VACANCY) indices from sc
        representation to pc representation.

        :param id_sc_array: (num_orb,) int64 array
            orbital indices in supercell representation
        :return: (num_orb, 4) int32 array
            orbital indices in primitive cell representation
        :raises ValueError: if id_sc_array is not a vector
        :raises IDSCIndexError: if any id_sc in id_sc_array is out of range
        """
        self.sync_array()
        if not isinstance(id_sc_array, np.ndarray) \
                or id_sc_array.dtype != np.int64:
            id_sc_array = np.array(id_sc_array, dtype=np.int64)
        if len(id_sc_array.shape) != 1:
            raise ValueError("id_sc_array should be a vector")
        status = core.check_id_sc_array(self.num_orb_sc, id_sc_array)
        if status[0] == -1:
            raise exc.IDSCIndexError(id_sc_array[status[1]])
        id_pc_array = core.id_sc2pc_array(self._orb_id_pc, id_sc_array)
        return id_pc_array

    def orb_id_pc2sc_array(self, id_pc_array: np.ndarray) -> np.ndarray:
        """
        Convert an array of orbital (NOT VACANCY) indices from pc
        representation to sc representation.

        :param id_pc_array: (num_orb, 4) int32 array
            orbital indices in primitive cell representation
        :return: (num_orb,) int64 array
            orbital indices in supercell representation
        :raises IDPCLenError: if id_pc_array.shape[1] != 4
        :raises IDPCIndexError: if any id_pc in id_pc_array is out of range
        :raises IDPCVacError: if any id_pc in id_pc_array is a vacancy
        """
        self.sync_array()
        if not isinstance(id_pc_array, np.ndarray) \
                or id_pc_array.dtype != np.int32:
            id_pc_array = np.array(id_pc_array, dtype=np.int32)
        if id_pc_array.shape[1] != 4:
            raise exc.IDPCLenError(id_pc_array[0])
        status = core.check_id_pc_array(self._dim, self.num_orb_pc,
                                        id_pc_array, self._vac_id_sc)
        if status[0] == -2:
            raise exc.IDPCIndexError(status[2], id_pc_array[status[1]])
        if status[0] == -1:
            raise exc.IDPCVacError(id_pc_array[status[1]])
        id_sc_array = core.id_pc2sc_array(self._dim, self.num_orb_pc,
                                          id_pc_array, self._vac_id_sc)
        return id_sc_array

    @property
    def num_orb_pc(self) -> int:
        """
        Get the number of orbitals of primitive cell.

        :return: number of orbitals in primitive cell.
        """
        return self._prim_cell.num_orb

    @property
    def num_orb_sc(self) -> int:
        """
        Get the number of orbitals of supercell.

        :return: number of orbitals in supercell
        """
        num_orb_sc = self.num_orb_pc * np.prod(self._dim).item()
        num_orb_sc -= len(self._vacancy_set)
        return num_orb_sc


class SuperCell(OrbitalSet):
    """
    Class for representing a supercell from which the sample is constructed.

    Notes on hop_modifier
    ---------------------
    1. Reduction

    We reduce hopping terms according to the conjugate relation
        <0, bra|H|R, ket> = <0, ket|H|-R, bra>*.
    So actually only half of hopping terms are stored.

    2. Rules

    If the hopping terms claimed here are already included in the supercell,
    they will overwrite the existing terms. If the hopping terms or their
    conjugate counterparts are new to 'SuperCell', they will be appended to
    hop_* arrays. The dr array will also be updated accordingly.

    Attributes
    ----------
    _hop_modifier: 'IntraHopping' instance
        modification to hopping terms in the supercell
    _orb_pos_modifier: function
        modification to orbital positions in the supercell
    """
    def __init__(self, prim_cell: PrimitiveCell,
                 dim: rn_type,
                 pbc: pbc_type = (False, False, False),
                 vacancies: Union[Iterable[id_pc_type], np.ndarray] = None,
                 orb_pos_modifier: Callable[[np.ndarray], None] = None) -> None:
        """
        :param prim_cell: primitive cell from which the supercell is constructed
        :param dim: dimension of the supercell along a, b, and c directions
        :param pbc: whether to enable periodic boundary condition along a, b and
            c directions
        :param vacancies: indices of vacancies in primitive cell representation
        :param orb_pos_modifier: modification to orbital positions in the supercell
        :return: None
        :raises SCDimLenError: if len(dim) != 2 or 3
        :raises SCDimSizeError: if dimension is smaller than minimal value
        :raises PBCLenError: if len(pbc) != 2 or 3
        :raises IDPCLenError: if any vacancy does not have right length
        :raises IDPCIndexError: if cell or orbital index of any vacancy is
            out of range
        """
        # Build orbital set
        super().__init__(prim_cell, dim, pbc=pbc, vacancies=vacancies)

        # Initialize hop_modifier and orb_pos_modifier
        self._hop_modifier = IntraHopping()
        self._orb_pos_modifier = orb_pos_modifier

    def __hash__(self) -> int:
        """Return the hash of this instance."""
        fp = (self._prim_cell, tuple(self._dim), tuple(self._pbc),
              tuple(self._vacancy_set), self._hop_modifier,
              self._orb_pos_modifier)
        return hash(fp)

    def add_hopping(self, rn: rn_type,
                    orb_i: int,
                    orb_j: int,
                    energy: complex) -> None:
        """
        Add a new term to the hopping modifier.

        :param rn: cell index of the hopping term, i.e. R
        :param orb_i: index of orbital i in <i,0|H|j,R>
        :param orb_j: index of orbital j in <i,0|H|j,R>
        :param energy: hopping integral in eV
        :return: None
        :raises LockError: if the supercell is locked
        :raises SCOrbIndexError: if orb_i or orb_j falls out of range
        :raises SCHopDiagonalError: if rn == (0, 0, 0) and orb_i == orb_j
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        self.check_lock()

        # Check params, adapted from the '_check_hop_index' method
        # of 'PrimitiveCell' class
        rn, legal = check_rn(rn)
        if not legal:
            raise exc.CellIndexLenError(rn)
        num_orbitals = self.num_orb_sc
        if not (0 <= orb_i < num_orbitals):
            raise exc.SCOrbIndexError(orb_i)
        if not (0 <= orb_j < num_orbitals):
            raise exc.SCOrbIndexError(orb_j)
        if rn == (0, 0, 0) and orb_i == orb_j:
            raise exc.SCHopDiagonalError(rn, orb_i)

        # Add the hopping term
        self._hop_modifier.add_hopping(rn, orb_i, orb_j, energy)

    def set_orb_pos_modifier(self, orb_pos_modifier: Callable = None) -> None:
        """
        Reset orb_pos_modifier.

        :param orb_pos_modifier: modifier to orbital positions
        :return: None
        :raises LockError: if the supercell is locked
        """
        self.check_lock()
        self._orb_pos_modifier = orb_pos_modifier

    def trim(self) -> None:
        """
        Trim dangling orbitals and associated hopping terms.

        :return: None.
        :raises LockError: if the object is locked
        """
        self.check_lock()

        # Get indices of dangling orbitals
        hop_i, hop_j, hop_v = self.get_hop()[:3]
        id_pc_trim = core.get_orb_id_trim(self._orb_id_pc, hop_i, hop_j)
        id_sc_trim = self.orb_id_pc2sc_array(id_pc_trim)

        # Add vacancies
        self.add_vacancies(id_pc_trim)

        # Also trim hop_modifier
        self._hop_modifier.remove_orbitals(id_sc_trim)

    def plot(self, axes: plt.Axes,
             with_orbitals: bool = True,
             with_cells: bool = True,
             with_conj: bool = False,
             hop_as_arrows: bool = True,
             hop_eng_cutoff: float = 1e-5,
             hop_color: str = "r",
             view: str = "ab") -> None:
        """
        Plot lattice vectors, orbitals, and hopping terms to axes.

        :param axes: axes on which the figure will be plotted
        :param with_orbitals: whether to plot orbitals as filled circles
        :param with_cells: whether to plot borders of primitive cells
        :param with_conj: whether to plot conjugate hopping terms as well
        :param hop_as_arrows: whether to plot hopping terms as arrows
        :param hop_eng_cutoff: cutoff for showing hopping terms
        :param hop_color: color of hopping terms
        :param view: kind of view point, should be in 'ab', 'bc', 'ca', 'ba',
            'cb', 'ac'
        :return: None
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier is out of range
        :raises IDPCVacError: if bra or ket in hop_modifier corresponds
            to a vacancy
        :raises ValueError: if view is illegal
        """
        viewer = ModelViewer(axes, self.pc_lat_vec, self.pc_origin, view)

        # Plot orbitals
        orb_pos = self.get_orb_pos()
        orb_eng = self.get_orb_eng()
        if with_orbitals:
            viewer.scatter(orb_pos, c=orb_eng)

        # Plot hopping terms
        hop_i, hop_j, hop_v, dr = self.get_hop()
        arrow_args = {"color": hop_color, "length_includes_head": True,
                      "width": 0.002, "head_width": 0.02, "fill": False}
        for i_h in range(hop_i.shape[0]):
            if abs(hop_v.item(i_h)) >= hop_eng_cutoff:
                # Original term
                pos_i = orb_pos[hop_i.item(i_h)]
                pos_j = pos_i + dr[i_h]
                if hop_as_arrows:
                    viewer.plot_arrow(pos_i, pos_j, **arrow_args)
                else:
                    viewer.add_line(pos_i, pos_j)
                # Conjugate term
                if with_conj:
                    pos_j = orb_pos[hop_j.item(i_h)]
                    pos_i = pos_j - dr[i_h]
                    if hop_as_arrows:
                        viewer.plot_arrow(pos_j, pos_i, **arrow_args)
                    else:
                        viewer.add_line(pos_j, pos_i)
        if not hop_as_arrows:
            viewer.plot_line(color=hop_color)

        # Plot cells
        if with_cells:
            if view in ("ab", "ba"):
                viewer.add_grid(0, self._dim.item(0), 0, self._dim.item(1))
            elif view in ("bc", "cb"):
                viewer.add_grid(0, self._dim.item(1), 0, self._dim.item(2))
            else:
                viewer.add_grid(0, self._dim.item(0), 0, self._dim.item(2))
            viewer.plot_grid(color="k", linestyle=":")
            viewer.plot_lat_vec(color="k", length_includes_head=True,
                                width=0.005, head_width=0.02)

    def get_orb_eng(self) -> np.ndarray:
        """
        Get energies of all orbitals in the supercell.

        :return: (num_orb_sc,) float64 array
            on-site energies of orbitals in the supercell in eV
        """
        self.sync_array()
        return core.build_orb_eng(self.pc_orb_eng, self._orb_id_pc)

    def get_orb_pos(self) -> np.ndarray:
        """
        Get positions of all orbitals in the supercell.

        :return: (num_orb_sc, 3) float64 array
            Cartesian coordinates of orbitals in the supercell in nm
        """
        self.sync_array()
        orb_pos = core.build_orb_pos(self.pc_lat_vec, self.pc_orb_pos,
                                     self._orb_id_pc)
        orb_pos += self.pc_origin
        if self._orb_pos_modifier is not None:
            self._orb_pos_modifier(orb_pos)
        return orb_pos

    def _init_hop_gen(self, orb_pos: np.ndarray) -> hop_type:
        """
        Get initial hopping terms and distances using the general algorithm.

        :param orb_pos: (num_orb_sc, 3) float64 array
            Cartesian coordinates of orbitals in NM
        :return: (hop_i, hop_j, hop_v, dr)
            initial hopping terms and distances
        """
        hop_i, hop_j, hop_v, dr = \
            core.build_hop_gen(self.pc_hop_ind, self.pc_hop_eng,
                               self._dim, self._pbc, self.num_orb_pc,
                               self._orb_id_pc, self._vac_id_sc,
                               self.sc_lat_vec, orb_pos)
        return hop_i, hop_j, hop_v, dr

    def _init_hop_fast(self, orb_pos: np.ndarray) -> hop_type:
        """
        Get initial hopping terms and distance using the fast algorithm.

        NOTE: this algorithm works only for supercells without vacancies.

        TODO: parallelize this method with MPI.

        :param orb_pos: (num_orb_sc, 3) float64 array
            Cartesian coordinates of orbitals in NM
        :return: (hop_i, hop_j, hop_v, dr)
            initial hopping terms and distances
        """
        # Split pc hopping terms into free and periodic parts
        ind_pbc, eng_pbc, ind_free, eng_free = \
            core.split_pc_hop(self.pc_hop_ind, self.pc_hop_eng, self._pbc)

        # Build sc hopping terms from periodic parts
        # This is fast since we can predict the number of hopping terms.
        i_pbc, j_pbc, v_pbc, dr_pbc = \
            core.build_hop_pbc(ind_pbc, eng_pbc,
                               self._dim, self._pbc, self.num_orb_pc,
                               self.sc_lat_vec, orb_pos)

        # Build hopping terms from free parts
        # Here we must call the general Cython function as we cannot predict
        # the number of hopping terms.
        i_free, j_free, v_free, dr_free = \
            core.build_hop_gen(ind_free, eng_free,
                               self._dim, self._pbc, self.num_orb_pc,
                               self._orb_id_pc, self._vac_id_sc,
                               self.sc_lat_vec, orb_pos)

        # Assemble hopping terms and distances
        hop_i = np.append(i_pbc, i_free)
        hop_j = np.append(j_pbc, j_free)
        hop_v = np.append(v_pbc, v_free)
        dr = np.vstack((dr_pbc, dr_free))
        return hop_i, hop_j, hop_v, dr

    def get_hop(self) -> hop_type:
        """
        Get hopping indices, energies and distances.

        The hopping terms will be reduced by conjugate relation. So only half
        of them will be returned as results.

        If periodic conditions are enabled, orbital indices in hop_j may be
        wrapped back if it falls out of the supercell. Nevertheless, the
        distances in dr are still the ones before wrapping. This is essential
        for adding magnetic field, calculating band structure and many
        properties involving dx and dy.

        :return: (hop_i, hop_j, hop_v, dr)
            hop_i: (num_hop_sc,) int64 array
            row indices of hopping terms reduced by conjugate relation
            hop_j: (num_hop_sc,) int64 array
            column indices of hopping terms reduced by conjugate relation
            hop_v: (num_hop_sc,) complex128 array
            energies of hopping terms in accordance with hop_i and hop_j in eV
            dr: (num_hop_sc, 3) float64 array
            distances of hopping terms in accordance with hop_i and hop_j in nm
        """
        self.sync_array()
        orb_pos = self.get_orb_pos()

        # Get initial hopping terms
        use_fast = (len(self._vacancy_set) == 0)
        if use_fast:
            hop_i, hop_j, hop_v, dr = self._init_hop_fast(orb_pos)
        else:
            hop_i, hop_j, hop_v, dr = self._init_hop_gen(orb_pos)

        # Apply hopping modifier
        if self._hop_modifier.num_hop != 0:
            hop_ind, hop_eng = self._hop_modifier.to_array(use_int64=True)
            hop_i_new, hop_j_new, hop_v_new, dr_new = [], [], [], []

            for ih in range(hop_ind.shape[0]):
                # Extract data
                id_bra = hop_ind.item(ih, 3)
                id_ket = hop_ind.item(ih, 4)
                hop_energy = hop_eng.item(ih)
                rn = np.matmul(hop_ind[ih, :3], self.sc_lat_vec)
                dr_i = orb_pos[id_ket] + rn - orb_pos[id_bra]

                # Check for equivalent terms
                id_same, id_conj = \
                    core.find_equiv_hopping(hop_i, hop_j, id_bra, id_ket)
                if id_same != -1:
                    hop_v[id_same] = hop_energy
                    dr[id_same] = dr_i
                elif id_conj != -1:
                    hop_v[id_conj] = hop_energy.conjugate()
                    dr[id_conj] = -dr_i
                else:
                    hop_i_new.append(id_bra)
                    hop_j_new.append(id_ket)
                    hop_v_new.append(hop_energy)
                    dr_new.append(dr_i)

            # Append additional hopping terms
            hop_i = np.append(hop_i, hop_i_new)
            hop_j = np.append(hop_j, hop_j_new)
            hop_v = np.append(hop_v, hop_v_new)
            if len(dr_new) != 0:
                dr = np.vstack((dr, dr_new))
        return hop_i, hop_j, hop_v, dr

    def get_lattice_area(self, direction: str = "c") -> float:
        """
        Get the area formed by lattice vectors normal to given direction.

        :param direction: direction of area, e.g. "c" indicates the area formed
            by lattice vectors in the aOb plane.
        :return: area formed by lattice vectors in NM^2
        """
        return lat.get_lattice_area(self.sc_lat_vec, direction)

    def get_lattice_volume(self) -> float:
        """
        Get the volume formed by all three lattice vectors in NM^3.

        :return: volume in NM^3
        """
        return lat.get_lattice_volume(self.sc_lat_vec)

    def get_reciprocal_vectors(self) -> np.ndarray:
        """
        Get the Cartesian coordinates of reciprocal lattice vectors in 1/NM.

        :return: (3, 3) float64 array
            reciprocal vectors in 1/NM
        """
        return lat.gen_reciprocal_vectors(self.sc_lat_vec)

    @property
    def prim_cell(self) -> PrimitiveCell:
        """
        Interface for the '_prim_cell' attribute.

        :return: the primitive cell
        """
        return self._prim_cell

    @property
    def pc_lat_vec(self) -> np.ndarray:
        """
        Get the lattice vectors of primitive cell.

        :return: (3, 3) float64 array
            lattice vectors of primitive cell in nm.
        """
        return self._prim_cell.lat_vec

    @property
    def sc_lat_vec(self) -> np.ndarray:
        """
        Get the lattice vectors of supercell.

        :return: (3, 3) float64 array
            lattice vectors of primitive cell in nm.
        """
        sc_lat_vec = self.pc_lat_vec.copy()
        for i in range(3):
            sc_lat_vec[i] *= self._dim.item(i)
        return sc_lat_vec

    @property
    def pc_origin(self) -> np.ndarray:
        """
        Get the lattice origin of primitive cell.

        :return: (3,) float64 array
            lattice origin of primitive cell in NM
        """
        return self._prim_cell.origin

    @property
    def pc_orb_pos(self) -> np.ndarray:
        """
        Get the orbital positions of primitive cell.

        :return: (num_orb_pc, 3) float64 array
            fractional positions of primitive cell
        """
        return self._prim_cell.orb_pos

    @property
    def pc_orb_eng(self) -> np.ndarray:
        """
        Get the energies of orbitals of primitive cell.

        :return: (num_orb_pc,) float64 array
            energies of orbitals of primitive cell in eV.
        """
        return self._prim_cell.orb_eng

    @property
    def pc_hop_ind(self) -> np.ndarray:
        """
        Get the indices of hopping terms of primitive cell.

        :return: (num_hop_pc, 5) int32 array
            indices of hopping terms of primitive cell
        """
        return self._prim_cell.hop_ind

    @property
    def pc_hop_eng(self) -> np.ndarray:
        """
        Get the energies of hopping terms of primitive cell.

        :return: (num_hop_pc,) complex128 array
            hopping energies of primitive cell in eV
        """
        return self._prim_cell.hop_eng
