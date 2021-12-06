"""
Functions and classes for sample.

Functions
---------
    None.

Classes
-------
    InterHopping: user class
        container class for hopping terms between different super cells within
        the sample
    Sample: user class
        interface class to FORTRAN backend
"""

import math
from typing import Union

import numpy as np
import scipy.linalg.lapack as lapack
from scipy.sparse import dia_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from . import lattice as lat
from . import kpoints as kpt
from . import exceptions as exc
from . import core
from .primitive import correct_coord, LockableObject
from .super import SuperCell
from .utils import ModelViewer


class InterHopping(LockableObject):
    """
    Container class for hopping terms between different super cells within the
    sample.

    Attributes
    ----------
    _sc_bra: instance of 'SuperCell' class
        the 'bra' super cell from which the hopping terms exist
    _sc_ket: instance of 'SuperCell' class
        the 'ket' super cell to which the hopping terms exist
    _indices: list of ((ia, ib, ic, io), (ia', ib', ic', io'))
        where (ia, ib, ic, io) is the index of bra in primitive cell
        representation and (ia', ib', ic', io') is the index of ket
    _energies: list of complex numbers
        hopping energies corresponding to indices in eV

    NOTES
    -----
    1. Sanity check

    This class is intended to constitute the 'Sample' class and is loosely
    coupled to 'SuperCell' class. So we need to check the hopping indices
    here. This is done in the 'get_hop' method.

    2. Reduction

    Since inter-hopping terms exist within different super-cells, there is no
    need to reduce them according to the conjugate relation.

    3. Rules

    We restrict hopping terms to be from the (0, 0, 0) 'bra' supercell to any
    'ket' supercell even if periodic conditions are enabled. Other hopping terms
    will be treated as illegal. The counterparts are restored via the conjugate
    relation:
        <bra, R0, i|H|ket, Rn, j> = <ket, R0, j|H|bra, -Rn, i>*
    """
    def __init__(self, sc_bra: SuperCell, sc_ket: SuperCell):
        """
        :param sc_bra: instance of 'SuperCell' class
            the 'bra' super cell from which the hopping terms exist
        :param sc_ket: instance of 'SuperCell' class
            the 'ket' super cell to which the hopping terms exist
        """
        super().__init__()
        self._sc_bra = sc_bra
        self._sc_ket = sc_ket
        self._indices = []
        self._energies = []

    def __find_equiv_hopping(self, bra, ket):
        """
        Find the index of equivalent hopping term of <bra|H|ket>.

        :param bra: (ia, ib, ic, io)
            index of bra of the hopping term in primitive cell representation
        :param ket: (ia', ib', ic', io')
            index of ket of the hopping term in primitive cell representation
        :return: id_same: integer
            index of the same hopping term, none if not found
        """
        assert len(bra) == len(ket) == 4
        id_same = None
        hop_same = (bra, ket)
        try:
            id_same = self._indices.index(hop_same)
        except ValueError:
            pass
        return id_same

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
        :return: None.
            self._indices and self._energies are modified.
        :raises InterHopLockError: if the object is locked
        :raises IDPCLenError: if len(rn_i) or len(rn_j) not in (2, 3)
        """
        try:
            self.check_lock_state()
        except exc.LockError as err:
            raise exc.InterHopLockError() from err

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

        # Update existing hopping term, or add the new term to hopping_list.
        id_same = self.__find_equiv_hopping(bra, ket)
        if id_same is not None:
            self._energies[id_same] = energy
        else:
            self._indices.append((bra, ket))
            self._energies.append(energy)

    def __get_ket_dim_min(self):
        """
        Get the minimal dimension of sc_ket in order to avoid duplicate terms
        in hop_i and hop_j.

        :return: ket_dim_min: (na, nb, nc)
            minimal dimension of sc_ket
        """
        ket_dim_min = [1, 1, 1]
        id_bra_set = set([_[0] for _ in self._indices])
        for id_bra in id_bra_set:
            id_ket_array = np.array([_[1] for _ in self._indices
                                     if _[0] == id_bra], dtype=np.int32)
            for i in range(3):
                ri = id_ket_array[:, i]
                ni = ri.max() - ri.min() + 1
                if ket_dim_min[i] <= ni:
                    ket_dim_min[i] = ni
        return ket_dim_min

    def get_hop(self, algo=1):
        """
        Get hopping indices and energies.

        NOTE: we need to check if there are duplicate terms in hop_i and hop_j.
        Two algorithms have been implemented. algo=1 uses a Cython function to
        check orbital index pairs for exactly duplicate terms, while algo=2
        checks the dimension of _sc_ket for potential duplicate terms. Both
        algorithms are considered to be slow for large systems. Set algo=0
        to turn it off to reduce time cost, at your own risk.

        :param algo: integer
            kind of algorithm to check for duplicate terms
            Set to 0 to turn it off.
        :return: hop_i: (num_hop,) int64 array
            row indices of hopping terms
        :return: hop_j: (num_hop,) int64 array
            column indices of hopping terms
        :return: hop_v: (num_hop,) complex128 array
            energies of hopping terms in accordance with hop_i and hop_j in eV
        :raises InterHopVoidError: if no hopping terms have been added to the
            instance
        :raises IDPCIndexError: if cell or orbital index of ket in self._indices
            is out of range
        :raises IDPCVacError: if bra or ket in self._indices corresponds to a
            vacancy
        :raises ValueError: if duplicate terms have been detected in hop_i
            and hop_j, or dimension of _sc_ket is too small
        """
        if len(self._indices) == 0:
            raise exc.InterHopVoidError()
        id_bra_pc = np.array([_[0] for _ in self._indices], dtype=np.int32)
        id_ket_pc = np.array([_[1] for _ in self._indices], dtype=np.int32)

        # NOTE: id_ket_pc needs to be wrapped back into (0, 0, 0) super cell
        # since it can reside at any super cell. On the contrary, id_bra_ket
        # is restricted (0, 0, 0) super cell and need on wrap. Errors will
        # be raised by orb_id_pc2sc_array it falls out of (0, 0, 0).
        self._sc_ket.wrap_id_pc_array(id_ket_pc)
        hop_i = self._sc_bra.orb_id_pc2sc_array(id_bra_pc)
        hop_j = self._sc_ket.orb_id_pc2sc_array(id_ket_pc)
        hop_v = np.array(self._energies, dtype=np.complex128)

        # Check for duplicate terms
        if algo == 0:
            pass
        elif algo == 1:
            status = core.check_inter_hop(hop_i, hop_j)
            if status[0] == -1:
                raise ValueError(f"Duplicate terms detected {status[1]} "
                                 f"{status[2]}")
        else:
            dim_min = self.__get_ket_dim_min()
            for i in range(3):
                current_dim = self._sc_ket.dim[i]
                if current_dim < dim_min[i]:
                    raise ValueError(f"Dimension of sc_ket {current_dim} is "
                                     f"smaller than {dim_min[i]}")
        return hop_i, hop_j, hop_v

    def get_dr(self):
        """
        Get hopping distances.

        NOTE: If periodic conditions are enabled, orbital indices in hop_j may
        be wrapped back if it falls out of the super cell. Nevertheless, the
        distances in dr are still the ones before wrapping. This is essential
        for adding magnetic field, calculating band structure and many
        properties involving dx and dy.

        :return: dr: (num_hop, 3) float64 array
            distances of hopping terms in accordance with hop_i and hop_j in nm
        :raises InterHopVoidError: if no hopping terms have been added to the
            instance
        :raises IDPCIndexError: if cell or orbital index of ket in self._indices
            is out of range
        :raises IDPCVacError: if bra or ket in self._indices corresponds
            to a vacancy
        :raises ValueError: if duplicate terms have been detected in hop_i
            and hop_j, or dimension of _sc_ket is too small
        """
        if len(self._indices) == 0:
            raise exc.InterHopVoidError()
        hop_i, hop_j, hop_v = self.get_hop()
        pos_bra = self._sc_bra.get_orb_pos()
        pos_ket = self._sc_ket.get_orb_pos()
        id_ket_pc = np.array([_[1] for _ in self._indices], dtype=np.int32)
        dr = core.build_inter_dr(hop_i, hop_j, pos_bra, pos_ket, id_ket_pc,
                                 self._sc_ket.dim, self._sc_ket.sc_lat_vec)
        return dr

    def plot(self, axes: plt.Axes, hop_as_arrows=True, hop_eng_cutoff=1e-5,
             view="ab"):
        """
        Plot hopping terms to axes.

        :param axes: instance of matplotlib 'Axes' class
            axes on which the figure will be plot
        :param hop_as_arrows: boolean
            whether to plot hopping terms as arrows
        :param hop_eng_cutoff: float
            cutoff for showing hopping terms
            Hopping terms with absolute energy below this value will not be
            shown in the plot.
        :param view: string
            kind of view point
            should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
        :return: None.
        :raises InterHopVoidError: if no hopping terms have been added to the
            instance
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            self._indices is out of range
        :raises IDPCVacError: if bra or ket in self._indices corresponds
            to a vacancy
        :raises ValueError: if view is illegal
        """
        viewer = ModelViewer(axes, self._sc_bra.pc_lat_vec, view)

        # Plot hopping terms
        orb_pos_i = self._sc_bra.get_orb_pos()
        orb_pos_j = self._sc_ket.get_orb_pos()
        hop_i, hop_j, hop_v = self.get_hop()
        for i_h in range(hop_i.shape[0]):
            if abs(hop_v.item(i_h)) >= hop_eng_cutoff:
                pos_i = orb_pos_i[hop_i.item(i_h)]
                pos_j = orb_pos_j[hop_j.item(i_h)]
                if hop_as_arrows:
                    viewer.plot_arrow(pos_i, pos_j, color='r',
                                      length_includes_head=True,
                                      width=0.002, head_width=0.02, fill=False)
                else:
                    viewer.add_line(pos_i, pos_j)
        if not hop_as_arrows:
            viewer.plot_line(color="r")

    @property
    def indices(self):
        """
        :return: self._indices
        """
        return self._indices

    @property
    def energies(self):
        """
        :return: self._energies
        """
        return self._energies

    @property
    def sc_bra(self):
        """
        :return: self._sc_bra
        """
        return self._sc_bra

    @property
    def sc_ket(self):
        """
        :return: self._sc_ket
        """
        return self._sc_ket


class Sample:
    """
    Interface class to FORTRAN backend.

    Attributes
    ----------
    sc_list: list of 'SuperCell' instances
        list of super cells within the sample
    hop_list: list of 'IntraHopping' instances
        list of inter-hopping sets between super cells within the sample
    orb_eng: (num_orb_sc,) float64 array
        on-site energies of orbitals in the super cell in eV
    orb_pos: (num_orb_sc, 3) float64 array
        Cartesian coordinates of orbitals in the super cell in nm
    hop_i: (num_hop_sc,) int64 array
        row indices of hopping terms reduced by conjugate relation
    hop_j: (num_hop_sc,) int64 array
        column indices of hopping terms reduced by conjugate relation
    hop_v: (num_hop_sc,) complex128 array
        energies of hopping terms in accordance with hop_i and hop_j in eV
    dr: (num_hop_sc, 3) float64 array
        distances of hopping terms in accordance with hop_i and hop_j in nm
    rescale: float
        rescaling factor for the Hamiltonian
        reserved for compatibility with old version of TBPlaS

    NOTES
    -----
    If periodic conditions are enabled, orbital indices in hop_j may be wrapped
    back if it falls out of the super cell. Nevertheless, the distances in dr
    are still the ones before wrapping. This is essential for adding magnetic
    field, calculating band structure and many properties involving dx and dy.
    """
    def __init__(self, *args: Union[SuperCell, InterHopping]):
        """
        :param args: list of 'SuperCell' or 'IntraHopping' instances
            super cells and inter-hopping sets within this sample
        :returns: None
        :raises SampleVoidError: if len(args) == 0
        :raises SampleCompError: if any argument in args is not instance of
            'SuperCell' or 'InterHopping' classes
        :raises SampleClosureError: if any 'InterHopping' instance has super
            cells not included in the sample
        """
        # Check arguments
        if len(args) == 0:
            raise exc.SampleVoidError()

        # Parse super cells and inter-hopping terms
        self.sc_list = []
        self.hop_list = []
        for i_arg, arg in enumerate(args):
            if isinstance(arg, SuperCell):
                self.sc_list.append(arg)
            elif isinstance(arg, InterHopping):
                self.hop_list.append(arg)
                arg.lock()
            else:
                raise exc.SampleCompError(i_arg)

        # Check closure of inter-hopping instances
        for i_h, hop in enumerate(self.hop_list):
            if hop.sc_bra not in self.sc_list:
                raise exc.SampleClosureError(i_h, "sc_bra")
            if hop.sc_ket not in self.sc_list:
                raise exc.SampleClosureError(i_h, "sc_ket")

        # Declare arrays
        # The actual initialization will be done on-demand in init_* methods.
        self.orb_eng = None
        self.orb_pos = None
        self.hop_i = None
        self.hop_j = None
        self.hop_v = None
        self.dr = None
        self.rescale = 1.0

    def __get_num_orb(self):
        """
        Get numbers of orbitals in each super cell.

        :return: num_orb: list of integers
            numbers of orbitals in each super cell
        """
        num_orb = [sc.num_orb_sc for sc in self.sc_list]
        return num_orb

    def __get_ind_start(self):
        """
        Get starting indices of orbitals for each super cell for assembling
        hopping terms and distances.

        :return: ind_start: list of integers
            starting indices of orbitals for each super cell
        """
        num_orb = self.__get_num_orb()
        ind_start = [sum(num_orb[:_]) for _ in range(len(num_orb))]
        return ind_start

    def init_orb_eng(self, force_init=False):
        """
        Initialize self.orb_eng on demand.

        If self.orb_eng is None, build it from scratch. Otherwise, build it
        only when force_init is True.

        :param force_init: boolean
            whether to force initializing the array from scratch even if it
            has already been initialized
        :returns: None
            self.orb_eng is modified.
        """
        if force_init or self.orb_eng is None:
            orb_eng = [sc.get_orb_eng() for sc in self.sc_list]
            self.orb_eng = np.concatenate(orb_eng)

    def init_orb_pos(self, force_init=False):
        """
        Initialize self.orb_pos on demand.

        If self.orb_pos is None, build it from scratch. Otherwise, build it
        only when force_init is True.

        :param force_init: boolean
            whether to force initializing the array from scratch even if it
            has already been initialized
        :returns: None
            self.orb_pos is modified.
        """
        if force_init or self.orb_pos is None:
            orb_pos = [sc.get_orb_pos() for sc in self.sc_list]
            self.orb_pos = np.concatenate(orb_pos)

    def init_hop(self, force_init=False):
        """
        Initialize self.hop_i, self.hop_j, self.hop_v and reset self.rescale
        on demand.

        If the arrays are None, build them from scratch. Otherwise, build them
        only when force_init is True.

        :param force_init: boolean
            whether to force initializing the arrays from scratch even if they
            have already been initialized
        :return: None
            self.hop_i, self.hop_j and self.hop_j are modified.
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        if force_init or self.hop_i is None:
            hop_i_tot, hop_j_tot, hop_v_tot = [], [], []
            ind_start = self.__get_ind_start()

            # Collect hopping terms from each super cell
            for i_sc, sc in enumerate(self.sc_list):
                hop_i, hop_j, hop_v = sc.get_hop()
                hop_i += ind_start[i_sc]
                hop_j += ind_start[i_sc]
                hop_i_tot.append(hop_i)
                hop_j_tot.append(hop_j)
                hop_v_tot.append(hop_v)

            # Collect hopping terms from each inter hopping set
            for hop in self.hop_list:
                hop_i, hop_j, hop_v = hop.get_hop()
                hop_i += ind_start[self.sc_list.index(hop.sc_bra)]
                hop_j += ind_start[self.sc_list.index(hop.sc_ket)]
                hop_i_tot.append(hop_i)
                hop_j_tot.append(hop_j)
                hop_v_tot.append(hop_v)

            # Assemble hopping terms
            self.hop_i = np.concatenate(hop_i_tot)
            self.hop_j = np.concatenate(hop_j_tot)
            self.hop_v = np.concatenate(hop_v_tot)

    def init_dr(self, force_init=False):
        """
        Initialize self.dr on demand.

        If self.dr is None, build it from scratch. Otherwise, build it only when
        force_init is True.

        :param force_init: boolean
            whether to force initializing the array from scratch even if it
            has already been initialized
        :returns: None
            self.dr is modified.
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        if force_init or self.dr is None:
            dr_tot = []
            for sc in self.sc_list:
                dr_tot.append(sc.get_dr())
            for hop in self.hop_list:
                dr_tot.append(hop.get_dr())
            self.dr = np.concatenate(dr_tot)

    def reset_array(self, force_reset=False):
        """
        Reset all modifications to self.orb_*, self.hop_* and self.dr.

        :param force_reset: boolean
            whether to force resetting the arrays even if they have not been
            initialized or modified
        :return: None
            self.orb_*, self.hop_* and self.dr are modified.
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        if force_reset or self.orb_eng is not None:
            self.init_orb_eng(force_init=True)
        if force_reset or self.orb_pos is not None:
            self.init_orb_pos(force_init=True)
        if force_reset or self.hop_i is not None:
            self.init_hop(force_init=True)
        if force_reset or self.dr is not None:
            self.init_dr(force_init=True)

    def rescale_ham(self, factor=None):
        """
        Rescale orbital energies and hopping terms.

        Reserved for compatibility with old version of TBPlaS.

        :param factor: float
            rescaling factor
            All orbital energies and hopping terms will be divided by this
            factor w.r.t. their original values in primitive cell. So only
            the last call to this method will take effect.
            Choose it such that the absolute value of the largest eigenvalue
            is smaller than 1 after rescaling. If no value is chosen,
            a reasonable value will be estimated from the Hamiltonian.
        :returns: None
            self.orb_eng, self.hop_v and self.rescale are modified.
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        self.init_orb_eng()
        self.init_hop()
        if factor is None:
            factor = core.get_rescale(self.orb_eng, self.hop_i, self.hop_j,
                                      self.hop_v)
        self.orb_eng /= (factor / self.rescale)
        self.hop_v /= (factor / self.rescale)
        self.rescale = factor

    def set_magnetic_field(self, intensity):
        """
        Apply magnetic field perpendicular to xOy-plane via Peierls substitution.

        The gauge-invariant Peierls phase from R_i to R_j is evaluated as
            exp[-1j * e/(2*h_bar*c) * (A(R_i) - A(R_j)) * (R_i + R_j)].
        With Landau gauge we have A(R) = (0, B*R_x, 0), then Peierls phase
        becomes
            exp[-1j * e/(2*h_bar*c) * B * (R_ix - R_jx) * (R_iy + R_jy)],
        or
            exp[1j * e/(2*h_bar*c) * B * (R_jx - R_ix) * (R_iy + R_jy)].

        Reference:
        https://journals.aps.org/prb/pdf/10.1103/PhysRevB.51.4940
        https://journals.jps.jp/doi/full/10.7566/JPSJ.85.074709
        Note that the elementary charge may have different signs in these
        papers.

        :param intensity: float
            magnetic B field in Tesla
        :return: None
            self.hop_v is modified.
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        self.init_orb_pos()
        self.init_hop()
        self.init_dr()
        core.set_mag_field(self.hop_i, self.hop_j, self.hop_v, self.dr,
                           self.orb_pos, intensity)

    def build_ham_csr(self):
        """
        Build sparse Hamiltonian for DOS and LDOS calculations using TBPM.

        :return: ham_scr
            sparse Hamiltonian matrix in CSR format
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        self.init_orb_eng()
        self.init_hop()
        ham_size = sum(self.__get_num_orb())
        ham_shape = (ham_size, ham_size)
        ham_dia = dia_matrix((self.orb_eng, 0), shape=ham_shape)
        ham_half = csr_matrix((self.hop_v, (self.hop_i, self.hop_j)),
                              shape=ham_shape)
        ham_csr = ham_dia + ham_half + ham_half.getH()
        return ham_csr

    def build_dxy_csr(self):
        """
        Build sparse dx and dy matrices in CSR format for TESTING purposes.

        NOTE: As zero elements are removed automatically when creating sparse
        matrices using scipy.sparse, Hamiltonian and dx/dy may have different
        indices and indptr. So, DO NOT use this method to generate dx and dy
        for TBPM calculations. Use the 'build_ham_dxy' method instead.

        :return: dx_csr, dy_csr
            sparse dx an dy matrices in CSR format
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        self.init_dr()
        dx, dy = self.dr[:, 0], self.dr[:, 1]
        mat_size = sum(self.__get_num_orb())
        shape = (mat_size, mat_size)
        dx_csr = csr_matrix((dx, (self.hop_i, self.hop_j)), shape)
        dy_csr = csr_matrix((dy, (self.hop_i, self.hop_j)), shape)
        dx_csr = dx_csr - dx_csr.getH()
        dy_csr = dy_csr - dy_csr.getH()
        return dx_csr, dy_csr

    def build_ham_dxy(self, orb_eng_cutoff=1.0e-5, algo="fast", sort_col=False):
        """
        Build the arrays for conductivity calculations using TBPM.

        NOTE: Two algorithms are implemented to build the arrays. "fast" is more
        efficient and uses less memory, but relies heavily on delicate pointer
        operations. "safe" is built on numpy functions and considered to be more
        robust, but uses more memory and twice slower than "fast". In short, use
        "fast" for production runs and "safe" for testing purposes.

        :param orb_eng_cutoff: float
            cutoff for orbital energies in eV
            Orbital energies below this value will be dropped when constructing
            sparse Hamiltonian.
        :param algo: string, should be "fast" or "safe"
            specifies which core function to call to generate the arrays
            Use "fast" for production runs and "safe" for testing purposes.
        :param sort_col: boolean
            whether to sort column indices and data of CSR matrices after
            creation, for TESTING purposes
            Sorting the columns will take much time, yet does not boost TBPM
            calculations. DO NOT enable this option for production runs.
        :return: indptr, int64 array
        :return: indices, int64 array
        :return: hop, complex128 array
        :return: dx, float64 array
        :return: dy, float64 array
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        self.init_orb_eng()
        self.init_orb_pos()
        self.init_hop()
        self.init_dr()
        if algo == "fast":
            indptr, indices, hop, dx, dy = \
                core.build_ham_dxy_fast(self.orb_eng, self.hop_i, self.hop_j,
                                        self.hop_v, self.dr, orb_eng_cutoff)
        else:
            indptr, indices, hop, dx, dy = \
                core.build_ham_dxy_safe(self.orb_eng, self.hop_i, self.hop_j,
                                        self.hop_v, self.dr, orb_eng_cutoff)
        if sort_col:
            core.sort_col_csr(indptr, indices, hop, dx, dy)
        return indptr, indices, hop, dx, dy

    def plot(self, fig_name=None, fig_dpi=300, with_orbitals=True,
             with_cells=True, hop_as_arrows=True, hop_eng_cutoff=1e-5,
             view="ab"):
        """
        Plot lattice vectors, orbitals, and hopping terms.

        If figure name is give, save the figure to file. Otherwise, show it on
        the screen.

        :param fig_name: string
            file name to which the figure will be saved
        :param fig_dpi: integer
            resolution of the figure file
        :param with_orbitals: boolean
            whether to plot orbitals as filled circles
        :param with_cells: boolean
            whether to plot borders of primitive cells
        :param hop_as_arrows: boolean
            whether to plot hopping terms as arrows
            If true, hopping terms will be plotted as arrows using axes.arrow()
            method. Otherwise, they will be plotted as lines using
            LineCollection. The former is more intuitive but much slower.
        :param hop_eng_cutoff: float
            cutoff for showing hopping terms
            Hopping terms with absolute energy below this value will not be
            shown in the plot.
        :param view: string
            kind of view point
            should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
        :return: None
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        :raises ValueError: if view is illegal
        """
        fig, axes = plt.subplots()
        axes.set_aspect('equal')

        # Plot super cells and hopping terms
        for sc in self.sc_list:
            sc.plot(axes, with_orbitals=with_orbitals, with_cells=with_cells,
                    hop_as_arrows=hop_as_arrows, hop_eng_cutoff=hop_eng_cutoff,
                    view=view)
        for hop in self.hop_list:
            hop.plot(axes, hop_as_arrows=hop_as_arrows,
                     hop_eng_cutoff=hop_eng_cutoff, view=view)

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

    def calc_bands(self, k_path, solver="lapack", num_bands=None):
        """
        Calculate band structure along given k_path.

        :param k_path: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points along given path
        :param solver: string
            mathematical library to obtain the eigen spectrum
            should be either "lapack" or "arpack"
        :param num_bands: integer
            number of bands to calculate, ignored for lapack solver
            For arpack solver, the lowest num_bands bands will be calculated.
            If not given, default value is to calculate the lowest 60% of all
            bands.
        :return: k_len: (num_kpt,) float64 array
            length of k-path in 1/nm in reciprocal space
            x-data for plotting band structure
        :return: bands: (num_kpt, num_orb_sc) float64 array
            energies corresponding to k-points in eV
        :raises SolverError: if solver is neither lapack nor arpack
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        self.init_orb_pos()
        self.init_orb_eng()
        self.init_hop()
        self.init_dr()

        # Initialize working arrays.
        num_k_points = len(k_path)
        num_orbitals = sum(self.__get_num_orb())
        ham_shape = (num_orbitals, num_orbitals)
        if solver == "lapack":
            num_bands = num_orbitals
        elif solver == "arpack":
            if num_bands is None:
                num_bands = int(num_orbitals * 0.6)
        else:
            raise exc.SolverError(solver)
        bands = np.zeros((num_k_points, num_bands), dtype=np.float64)
        hop_k = np.zeros(self.hop_v.shape, dtype=np.complex128)

        # Get length of k-path in reciprocal space
        sc0 = self.sc_list[0]
        k_len = kpt.gen_kdist(sc0.sc_lat_vec, k_path)

        # Convert k_path to Cartesian Coordinates
        sc_recip_vec = lat.gen_reciprocal_vectors(sc0.sc_lat_vec)
        k_path = lat.frac2cart(sc_recip_vec, k_path)

        # Loop over k-points to evaluate the band structure
        if solver == "lapack":
            ham_dense = np.zeros(ham_shape, dtype=np.complex128)
            for i_k, k_point in enumerate(k_path):
                # Update hop_k
                core.build_hop_k(self.hop_v, self.dr, k_point, hop_k)

                # Fill dense Hamiltonian
                ham_dense *= 0.0
                core.fill_ham(self.orb_eng, self.hop_i, self.hop_j, hop_k,
                              ham_dense)

                # Evaluate eigenvalues and eigenstates.
                eigenvalues, eigenstates, info = lapack.zheev(ham_dense)
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                bands[i_k, :] = eigenvalues
        elif solver == "arpack":
            ham_dia = dia_matrix((self.orb_eng, 0), shape=ham_shape)
            for i_k, k_point in enumerate(k_path):
                # Update hop_k
                core.build_hop_k(self.hop_v, self.dr, k_point, hop_k)

                # Create Hamiltonian
                ham_half = csr_matrix((hop_k, (self.hop_i, self.hop_j)),
                                      shape=ham_shape)
                ham_csr = ham_dia + ham_half + ham_half.getH()

                # Evaluate eigenvalues and eigenstates.
                eigenvalues, eigenstates = eigsh(ham_csr, num_bands, which="LA")
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                bands[i_k, :] = eigenvalues
        else:
            raise exc.SolverError(solver)
        return k_len, bands

    def calc_dos(self, k_points, e_min=None, e_max=None, e_step=0.05,
                 sigma=0.05, basis="Gaussian", **kwargs):
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
        :param kwargs: dictionary
            arguments for method 'calc_bands'
        :return: energies: (num_grid,) float64 array
            energy grid corresponding to e_min, e_max and e_step
        :return: dos: (num_grid,) float64 array
            density of states in states/eV
        :raises BasisError: if basis is neither Gaussian or Lorentzian
        :raises SolverError: if solver is neither lapack nor arpack
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            hop_modifier of any super cell or in any inter hopping set is out
            of range
        :raises IDPCVacError: if bra or ket in hop_modifier of any super cell
            or in any inter-hopping set corresponds to a vacancy
        """
        # Get the band energies
        k_len, bands = self.calc_bands(k_points, **kwargs)
    
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
            for eng_k in bands:
                for eng_i in eng_k:
                    dos += _gaussian(energies, eng_i)
        elif basis == "Lorentzian":
            for eng_k in bands:
                for eng_i in eng_k:
                    dos += _lorentzian(energies, eng_i)
        else:
            raise exc.BasisError(basis)
    
        # Re-normalize dos
        # For each energy in bands, we use a normalized Gaussian or Lorentzian
        # basis function to approximate the Delta function. Totally, there are
        # bands.size basis functions. So we divide dos by this number.
        dos /= bands.size
        return energies, dos

    @property
    def energy_range(self):
        """
        Get energy range to consider in calculations.

        Reserved for compatibility with old version of TBPlaS.

        :returns: en_range : float
            All eigenvalues are between (-en_range / 2, en_range / 2) in eV.
        """
        en_range = 2.0 * self.rescale
        return en_range

    @property
    def area_unit_cell(self):
        """
        Get the area formed by a1 and a2 of the primitive cell.

        Reserved for compatibility with old version of TBPlaS.

        :return: area, float
            area formed by a1 and a2 in NM^2
        """
        sc0 = self.sc_list[0]
        return sc0.prim_cell.get_lattice_area()

    @property
    def volume_unit_cell(self):
        """
        Get the volume of primitive cell.

        Reserved for compatibility with old version of TBPlaS.

        :return: volume: float
            volume of primitive cell in NM^3
        """
        sc0 = self.sc_list[0]
        return sc0.prim_cell.get_lattice_volume()

    @property
    def extended(self):
        """
        Get the number of extended times of primitive cell.

        Reserved for compatibility with old version of TBPlaS.

        :return: extended: integer
            number of extended times of primitive cells
        """
        sc0 = self.sc_list[0]
        return sc0.prim_cell.extended

    @property
    def nr_orbitals(self):
        """
        Get the number of orbitals of primitive cell.

        Reserved for compatibility with old version of TBPlaS.

        :return: nr_orbitals: integer
            number of orbitals in the primitive cell.
        """
        sc0 = self.sc_list[0]
        return sc0.prim_cell.num_orb
