"""
Functions and classes for sample.

Functions
---------
    None.

Classes
-------
    SCInterHopping: user class
        container class for hopping terms between different supercells within
        the sample.
    Sample: user class
        interface class to FORTRAN backend
"""

from typing import Union, Tuple

import numpy as np
from scipy.sparse import dia_matrix, csr_matrix
import matplotlib.pyplot as plt

from . import lattice as lat
from . import exceptions as exc
from . import core
from .base import Lockable, InterHopping
from .super import SuperCell
from .utils import ModelViewer
from ..diagonal import DiagSolver


class SCInterHopping(Lockable, InterHopping):
    """
    Container class for hopping terms between different supercells within the
    sample.

    Attributes
    ----------
    sc_bra: instance of 'SuperCell' class
        the 'bra' supercell from which the hopping terms exist
    sc_ket: instance of 'SuperCell' class
        the 'ket' supercell to which the hopping terms exist

    NOTES
    -----
    1. Reduction

    Since inter-hopping terms exist within different super-cells, there is no
    need to reduce them according to the conjugate relation.

    2. Rules

    We restrict hopping terms to be from the (0, 0, 0) 'bra' supercell to any
    'ket' supercell. The counterparts are restored via the conjugate relation:
        <bra, R0, i|H|ket, Rn, j> = <ket, R0, j|H|bra, -Rn, i>*
    """
    def __init__(self, sc_bra: SuperCell, sc_ket: SuperCell):
        """
        :param sc_bra: instance of 'SuperCell' class
            the 'bra' supercell from which the hopping terms exist
        :param sc_ket: instance of 'SuperCell' class
            the 'ket' supercell to which the hopping terms exist
        """
        Lockable.__init__(self)
        InterHopping.__init__(self)
        self.sc_bra = sc_bra
        self.sc_ket = sc_ket

    def check_lock(self):
        """Check lock state of this instance."""
        if self.is_locked:
            raise exc.InterHopLockError()

    def add_hopping(self, rn: tuple, orb_i: int, orb_j: int, energy: complex):
        """
        Add a new hopping term or update existing term.

        :param tuple rn: (r_a, r_b, r_c), cell index
        :param int orb_i: orbital index or bra
        :param int orb_j: orbital index of ket
        :param complex energy: hopping energy
        :return: None
        :raises InterHopLockError: is the object is locked
        """
        self.check_lock()
        super().add_hopping(rn, orb_i, orb_j, energy)

    def get_hop(self, check_dup=False):
        """
        Get hopping indices and energies.

        :param check_dup: boolean
            whether to check for duplicate hopping terms in the results
        :return: hop_i: (num_hop,) int64 array
            row indices of hopping terms
        :return: hop_j: (num_hop,) int64 array
            column indices of hopping terms
        :return: hop_v: (num_hop,) complex128 array
            energies of hopping terms in accordance with hop_i and hop_j in eV
        :raises InterHopVoidError: if no hopping terms have been added to the
            instance
        :raises ValueError: if duplicate terms have been detected
        """
        if self.num_hop == 0:
            raise exc.InterHopVoidError()
        hop_ind, hop_eng = self.to_array(use_int64=True)
        hop_i, hop_j = hop_ind[:, 3], hop_ind[:, 4]
        hop_v = hop_eng

        # Check for duplicate terms
        if check_dup:
            for ih in range(hop_i.shape[0]):
                ii, jj = hop_i.item(ih), hop_j.item(ih)
                if self.count_pair(ii, jj) > 1:
                    raise ValueError(f"Duplicate terms detected {ii} {jj}")
        return hop_i, hop_j, hop_v

    def get_dr(self):
        """
        Get hopping distances.

        NOTE: If periodic conditions are enabled, orbital indices in hop_j may
        be wrapped back if it falls out of the supercell. Nevertheless, the
        distances in dr are still the ones before wrapping. This is essential
        for adding magnetic field, calculating band structure and many
        properties involving dx and dy.

        :return: dr: (num_hop, 3) float64 array
            distances of hopping terms in accordance with hop_i and hop_j in nm
        :raises InterHopVoidError: if no hopping terms have been added to the
            instance
        """
        if self.num_hop == 0:
            raise exc.InterHopVoidError()
        hop_ind, hop_eng = self.to_array(use_int64=True)
        pos_bra = self.sc_bra.get_orb_pos()
        pos_ket = self.sc_ket.get_orb_pos()
        dr = core.build_inter_dr(hop_ind, pos_bra, pos_ket,
                                 self.sc_ket.sc_lat_vec)
        return dr

    def plot(self, axes: plt.Axes, hop_as_arrows=True, hop_eng_cutoff=1e-5,
             hop_color="r", view="ab"):
        """
        Plot hopping terms to axes.

        :param axes: instance of matplotlib 'Axes' class
            axes on which the figure will be plotted
        :param hop_as_arrows: boolean
            whether to plot hopping terms as arrows
        :param hop_eng_cutoff: float
            cutoff for showing hopping terms.
            Hopping terms with absolute energy below this value will not be
            shown in the plot.
        :param hop_color: string
            color of hopping terms
        :param view: string
            kind of view point
            should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
        :return: None.
        :raises InterHopVoidError: if no hopping terms have been added to the
            instance
        :raises IDPCIndexError: if cell or orbital index of bra or ket in
            self.indices is out of range
        :raises IDPCVacError: if bra or ket in self.indices corresponds
            to a vacancy
        :raises ValueError: if view is illegal
        """
        viewer = ModelViewer(axes, self.sc_bra.pc_lat_vec, view)

        # Plot hopping terms
        orb_pos_i = self.sc_bra.get_orb_pos()
        # orb_pos_j = self.sc_ket.get_orb_pos()
        hop_i, hop_j, hop_v = self.get_hop()
        dr = self.get_dr()
        for i_h in range(hop_i.shape[0]):
            if abs(hop_v.item(i_h)) >= hop_eng_cutoff:
                pos_i = orb_pos_i[hop_i.item(i_h)]
                # pos_j = orb_pos_j[hop_j.item(i_h)]
                pos_j = pos_i + dr[i_h]
                if hop_as_arrows:
                    viewer.plot_arrow(pos_i, pos_j, color=hop_color,
                                      length_includes_head=True,
                                      width=0.002, head_width=0.02, fill=False)
                else:
                    viewer.add_line(pos_i, pos_j)
        if not hop_as_arrows:
            viewer.plot_line(color=hop_color)


class Sample:
    """
    Interface class to FORTRAN backend.

    Attributes
    ----------
    sc_list: list of 'SuperCell' instances
        list of supercells within the sample
    hop_list: list of 'IntraHopping' instances
        list of inter-hopping sets between supercells within the sample
    orb_eng: (num_orb_sc,) float64 array
        on-site energies of orbitals in the supercell in eV
    orb_pos: (num_orb_sc, 3) float64 array
        Cartesian coordinates of orbitals in the supercell in nm
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
    back if it falls out of the supercell. Nevertheless, the distances in dr
    are still the ones before wrapping. This is essential for adding magnetic
    field, calculating band structure and many properties involving dx and dy.
    """
    def __init__(self, *args: Union[SuperCell, SCInterHopping]):
        """
        :param args: list of 'SuperCell' or 'IntraHopping' instances
            supercells and inter-hopping sets within this sample
        :returns: None
        :raises SampleVoidError: if len(args) == 0
        :raises SampleCompError: if any argument in args is not instance of
            'SuperCell' or 'SCInterHopping' classes
        :raises SampleClosureError: if any 'SCInterHopping' instance has
            supercells not included in the sample
        """
        # Check arguments
        if len(args) == 0:
            raise exc.SampleVoidError()

        # Parse supercells and inter-hopping terms
        self.sc_list = []
        self.hop_list = []
        for i_arg, arg in enumerate(args):
            if isinstance(arg, SuperCell):
                self.sc_list.append(arg)
            elif isinstance(arg, SCInterHopping):
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

    def _get_num_orb(self):
        """
        Get numbers of orbitals in each supercell.

        :return: num_orb: list of integers
            numbers of orbitals in each supercell
        """
        num_orb = [sc.num_orb_sc for sc in self.sc_list]
        return num_orb

    def _get_ind_start(self):
        """
        Get starting indices of orbitals for each supercell for assembling
        hopping terms and distances.

        :return: ind_start: list of integers
            starting indices of orbitals for each supercell
        """
        num_orb = self._get_num_orb()
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
            self.hop_i, self.hop_j, self.hop_j and self.rescale are modified.
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        if force_init or self.hop_i is None:
            hop_i_tot, hop_j_tot, hop_v_tot = [], [], []
            ind_start = self._get_ind_start()

            # Collect hopping terms from each supercell
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

            # Reset scaling factor
            self.rescale = 1.0

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
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        if force_init or self.dr is None:
            dr_tot = []
            for sc in self.sc_list:
                dr_tot.append(sc.get_dr())
            for hop in self.hop_list:
                dr_tot.append(hop.get_dr())
            self.dr = np.concatenate(dr_tot)

    def reset_array(self):
        """
        Reset all modifications to self._orb_*, self._hop_* and self.dr.

        :return: None
            self._orb_*, self._hop_* and self.dr are modified.
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        if self.orb_eng is not None:
            self.init_orb_eng(force_init=True)
        if self.orb_pos is not None:
            self.init_orb_pos(force_init=True)
        if self.hop_i is not None:
            self.init_hop(force_init=True)
        if self.dr is not None:
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
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        self.init_orb_eng()
        self.init_hop()
        if factor is None:
            factor = core.get_rescale(self.orb_eng, self.hop_i, self.hop_j,
                                      self.hop_v)
        self.orb_eng /= (factor / self.rescale)
        self.hop_v /= (factor / self.rescale)
        self.rescale = factor

    def set_magnetic_field(self, intensity, gauge=0):
        """
        Apply magnetic field perpendicular to xOy-plane to -z direction via
        Peierls substitution.

        The gauge-invariant Peierls phase from R_i to R_j is evaluated as
            exp[-1j * |e|/(2*h_bar*c) * (R_i - R_j) * (A_i + A_j)]
        or
            exp[1j * |e|/(2*h_bar*c) * (R_j - R_i) * (A_j + A_i)]
        For the reference, see eqn. 10-11 of ref [1] and eqn. 19-20 of ref [2].
        Be aware that ref [1] uses |e| while ref [2] uses e with sign.

        For perpendicular magnetic pointing to +z, we have A = (-By, 0, 0). Then
            (R_j - R_i) * (A_j + A_i) = -B * (y_j + y_i) * (x_j - x_i)
        However, there isn't the minus sign in the source code of TBPLaS. This is
        because we are considering a magnetic field pointing to -z.

        Note that the above formulae use Gaussian units. For SI units, the speed
        of light vanishes. This is verified by checking the dimension under SI
        units:
            [e]/[h_bar] = IT/(L^2 M T^-1) = L^-2 M^-1 I T^-2
            [R][A] = L * (M T^-2 I^-1 L) = L^2 M I^-1 T^-2
        which cancel out upon multiplication. Similarly, under Gaussian units
        we have:
        [e]/[c*h_bar] = L^(-3/2) M^(-1/2) T
        [R][A] = L^(3/2) M^(1/2) T
        which also cancel out upon multiplication.

        The analysis also inspires to calculate the scaling factor in SI units
        as:
            |e/2h_bar| = 759633724404755.2
        However, we use nm for lengths. So the scaling factor should be divided
        by 1e18, yielding 0.0007596337244047553. That is exactly the magic number
        pi / 4135.666734 = 0.0007596339008078771 in the source code of TBPLaS.

        Reference:
        [1] https://journals.aps.org/prb/pdf/10.1103/PhysRevB.51.4940
        [2] https://journals.jps.jp/doi/full/10.7566/JPSJ.85.074709

        :param intensity: float
            magnetic B field in Tesla
        :param gauge: int
            gauge of vector potential which produces magnetic field to -z
            0 for (By, 0, 0), 1 for (0, -Bx, 0), 2 for (0.5By, -0.5Bx, 0)
        :return: None
            self.hop_v is modified.
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        :raises ValueError: if gauge is not in (0, 1, 2)
        """
        self.init_orb_pos()
        self.init_hop()
        self.init_dr()
        if gauge not in (0, 1, 2):
            raise ValueError(f"Illegal gauge {gauge}")
        core.set_mag_field(self.hop_i, self.hop_j, self.hop_v, self.dr,
                           self.orb_pos, intensity, gauge)

    def build_ham_csr(self):
        """
        Build sparse Hamiltonian for DOS and LDOS calculations using TBPM.

        :return: ham_scr
            sparse Hamiltonian matrix in CSR format
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        self.init_orb_eng()
        self.init_hop()
        ham_size = sum(self._get_num_orb())
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
            sparse dx and dy matrices in CSR format
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        self.init_dr()
        dx, dy = self.dr[:, 0], self.dr[:, 1]
        mat_size = sum(self._get_num_orb())
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
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
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

    def plot(self, fig_name=None, fig_size=None, fig_dpi=300,
             with_orbitals=True, with_cells=True,
             hop_as_arrows=True, hop_eng_cutoff=1e-5,
             sc_colors=None, hop_colors=None, view="ab"):
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
        :param hop_as_arrows: boolean
            whether to plot hopping terms as arrows
            If true, hopping terms will be plotted as arrows using axes.arrow()
            method. Otherwise, they will be plotted as lines using
            LineCollection. The former is more intuitive but much slower.
        :param hop_eng_cutoff: float
            cutoff for showing hopping terms.
            Hopping terms with absolute energy below this value will not be
            shown in the plot.
        :param sc_colors: List[str]
            colors for the hopping terms of each supercell
        :param hop_colors: List[str]
            colors for the hopping terms each inter-hopping container
        :param view: string
            kind of view point
            should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
        :return: None
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        :raises ValueError: if view is illegal
        """
        fig, axes = plt.subplots(figsize=fig_size)
        axes.set_aspect('equal')

        if sc_colors is None:
            sc_colors = ['r' for _ in range(len(self.sc_list))]
        if hop_colors is None:
            hop_colors = ['r' for _ in range(len(self.hop_list))]

        # Plot supercells and hopping terms
        for i, sc in enumerate(self.sc_list):
            sc.plot(axes, with_orbitals=with_orbitals, with_cells=with_cells,
                    hop_as_arrows=hop_as_arrows, hop_eng_cutoff=hop_eng_cutoff,
                    hop_color=sc_colors[i], view=view)
        for i, hop in enumerate(self.hop_list):
            hop.plot(axes,
                     hop_as_arrows=hop_as_arrows, hop_eng_cutoff=hop_eng_cutoff,
                     hop_color=hop_colors[i], view=view)

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

    def set_ham_dense(self, k_point: np.ndarray,
                      ham_dense: np.ndarray,
                      convention: int = 1) -> None:
        """
        Set up Hamiltonian for given k-point.

        This is the interface to be called by external exact solvers. The
        callers are responsible to call the 'init_*' method.

        :param k_point: (3,) float64 array
            Fractional coordinate of the k-point
        :param ham_dense: (num_orb, num_orb) complex128 array
            incoming Hamiltonian
        :param convention: convention for setting up the Hamiltonian
        :return: None
        :raises ValueError: if convention != 1
        """
        if convention not in (1,):
            raise ValueError(f"Illegal convention {convention}")

        # Convert k-point to Cartesian Coordinates
        sc0 = self.sc_list[0]
        sc_recip_vec = lat.gen_reciprocal_vectors(sc0.sc_lat_vec)
        k_point = np.matmul(k_point, sc_recip_vec)

        # Set up the Hamiltonian
        ham_dense *= 0.0
        hop_k = np.zeros(self.hop_v.shape, dtype=np.complex128)
        core.build_hop_k(self.hop_v, self.dr, k_point, hop_k)
        core.fill_ham(self.orb_eng, self.hop_i, self.hop_j, hop_k, ham_dense)

    def set_ham_csr(self, k_point: np.ndarray, convention: int = 1) -> csr_matrix:
        """
        Set up sparse Hamiltonian in csr format for given k-point.

        This is the interface to be called by external exact solvers. The
        callers are responsible to call the 'init_*' method.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param convention: convention for setting up the Hamiltonian
        :return: sparse Hamiltonian
        :raises ValueError: if convention != 1
        """
        if convention not in (1,):
            raise ValueError(f"Illegal convention {convention}")

        # Convert k-point to Cartesian Coordinates
        sc0 = self.sc_list[0]
        sc_recip_vec = lat.gen_reciprocal_vectors(sc0.sc_lat_vec)
        k_point = np.matmul(k_point, sc_recip_vec)

        # Diagonal terms
        ham_shape = (self.num_orb_tot, self.num_orb_tot)
        ham_dia = dia_matrix((self.orb_eng, 0), shape=ham_shape)

        # Off-diagonal terms
        hop_k = np.zeros(self.hop_v.shape, dtype=np.complex128)
        core.build_hop_k(self.hop_v, self.dr, k_point, hop_k)
        ham_half = csr_matrix((hop_k, (self.hop_i, self.hop_j)),
                              shape=ham_shape)

        # Sum up the terms
        ham_csr = ham_dia + ham_half + ham_half.getH()
        return ham_csr

    def calc_bands(self, k_points: np.ndarray,
                   enable_mpi: bool = False,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate band structure along given k_path.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of the k-points
        :param enable_mpi: whether to enable parallelization over k-points
            using mpi
        :param kwargs: arguments for 'calc_bands' of 'DiagSolver' class
        :return: (k_len, bands)
            k_len: (num_kpt,) float64 array in 1/NM
            length of k-path in reciprocal space, for plotting band structure
            bands: (num_kpt, num_orb) float64 array
            Energies corresponding to k-points in eV
        """
        diag_solver = DiagSolver(self, enable_mpi=enable_mpi)
        k_len, bands = diag_solver.calc_bands(k_points, **kwargs)[:2]
        return k_len, bands

    def calc_dos(self, k_points: np.ndarray,
                 enable_mpi: bool = False,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate density of states for given energy range and step.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of the k-points
        :param enable_mpi: whether to enable parallelization over k-points using mpi
        :param kwargs: arguments for 'calc_dos' of 'DiagSolver' class
        :return: (energies, dos)
            energies: (num_grid,) float64 array
            energy grid corresponding to e_min, e_max and e_step
            dos: (num_grid,) float64 array
            density of states in states/eV
        :raises BasisError: if basis is neither Gaussian nor Lorentzian
        """
        diag_solver = DiagSolver(self, enable_mpi=enable_mpi)
        energies, dos = diag_solver.calc_dos(k_points, **kwargs)
        return energies, dos

    @property
    def num_orb_tot(self):
        """
        Get the total number of orbitals of the sample.

        :return: num_orb_tot: int
            total number of orbitals
        """
        return sum(self._get_num_orb())

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
