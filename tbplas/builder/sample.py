"""Functions and classes for sample."""

import os
from typing import Union, Tuple, List, Callable

import numpy as np
from scipy.sparse import dia_matrix, csr_matrix
import matplotlib.pyplot as plt

from ..cython import sample as core
from . import exceptions as exc
from .base import Observable
from .super import SuperCell, SCInterHopping
from ..diagonal import DiagSolver


__all__ = ["Sample"]


ham_dxy_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
orb_color_type = Callable[[np.ndarray], List[str]]


class Sample(Observable):
    """
    Interface class to FORTRAN backend.

    Attributes
    ----------
    _sc_list: List[SuperCell]
        list of supercells within the sample
    _hop_list: List[SCInterHopping]
        list of inter-hopping sets between supercells within the sample
    orb_eng: (num_orb_tot,) float64 array
        on-site energies of orbitals in the supercell in eV
    orb_pos: (num_orb_tot, 3) float64 array
        Cartesian coordinates of orbitals in the supercell in nm
    hop_i: (num_hop_tot,) int64 array
        row indices of hopping terms reduced by conjugate relation
    hop_j: (num_hop_tot,) int64 array
        column indices of hopping terms reduced by conjugate relation
    hop_v: (num_hop_tot,) complex128 array
        energies of hopping terms in accordance with hop_i and hop_j in eV
    dr: (num_hop_tot, 3) float64 array
        distances of hopping terms in accordance with hop_i and hop_j in nm
    _rescale: float
        rescaling factor for the Hamiltonian
        reserved for compatibility with old version of TBPlaS

    NOTES
    -----
    If periodic conditions are enabled, orbital indices in hop_j may be wrapped
    back if it falls out of the supercell. Nevertheless, the distances in dr
    are still the ones before wrapping. This is essential for adding magnetic
    field, calculating band structure and many properties involving dx and dy.
    """
    def __init__(self, *args: Union[SuperCell, SCInterHopping]) -> None:
        """
        :param args: supercells and inter-hopping sets within this sample
        :returns: None
        :raises SampleVoidError: if len(args) == 0
        :raises SampleCompError: if any argument in args is not instance of
            'SuperCell' or 'SCInterHopping' classes
        :raises SampleClosureError: if any 'SCInterHopping' instance has
            supercells not included in the sample
        """
        super().__init__()

        # Check arguments
        if len(args) == 0:
            raise exc.SampleVoidError()

        # Parse supercells and inter-hopping terms
        self._sc_list = []
        self._hop_list = []
        for i_arg, arg in enumerate(args):
            if isinstance(arg, SuperCell):
                self._sc_list.append(arg)
            elif isinstance(arg, SCInterHopping):
                self._hop_list.append(arg)
            else:
                raise exc.SampleCompError(i_arg)
            arg.add_subscriber(f"sample #{id(self)}", self)
            arg.lock(f"sample #{id(self)}")

        # Check closure of inter-hopping instances
        for i_h, hop in enumerate(self._hop_list):
            if hop.sc_bra not in self._sc_list:
                raise exc.SampleClosureError(i_h, "sc_bra")
            if hop.sc_ket not in self._sc_list:
                raise exc.SampleClosureError(i_h, "sc_ket")

        # Declare arrays
        # The actual initialization will be done on-demand in init_* methods.
        self.orb_eng = None
        self.orb_pos = None
        self.hop_i = None
        self.hop_j = None
        self.hop_v = None
        self.dr = None
        self._rescale = 1.0

    def _get_num_orb_sc(self) -> List[int]:
        """
        Get numbers of orbitals in each supercell.

        :return: numbers of orbitals in each supercell
        """
        num_orb_sc = [sc.num_orb_sc for sc in self._sc_list]
        return num_orb_sc

    def _get_ind_start(self) -> List[int]:
        """
        Get starting indices of orbitals for each supercell for assembling
        hopping terms and distances.

        :return: starting indices of orbitals for each supercell
        """
        num_orb_sc = self._get_num_orb_sc()
        ind_start = [sum(num_orb_sc[:_]) for _ in range(len(num_orb_sc))]
        return ind_start

    def init_orb_eng(self, force_init: bool = False) -> None:
        """
        Initialize self.orb_eng on demand.

        If self.orb_eng is None, build it from scratch. Otherwise, build it
        only when force_init is True.

        :param force_init: whether to force initializing the array from scratch
            even if it has already been initialized
        :returns: None
        """
        if force_init or self.orb_eng is None:
            orb_eng = [sc.get_orb_eng() for sc in self._sc_list]
            self.orb_eng = np.concatenate(orb_eng)

    def init_orb_pos(self, force_init: bool = False) -> None:
        """
        Initialize self.orb_pos on demand.

        If self.orb_pos is None, build it from scratch. Otherwise, build it
        only when force_init is True.

        :param force_init: whether to force initializing the array from scratch
            even if it has already been initialized
        :returns: None
        """
        if force_init or self.orb_pos is None:
            orb_pos = [sc.get_orb_pos() for sc in self._sc_list]
            self.orb_pos = np.concatenate(orb_pos)

    def init_hop(self, force_init: bool = False) -> None:
        """
        Initialize hop_i, hop_j, hop_v, dr and reset rescale on demand.

        If the arrays are None, build them from scratch. Otherwise, build them
        only when force_init is True.

        :param force_init: whether to force initializing the arrays from scratch
         even if they have already been initialized
        :return: None
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        if force_init or self.hop_i is None:
            hop_i_tot, hop_j_tot, hop_v_tot, dr_tot = [], [], [], []
            ind_start = self._get_ind_start()

            # Collect hopping terms from each supercell
            for i_sc, sc in enumerate(self._sc_list):
                hop_i, hop_j, hop_v, dr = sc.get_hop()
                hop_i += ind_start[i_sc]
                hop_j += ind_start[i_sc]
                hop_i_tot.append(hop_i)
                hop_j_tot.append(hop_j)
                hop_v_tot.append(hop_v)
                dr_tot.append(dr)

            # Collect hopping terms from each inter hopping set
            for hop in self._hop_list:
                hop_i, hop_j, hop_v, dr = hop.get_hop()
                hop_i += ind_start[self._sc_list.index(hop.sc_bra)]
                hop_j += ind_start[self._sc_list.index(hop.sc_ket)]
                hop_i_tot.append(hop_i)
                hop_j_tot.append(hop_j)
                hop_v_tot.append(hop_v)
                dr_tot.append(dr)

            # Assemble hopping terms
            self.hop_i = np.concatenate(hop_i_tot)
            self.hop_j = np.concatenate(hop_j_tot)
            self.hop_v = np.concatenate(hop_v_tot)
            self.dr = np.concatenate(dr_tot)

            # Reset scaling factor
            self._rescale = 1.0

    def reset_array(self) -> None:
        """
        Reset all modifications to self._orb_*, self._hop_* and self.dr.

        :return: None
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
        for arg in self._sc_list:
            arg.lock(f"sample #{id(self)}")
        for arg in self._hop_list:
            arg.lock(f"sample #{id(self)}")

    def save_array(self, data_dir: str = "sample") -> None:
        """
        Save array attributes and scaling factor to disk.

        :param data_dir: directory to which data will be saved
        :return: None
        """
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if self.orb_eng is not None:
            np.save(f"{data_dir}/orb_eng", self.orb_eng)
        if self.orb_pos is not None:
            np.save(f"{data_dir}/orb_pos", self.orb_pos)
        if self.hop_i is not None:
            np.save(f"{data_dir}/hop_i", self.hop_i)
            np.save(f"{data_dir}/hop_j", self.hop_j)
            np.save(f"{data_dir}/hop_v", self.hop_v)
            np.save(f"{data_dir}/dr", self.dr)
            np.save(f"{data_dir}/rescale", self._rescale)

    def load_array(self, data_dir: str = "sample") -> None:
        """
        Load array attributes from disk.

        :param data_dir: directory in which data are saved
        :return: None
        """
        try:
            self.orb_eng = np.load(f"{data_dir}/orb_eng.npy")
        except FileNotFoundError:
            print(f"Ignoring {data_dir}/orb_eng.npy")
        try:
            self.orb_pos = np.load(f"{data_dir}/orb_pos.npy")
        except FileNotFoundError:
            print(f"Ignoring {data_dir}/orb_pos.npy")
        try:
            self.hop_i = np.load(f"{data_dir}/hop_i.npy")
            self.hop_j = np.load(f"{data_dir}/hop_j.npy")
            self.hop_v = np.load(f"{data_dir}/hop_v.npy")
            self.dr = np.load(f"{data_dir}/dr.npy")
            self._rescale = np.load(f"{data_dir}/rescale.npy").item()
        except FileNotFoundError:
            print(f"Ignoring {data_dir}/hop_i.npy")

    def update(self) -> None:
        """
        Interface for keeping data consistency in observer pattern.

        :return: None
        """
        self.reset_array()

    def rescale_ham(self, factor: float = None) -> None:
        """
        Rescale orbital energies and hopping terms.

        Reserved for compatibility with old version of TBPlaS.

        All orbital energies and hopping terms will be divided by the factor
        w.r.t. their original values in primitive cell. So only the last call
        to this method will take effect.

        Choose the factor such that the absolute value of the largest eigenvalue
        is smaller than 1 after rescaling. If no value is chosen, a reasonable
        value will be estimated from the Hamiltonian.

        :param factor: rescaling factor
        :returns: None
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        self.init_orb_eng()
        self.init_hop()
        if factor is None:
            factor = core.get_rescale(self.orb_eng, self.hop_i, self.hop_j,
                                      self.hop_v)
        self.orb_eng *= (self._rescale / factor)
        self.hop_v *= (self._rescale / factor)
        self._rescale = factor

    def set_magnetic_field(self, intensity: float, gauge: int = 0) -> None:
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

        :param intensity: magnetic B field in Tesla
        :param gauge: gauge of vector potential of the magnetic field to -z
            0 for (By, 0, 0), 1 for (0, -Bx, 0), 2 for (0.5By, -0.5Bx, 0)
        :return: None
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        :raises ValueError: if gauge is not in (0, 1, 2)
        """
        self.init_orb_pos()
        self.init_hop()
        if gauge not in (0, 1, 2):
            raise ValueError(f"Illegal gauge {gauge}")
        core.set_mag_field(self.hop_i, self.hop_j, self.hop_v, self.dr,
                           self.orb_pos, intensity, gauge)

    def set_k_point(self, k_point: np.ndarray) -> None:
        """
        Set the k-point of Hamiltonian.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :return: None
        """
        self.init_hop()

        # Convert k-point to Cartesian Coordinates
        sc_recip_vec = self.sc0.get_reciprocal_vectors()
        k_point = np.matmul(k_point, sc_recip_vec)

        # Set up the Hamiltonian
        hop_k = np.zeros(self.hop_v.shape, dtype=np.complex128)
        core.build_hop_k(self.hop_v, self.dr, k_point, hop_k)
        self.hop_v = hop_k

    def build_ham_csr(self) -> csr_matrix:
        """
        Build sparse Hamiltonian for DOS and LDOS calculations using TBPM.

        :return: sparse Hamiltonian matrix in CSR format
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        self.init_orb_eng()
        self.init_hop()
        ham_shape = (self.num_orb, self.num_orb)
        ham_dia = dia_matrix((self.orb_eng, 0), shape=ham_shape)
        ham_half = csr_matrix((self.hop_v, (self.hop_i, self.hop_j)),
                              shape=ham_shape)
        ham_csr = ham_dia + ham_half + ham_half.getH()
        return ham_csr

    def build_dxy_csr(self) -> Tuple[csr_matrix, csr_matrix]:
        """
        Build sparse dx and dy matrices in CSR format for TESTING purposes.

        NOTE: As zero elements are removed automatically when creating sparse
        matrices using scipy, Hamiltonian and dx/dy may have different indices
        and indptr. So, DO NOT use this method to generate dx and dy for TBPM
        calculations. Use the 'build_ham_dxy' method instead.

        :return: (dx_csr, dy_csr)
            sparse dx and dy matrices in CSR format
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        self.init_hop()
        dx, dy = self.dr[:, 0], self.dr[:, 1]
        shape = (self.num_orb, self.num_orb)
        dx_csr = csr_matrix((dx, (self.hop_i, self.hop_j)), shape)
        dy_csr = csr_matrix((dy, (self.hop_i, self.hop_j)), shape)
        dx_csr = dx_csr - dx_csr.getH()
        dy_csr = dy_csr - dy_csr.getH()
        return dx_csr, dy_csr

    def build_ham_dxy(self, orb_eng_cutoff: float = 1.0e-5,
                      algo: str = "fast",
                      sort_col: bool = False) -> ham_dxy_type:
        """
        Build the arrays for conductivity calculations using TBPM.

        NOTE: Two algorithms are implemented to build the arrays. "fast" is more
        efficient and uses less memory, but relies heavily on delicate pointer
        operations. "safe" is built on numpy functions and considered to be more
        robust, but uses more memory and twice slower than "fast". In short, use
        "fast" for production runs and "safe" for testing purposes.

        :param orb_eng_cutoff: cutoff for orbital energies in eV
            Orbital energies below this value will be dropped when constructing
            sparse Hamiltonian.
        :param algo: specifies which core function to call to generate the arrays
            Use "fast" for production runs and "safe" for testing purposes.
        :param sort_col: whether to sort column indices and data of CSR matrices
            after creation, for TESTING purposes
            Sorting the columns will take much time, yet does not boost TBPM
            calculations. DO NOT enable this option for production runs.
        :return: (indptr, indices, hop, dx, dy)
            indptr: int64 array
            indices: int64 array
            hop: complex128 array
            dx: float64 array
            dy: float64 array
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        """
        self.init_orb_eng()
        self.init_orb_pos()
        self.init_hop()
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

    def plot(self, fig_name: str = None,
             fig_size: Tuple[float, float] = None,
             fig_dpi: int = 300,
             sc_orb_colors: List[orb_color_type] = None,
             sc_hop_colors: List[str] = None,
             inter_hop_colors: List[str] = None,
             **kwargs) -> None:
        """
        Plot lattice vectors, orbitals, and hopping terms.

        If figure name is give, save the figure to file. Otherwise, show it on
        the screen.

        :param fig_name: file name to which the figure will be saved
        :param fig_size: size of the figure
        :param fig_dpi: resolution of the figure file
        :param sc_orb_colors:
        :param sc_hop_colors: colors for the hopping terms of each supercell
        :param inter_hop_colors: colors for the hopping terms each inter-hopping
            container
        :param kwargs: arguments for the 'plot' method of 'Super' and
            'SCInterHopping' classes
        :return: None
        :raises InterHopVoidError: if any inter-hopping set is empty
        :raises ValueError: if duplicate terms have been detected in any
            inter-hopping
        :raises ValueError: if view is illegal
        """
        fig, axes = plt.subplots(figsize=fig_size)
        axes.set_aspect('equal')

        if sc_orb_colors is None:
            sc_orb_colors = [None for _ in range(len(self._sc_list))]
        if sc_hop_colors is None:
            sc_hop_colors = ['r' for _ in range(len(self._sc_list))]
        if inter_hop_colors is None:
            inter_hop_colors = ['r' for _ in range(len(self._hop_list))]

        # Plot supercells and hopping terms
        for i, sc in enumerate(self._sc_list):
            sc.plot(axes, orb_color=sc_orb_colors[i],
                    hop_color=sc_hop_colors[i], **kwargs)
        for arg in ("with_orbitals", "with_cells"):
            if arg in kwargs.keys():
                kwargs.pop(arg)
        for i, hop in enumerate(self._hop_list):
            hop.plot(axes, hop_color=inter_hop_colors[i], **kwargs)

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
        callers are responsible to call the 'init_*' methods and initialize
        ham_dense as a zero matrix.

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
        sc_recip_vec = self.sc0.get_reciprocal_vectors()
        k_point = np.matmul(k_point, sc_recip_vec)

        # Set up the Hamiltonian
        # NOTE: DO NOT pass self_rescale to build_hop_k. Otherwise, the scaling
        # factor for self.orb_eng will be missing!
        ham_dense *= 0.0
        hop_k = np.zeros(self.hop_v.shape, dtype=np.complex128)
        core.build_hop_k(self.hop_v, self.dr, k_point, hop_k)
        core.fill_ham(self.orb_eng, self.hop_i, self.hop_j, hop_k, ham_dense)
        ham_dense *= self._rescale

    def set_ham_csr(self, k_point: np.ndarray,
                    convention: int = 1) -> csr_matrix:
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
        sc_recip_vec = self.sc0.get_reciprocal_vectors()
        k_point = np.matmul(k_point, sc_recip_vec)

        # Diagonal terms
        ham_shape = (self.num_orb, self.num_orb)
        ham_dia = dia_matrix((self.orb_eng, 0), shape=ham_shape)

        # Off-diagonal terms
        # NOTE: DO NOT pass self_rescale to build_hop_k. Otherwise, the scaling
        # factor for self.orb_eng will be missing!
        hop_k = np.zeros(self.hop_v.shape, dtype=np.complex128)
        core.build_hop_k(self.hop_v, self.dr, k_point, hop_k)
        ham_half = csr_matrix((hop_k, (self.hop_i, self.hop_j)),
                              shape=ham_shape)

        # Sum up the terms
        ham_csr = ham_dia + ham_half + ham_half.getH()
        ham_csr *= self._rescale
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
    def rescale(self) -> float:
        """
        Interface the '_rescale' attribute.

        :return: the rescaling factor
        """
        return self._rescale

    @property
    def num_orb(self) -> int:
        """
        Get the total number of orbitals of the sample.

        :return: total number of orbitals
        """
        return sum(self._get_num_orb_sc())

    @property
    def num_hop(self) -> int:
        """
        Get the total number of hopping terms of the sample.

        :return: total number of hopping terms
        """
        self.init_hop()
        return self.hop_i.shape[0]

    @property
    def sc0(self) -> SuperCell:
        """
        Interface for the 1st supercell.

        :return: the 1st supercell
        """
        return self._sc_list[0]

    @property
    def energy_range(self) -> float:
        """
        Get energy range to consider in calculations.

        Reserved for compatibility with old version of TBPlaS.

        All eigenvalues are between (-en_range / 2, en_range / 2) in eV.

        :returns: the energy range
        """
        en_range = 2.0 * self._rescale
        return en_range

    @property
    def area_unit_cell(self) -> float:
        """
        Get the area formed by a1 and a2 of the primitive cell.

        Reserved for compatibility with old version of TBPlaS.

        :return: area formed by a1 and a2 in NM^2
        """
        return self.sc0.prim_cell.get_lattice_area()

    @property
    def volume_unit_cell(self) -> float:
        """
        Get the volume of primitive cell.

        Reserved for compatibility with old version of TBPlaS.

        :return: volume of primitive cell in NM^3
        """
        return self.sc0.prim_cell.get_lattice_volume()

    @property
    def extended(self) -> float:
        """
        Get the number of extended times of primitive cell.

        Reserved for compatibility with old version of TBPlaS.

        :return: number of extended times of primitive cells
        """
        return self.sc0.prim_cell.extended

    @property
    def nr_orbitals(self) -> int:
        """
        Get the number of orbitals of primitive cell.

        Reserved for compatibility with old version of TBPlaS.

        :return: nr_orbitals: integer
            number of orbitals in the primitive cell.
        """
        return self.sc0.num_orb_pc
