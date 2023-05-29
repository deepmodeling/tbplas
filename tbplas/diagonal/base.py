"""Fundamentals of Solvers based on exact diagonalization."""

import math
from typing import Any, List, Tuple, Callable
from collections import namedtuple

import numpy as np
import scipy.linalg.lapack as lapack
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

from ..base import kpoints as kpt
from ..parallel import MPIEnv


__all__ = ["DiagSolver"]


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Gaussian type broadening function.

    :param x: incoming x
    :param mu: center of the Gaussian function
    :param sigma: half-width of the Gaussian function
    :return: normalized Gaussian function value at each x
    """
    part_a = 1.0 / (sigma * math.sqrt(2 * math.pi))
    part_b = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return part_a * part_b


def lorentzian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Lorentzian type broadening function.

    :param x: incoming x
    :param mu: center of the Lorentzian function
    :param sigma: half-width of the Lorentzian function
    :return: normalized Lorentzian function value at each x
    """
    part_a = 1.0 / (math.pi * sigma)
    part_b = sigma**2 / ((x - mu)**2 + sigma**2)
    return part_a * part_b


class DiagSolver(MPIEnv):
    """
    Base class for solvers based on exact diagonalization.

    NOTE: we declare the type of 'model' to be 'Any' for duck typing.

    Attributes
    ----------
    __model: 'PrimitiveCell' or 'Sample' instance
        model for which properties will be calculated
    __hk_dense: Callable[[np.ndarray, np.ndarray], None]
        user-defined function to set up the dense Hamiltonian in place, with the
        1st argument being the fractional coordinate of k-point and the 2nd
        argument being the Hamiltonian
    __hk_csr: Callable[[np.ndarray], csr_matrix]
        user-defined function to return the sparse Hamiltonian from the
        fractional coordinate of k-point as the 1st argument
    """
    def __init__(self, model: Any,
                 hk_dense: Callable[[np.ndarray, np.ndarray], None] = None,
                 hk_csr: Callable[[np.ndarray], csr_matrix] = None,
                 enable_mpi: bool = False,
                 echo_details: bool = True) -> None:
        """
        :param model: primitive cell or sample under study
        :param enable_mpi: whether to enable MPI-based parallelization
        :param echo_details: whether to output parallelization details
        """
        super().__init__(enable_mpi=enable_mpi, echo_details=echo_details)
        self.__model = model
        self.__hk_dense = hk_dense
        self.__hk_csr = hk_csr
        self.update_model()

    @property
    def model_is_pc(self) -> bool:
        """
        Check if the model is a primitive cell.

        :return: whether the model is a primitive cell
        """
        return hasattr(self.__model, "_orbital_list")

    @property
    def num_orb(self) -> int:
        """
        Get the number of orbitals in the model.

        :return: number of orbitals
        """
        return self.__model.num_orb

    @property
    def lat_vec(self) -> np.ndarray:
        """
        Get the lattice vectors of the model.

        :return: (3, 3) float64 array
            lattice vectors of in nm
        """
        if self.model_is_pc:
            return self.__model.lat_vec
        else:
            return self.__model.sc0.sc_lat_vec

    @property
    def recip_lat_vec(self) -> np.ndarray:
        """
        Get the reciprocal lattice vectors of the model in 1/nm.

        :return: (3, 3) float64 array
            reciprocal lattice vectors of in nm
        """
        if self.model_is_pc:
            return self.__model.get_reciprocal_vectors()
        else:
            return self.__model.sc0.get_reciprocal_vectors()

    def update_model(self) -> None:
        """
        Update the essential arrays of the model.

        :return: None
        """
        if self.model_is_pc:
            self.__model.sync_array()
        else:
            self.__model.init_orb_pos()
            self.__model.init_orb_eng()
            self.__model.init_hop()
            self.__model.init_dr()

    @staticmethod
    def _calc_proj(orbital_indices: List[int],
                   eigenstates: np.ndarray) -> np.ndarray:
        """
        Calculate the projection of eigenstates on given orbitals.

        :param orbital_indices: indices of the orbitals
        :param eigenstates: (num_bands, num_orb) complex128 array
            eigenstates to project, each ROW is an eigenstate
        :return: (num_bands,) float64 array
            projection of the eigenstates on given orbitals
        """
        num_bands = eigenstates.shape[0]
        proj_k = np.zeros(num_bands, dtype=np.float64)
        for i_b in range(num_bands):
            for i_o in orbital_indices:
                proj_k[i_b] += abs(eigenstates.item(i_b, i_o))**2
        return proj_k

    def set_ham_dense(self, k_point: np.ndarray,
                      ham_dense: np.ndarray,
                      convention: int = 1) -> None:
        """
        Set up dense Hamiltonian for given k-point.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param ham_dense: (num_orb, num_orb) complex128 array
            incoming Hamiltonian
        :param convention: convention for setting up the Hamiltonian
        :return: None
        :raises ValueError: if convention not in (1, 2)
        """
        if self.__hk_dense is not None:
            self.__hk_dense(k_point, ham_dense)
        else:
            self.__model.set_ham_dense(k_point, ham_dense, convention)

    def set_ham_csr(self, k_point: np.ndarray,
                    convention: int = 1) -> csr_matrix:
        """
        Set up sparse Hamiltonian in csr format for given k-point.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param convention: convention for setting up the Hamiltonian
        :return: sparse Hamiltonian
        :raises ValueError: if convention not in (1, 2)
        """
        if self.__hk_csr is not None:
            ham_csr = self.__hk_csr(k_point)
        else:
            ham_csr = self.__model.set_ham_csr(k_point, convention)
        return ham_csr

    def calc_bands(self, k_points: np.ndarray,
                   convention: int = 1,
                   orbital_indices: List[int] = None,
                   solver: str = "lapack",
                   num_bands: int = None,
                   **kwargs) -> namedtuple:
        """
        Calculate band structure along given k_path.

        The result is a named tuple with the following attributes:
        k_len: (num_kpt,) float64 array, length of k-path
        bands: (num_kpt, num_bands) float64 array, band structure
        proj: (num_kpt, num_bands) float64 array, projection on selected
        orbitals

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of the k-points
        :param convention: convention for setting up the Hamiltonian
        :param orbital_indices: indices of orbitals for evaluating projection,
            take all orbitals into consideration if no orbitals are specified
        :param solver: solver type, should be "lapack" or "arpack"
        :param num_bands: number of bands for arpack solver
        :param kwargs: arguments for the arpack solver
        :return: k_len, band structure and projection packed in named tuple
        :raises ValueError: if solver is neither lapack nor arpack
        """
        # Determine the shapes of arrays
        num_kpt = k_points.shape[0]
        num_orb = self.num_orb
        if solver == "lapack":
            num_bands = num_orb
        elif solver == "arpack":
            if num_bands is None:
                num_bands = int(num_orb * 0.6)
        else:
            raise ValueError(f"Illegal solver {solver}")

        # Initialize working arrays
        bands = np.zeros((num_kpt, num_bands), dtype=np.float64)
        if orbital_indices is not None:
            proj = np.zeros((num_kpt, num_bands), dtype=np.float64)
        else:
            proj = np.ones((num_kpt, num_bands), dtype=np.float64)
        if solver == "lapack":
            ham_dense = np.zeros((num_orb, num_orb), dtype=np.complex128)
        else:
            ham_dense = None

        # Distribute k-points over processes
        k_index = self.dist_range(num_kpt)

        # Calculate band structure
        for i_k in k_index:
            kpt_i = k_points[i_k]
            if solver == "lapack":
                self.set_ham_dense(kpt_i, ham_dense, convention)
                eigenvalues, eigenstates, info = lapack.zheev(ham_dense)
            else:
                ham_csr = self.set_ham_csr(kpt_i, convention)
                eigenvalues, eigenstates = eigsh(ham_csr, num_bands, **kwargs)

            # Sort eigenvalues
            idx = eigenvalues.argsort()[::-1]
            bands[i_k] = eigenvalues[idx]

            # Evaluate projection if essential
            if orbital_indices is not None:
                proj_k = self._calc_proj(orbital_indices, eigenstates.T)
                proj[i_k] = proj_k[idx]

        # Collect data
        bands = self.all_reduce(bands)
        if orbital_indices is not None:
            proj = self.all_reduce(proj)

        # Determine k-len
        k_len = kpt.gen_kdist(self.lat_vec, k_points)

        # Assemble results
        Result = namedtuple("Result", ["k_len", "bands", "proj"])
        return Result(k_len, bands, proj)

    def calc_dos(self, k_points: np.ndarray,
                 e_min: float = None,
                 e_max: float = None,
                 e_step: float = 0.05,
                 sigma: float = 0.05,
                 basis: str = "Gaussian",
                 g_s: int = 1,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate density of states for given energy range and step.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        :param e_min: lower bound of the energy range in eV
        :param e_max: upper hound of the energy range in eV
        :param e_step: energy step in eV
        :param sigma: broadening parameter in eV
        :param basis: basis function to approximate the Delta function
        :param g_s: spin degeneracy
        :param kwargs: arguments for 'calc_bands'
        :return: (energies, dos)
            energies: (num_grid,) float64 array
            energy grid corresponding to e_min, e_max and e_step
            dos: (num_grid,) float64 array
            density of states in states/eV
        :raises ValueError: if basis is neither Gaussian nor Lorentzian,
            or the solver is neither lapack nor arpack
        """
        # Get the band energies and projection
        k_len, bands, proj = self.calc_bands(k_points, **kwargs)

        # Create energy grid
        if e_min is None:
            e_min = np.min(bands)
        if e_max is None:
            e_max = np.max(bands)
        num_grid = int((e_max - e_min) / e_step)
        energies = np.linspace(e_min, e_max, num_grid + 1)

        # Evaluate DOS by collecting contributions from all energies
        dos = np.zeros(energies.shape, dtype=np.float64)
        if basis == "Gaussian":
            basis_func = gaussian
        elif basis == "Lorentzian":
            basis_func = lorentzian
        else:
            raise ValueError(f"Illegal basis function {basis}")

        # Distribute k-points over processes
        num_kpt = bands.shape[0]
        k_index = self.dist_range(num_kpt)

        # Collect contributions
        for i_k in k_index:
            for i_b, eng_i in enumerate(bands[i_k]):
                dos += basis_func(energies, eng_i, sigma) * proj.item(i_k, i_b)
        dos = self.all_reduce(dos)

        # Re-normalize dos
        # For each energy in bands, we use a normalized Gaussian or Lorentzian
        # basis function to approximate the Delta function. Totally, there are
        # bands.size basis functions. So we divide dos by this number.
        dos /= bands.size
        dos *= g_s
        return energies, dos

    def calc_states(self, k_points: np.ndarray,
                    convention: int = 1,
                    all_reduce: bool = True,
                    solver: str = "lapack",
                    num_bands: int = None,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate energies and wave functions on k-points.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of the k-points
        :param convention: convention for setting up the Hamiltonian
        :param all_reduce: whether to call MPI_Allreduce to synchronize the
            results on each process
        :param solver: solver type, should be "lapack" or "arpack"
        :param num_bands: number of bands for arpack solver
        :param kwargs: arguments for the arpack solver
        :return: (bands, states)
            bands: (num_kpt, num_bands) float64 array
            energies on the k-points
            states: (num_kpt, num_bands, num_orb) complex128 array
            wave functions on the k-points, each ROW of states[i_k] is a wave
            function
        :raises ValueError: if solver is neither lapack nor arpack
        """
        # Determine the shapes of arrays
        num_kpt = k_points.shape[0]
        num_orb = self.num_orb
        if solver == "lapack":
            num_bands = num_orb
        elif solver == "arpack":
            if num_bands is None:
                num_bands = int(num_orb * 0.6)
        else:
            raise ValueError(solver)

        # Initialize working arrays
        bands = np.zeros((num_kpt, num_bands), dtype=np.float64)
        states = np.zeros((num_kpt, num_bands, num_orb), dtype=np.complex128)
        if solver == "lapack":
            ham_dense = np.zeros((num_orb, num_orb), dtype=np.complex128)
        else:
            ham_dense = None

        # Distribute k-points over processes
        k_index = self.dist_range(num_kpt)

        # Calculate band structure
        for i_k in k_index:
            kpt_i = k_points[i_k]
            if solver == "lapack":
                self.set_ham_dense(kpt_i, ham_dense, convention)
                eigenvalues, eigenstates, info = lapack.zheev(ham_dense)
            else:
                ham_csr = self.set_ham_csr(kpt_i, convention)
                eigenvalues, eigenstates = eigsh(ham_csr, num_bands, **kwargs)
            bands[i_k] = eigenvalues
            states[i_k] = eigenstates.T

        # Collect energies and wave functions
        if all_reduce:
            bands = self.all_reduce(bands)
            states = self.all_reduce(states)
        return bands, states
