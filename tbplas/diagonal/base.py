"""Fundamentals of Solvers based on exact diagonalization."""

import math
from typing import Any, Tuple, Union, Iterable
from collections import namedtuple

import numpy as np
import scipy.linalg.lapack as lapack
from scipy.sparse.linalg import eigsh, lobpcg

from ..base import constants as consts
from ..base import lattice as lat
from ..base import kpoints as kpt
from ..parallel import MPIEnv


__all__ = ["FakePC", "FakeOverlap", "DiagSolver"]


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


class FakePC:
    """
    Base class for fake primitive cell holding the analytical Hamiltonian.

    Attributes
    ----------
    _num_orb: int
        number of orbitals
    _lat_vec: (3, 3) float64 array
        Cartesian coordinates of lattice vectors in NANO METER
        Each ROW corresponds to one lattice vector.
    """
    def __init__(self, num_orb: int,
                 lat_vec: np.ndarray = np.eye(3, dtype=np.float64),
                 unit: float = consts.ANG) -> None:
        """
        :param num_orb: number of orbitals
        :param lat_vec: (3, 3) float64 array
            Cartesian coordinates of lattice vectors in arbitrary unit
        :param unit: conversion coefficient from arbitrary unit to NM
        :return: None
        :raise ValueError: if shape of lat_vec is not (3, 3)
        """
        self._num_orb = num_orb
        lat_vec = np.array(lat_vec, dtype=np.float64)
        if lat_vec.shape != (3, 3):
            raise ValueError(f"Shape of origin is not (3, 3)")
        self._lat_vec = lat_vec * unit

    def get_reciprocal_vectors(self) -> np.ndarray:
        """
        Get the Cartesian coordinates of reciprocal lattice vectors in 1/NM.

        :return: (3, 3) float64 array
            reciprocal vectors in 1/NM.
        """
        return lat.gen_reciprocal_vectors(self._lat_vec)

    def sync_array(self) -> None:
        """Reserved for duck typing. Actually does nothing."""
        pass

    @property
    def num_orb(self) -> int:
        """Interface for the '_num_orb' attribute."""
        return self._num_orb

    @property
    def lat_vec(self) -> np.ndarray:
        """Interface for the '_lat_vec' attribute."""
        return self._lat_vec


class FakeOverlap:
    """
    Base class for fake overlap holding the analytical overlap matrix.

    Attributes
    ----------
    _num_orb: int
        number of orbitals
    """
    def __init__(self, num_orb: int) -> None:
        self._num_orb = num_orb

    def sync_array(self) -> None:
        """Reserved for duck typing. Actually does nothing."""
        pass


class DiagSolver(MPIEnv):
    """
    Base class for solvers based on exact diagonalization.

    NOTE: we declare the type of 'model' and 'overlap' to be 'Any' for duck
    typing.

    Attributes
    ----------
    __model: 'PrimitiveCell' or 'Sample' instance
        model for which properties will be calculated
    __overlap: 'Overlap' instance
        container for holding overlaps for primitive cells with non-orthogonal
        orbitals
    __h_mat: (num_orb, num_orb) complex128 array
        dense Hamiltonian matrix
    __s_mat: (num_orb, num_orb) complex128 array
        dense overlap matrix for the general eigenvalue problem
    """
    def __init__(self, model: Any,
                 overlap: Any = None,
                 enable_mpi: bool = False,
                 echo_details: bool = True) -> None:
        """
        :param model: primitive cell or sample under study
        :param overlap: container for overlaps between orbitals
        :param enable_mpi: whether to enable MPI-based parallelization
        :param echo_details: whether to output parallelization details
        """
        super().__init__(enable_mpi=enable_mpi, echo_details=echo_details)
        self.__model = model
        self.__overlap = overlap
        self.__h_mat = None
        self.__s_mat = None
        self._update_array()

    @property
    def model_is_pc(self) -> bool:
        """
        Check if the model is a primitive cell.

        :return: whether the model is a primitive cell
        """
        return hasattr(self.__model, "_lat_vec")

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

    def _update_array(self) -> None:
        """
        Update the essential arrays of the model and overlap.

        :return: None
        """
        if self.model_is_pc:
            self.__model.sync_array()
        else:
            self.__model.init_orb_pos()
            self.__model.init_orb_eng()
            self.__model.init_hop()
        if self.__overlap is not None:
            self.__overlap.sync_array()

    @staticmethod
    def _calc_proj(orbital_indices: Union[Iterable[int], np.ndarray],
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

    def _diag_ham_dense(self, k_point: np.ndarray,
                        convention: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up and diagonalize the dense Hamiltonian for given k-point.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param convention: convention for setting up the Hamiltonian
        :return: (eigenvalues, eigenvectors)
            eigenvalues: (num_orb,) float64 array
            eigenvectors: (num_orb, num_orb) complex128 array, with each column
            being an eigenvector
        :raises ValueError: if convention not in (1, 2)
        :raises RuntimeError: if diagonalization failed
        """
        # Set up the Hamiltonian
        if self.__h_mat is None:
            self.__h_mat = np.zeros((self.num_orb, self.num_orb),
                                    dtype=np.complex128)
        else:
            self.__h_mat *= 0.0
        self.__model.set_ham_dense(k_point, self.__h_mat, convention)

        # Set up the overlap matrix
        if self.__overlap is not None:
            if self.__s_mat is None:
                self.__s_mat = np.zeros((self.num_orb, self.num_orb),
                                        dtype=np.complex128)
            else:
                self.__s_mat *= 0.0
            self.__overlap.set_overlap_dense(k_point, self.__s_mat, convention)

        # Diagonalization
        if self.__overlap is None:
            eigenvalues, eigenstates, info = lapack.zheev(self.__h_mat)
        else:
            eigenvalues, eigenstates, info = \
                lapack.zhegv(self.__h_mat, self.__s_mat)
        if info != 0:
            raise RuntimeError("Diagonalization failed")
        return eigenvalues, eigenstates

    def _diag_ham_csr(self, k_point: np.ndarray,
                      convention: int = 1,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up and diagonalize the sparse Hamiltonian in csr format for given
        k-point.

        :param k_point: (3,) float64 array
            FRACTIONAL coordinate of the k-point
        :param convention: convention for setting up the Hamiltonian
        :param kwargs: arguments for the arpack solver 'eigsh' or 'logpcg'
        :return: (eigenvalues, eigenvectors)
            eigenvalues: (num_bands,) float64 array
            eigenvectors: (num_orb, num_bands) complex128 array, with each
            column being an eigenvector
        :raises ValueError: if convention not in (1, 2)
        """
        # Set up the Hamiltonian
        ham_csr = self.__model.set_ham_csr(k_point, convention)

        # Set up the overlap matrix
        if self.__overlap is not None:
            overlap_csr = self.__overlap.set_overlap_csr(k_point, convention)
        else:
            overlap_csr = None

        # Diagonalization
        if self.__overlap is None:
            eigenvalues, eigenstates = eigsh(ham_csr, **kwargs)
        else:
            rng = np.random.default_rng()
            x = rng.normal(size=(self.num_orb, kwargs["k"]))
            # Pop arguments not acceptable to lobpcg
            kwargs.pop("k")
            eigenvalues, eigenstates = \
                lobpcg(ham_csr, x, overlap_csr, **kwargs)
        return eigenvalues, eigenstates

    def calc_bands(self, k_points: np.ndarray,
                   convention: int = 1,
                   solver: str = "lapack",
                   orbital_indices: Union[Iterable[int], np.ndarray] = None,
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
        :param solver: solver type, should be "lapack" or "arpack"
        :param orbital_indices: indices of orbitals for evaluating projection,
            take all orbitals into consideration if no orbitals are specified
        :param kwargs: arguments for the arpack solver
        :return: k_len, band structure and projection packed in named tuple
        :raises ValueError: if solver is neither lapack nor arpack
        """
        self._update_array()

        # Determine the shapes of arrays
        num_kpt = k_points.shape[0]
        num_orb = self.num_orb
        if solver == "lapack":
            num_bands = num_orb
        elif solver == "arpack":
            try:
                num_bands = kwargs["k"]
            except KeyError:
                num_bands = int(num_orb * 0.6)
                kwargs["k"] = num_bands
        else:
            raise ValueError(f"Illegal solver {solver}")

        # Initialize working arrays
        bands = np.zeros((num_kpt, num_bands), dtype=np.float64)
        if orbital_indices is not None:
            proj = np.zeros((num_kpt, num_bands), dtype=np.float64)
        else:
            proj = np.ones((num_kpt, num_bands), dtype=np.float64)

        # Distribute k-points over processes
        k_index = self.dist_range(num_kpt)

        # Calculate band structure
        for i_k in k_index:
            kpt_i = k_points[i_k]
            if solver == "lapack":
                eigenvalues, eigenstates = \
                    self._diag_ham_dense(kpt_i, convention)
            else:
                eigenvalues, eigenstates = \
                    self._diag_ham_csr(kpt_i, convention, **kwargs)

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
                    solver: str = "lapack",
                    all_reduce: bool = True,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate energies and wave functions on k-points.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of the k-points
        :param convention: convention for setting up the Hamiltonian
        :param solver: solver type, should be "lapack" or "arpack"
        :param all_reduce: whether to call MPI_Allreduce to synchronize the
            results on each process
        :param kwargs: arguments for the arpack solver
        :return: (bands, states)
            bands: (num_kpt, num_bands) float64 array
            energies on the k-points
            states: (num_kpt, num_bands, num_orb) complex128 array
            wave functions on the k-points, each ROW of states[i_k] is a wave
            function
        :raises ValueError: if solver is neither lapack nor arpack
        """
        self._update_array()

        # Determine the shapes of arrays
        num_kpt = k_points.shape[0]
        num_orb = self.num_orb
        if solver == "lapack":
            num_bands = num_orb
        elif solver == "arpack":
            try:
                num_bands = kwargs["k"]
            except KeyError:
                num_bands = int(num_orb * 0.6)
                kwargs["k"] = num_bands
        else:
            raise ValueError(f"Illegal solver {solver}")

        # Initialize working arrays
        bands = np.zeros((num_kpt, num_bands), dtype=np.float64)
        states = np.zeros((num_kpt, num_bands, num_orb), dtype=np.complex128)

        # Distribute k-points over processes
        k_index = self.dist_range(num_kpt)

        # Calculate band structure
        for i_k in k_index:
            kpt_i = k_points[i_k]
            if solver == "lapack":
                eigenvalues, eigenstates = \
                    self._diag_ham_dense(kpt_i, convention)
            else:
                eigenvalues, eigenstates = \
                    self._diag_ham_csr(kpt_i, convention, **kwargs)
            bands[i_k] = eigenvalues
            states[i_k] = eigenstates.T

        # Collect energies and wave functions
        if all_reduce:
            bands = self.all_reduce(bands)
            states = self.all_reduce(states)
        return bands, states
