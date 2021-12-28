"""
Module for evaluating properties from Lindhard function.

Functions
---------
    None.

Classes
-------
    Lindhard: user class
        Lindhard function calculator
"""

from typing import Tuple
import math

import numpy as np
import scipy.linalg.lapack as spla
from scipy.signal import hilbert

from .builder import (PrimitiveCell, gen_reciprocal_vectors, get_lattice_volume,
                      frac2cart, cart2frac, NM, KB, BOHR2NM, HAR2EV)
from .fortran import f2py
import tbplas.builder.core as core


class Lindhard:
    """
    Lindhard function calculator.

    Attributes
    ----------
    cell: instance of 'PrimitiveCell' class
        primitive cell under investigation
    omegas: (num_omega,) float64 array
        energies in eV for which properties will be evaluated
    kmesh_size: (nk_a, nk_b, nk_c)
        dimension of mesh grid in 1st Brillouin zone
    kmesh_grid: list of (ik_a, ik_n, ik_c)
        grid coordinates of k-points on mesh grid
    mu: float
        chemical potential in eV
    beta: float
        value for 1/kBT in 1/eV
    g_s: int
        spin degeneracy
    back_epsilon: float
        background dielectric constant (eps_r in notes)
    dimension: int
        dimension of the system
    delta: float
        broadening parameter in eV

    NOTES
    -----
    1. Unit system

    Lindhard class uses eV for energies, elementary charge of electron for
    charges and nm for lengths. So the unit for dynamic polarization is
    1 / (eV * nm**2) in 2d case and 1 / (eV * nm**3) in 3d case.

    2. Coulomb potential and dielectric function

    The Coulomb potential in Lindhard class takes the following form:
        V(r) = 1 / (eps_0 * eps_r) * e**2 / r
    where eps_0 = 1 is the dielectric constant of vacuum. There is no 4*pi
    factor to eps_0 as in Hartree or Rydberg units, i.e.,
        V(r) = 1 / (4 * pi * eps_0 * eps_r) * e**2 / r
    Function calc_epsilon_* evaluates relative dielectric constant eps_r.

    3. Fourier transform of Coulomb potential

    The 3D Fourier transform of Coulomb potential V(r) as defined in section 2
    takes the following form:
        V(q) = 4 * pi * e**2 / (eps_0 * eps_r * q**2)
    For the derivation, see pp. 19 of Many-Particle Physics by Gerald D. Mahan.

    The 2D Fourier transform takes the form:
        V(q) = 2 * pi * e**2 / (eps_0 * eps_r * q)
    See the following url for the derivation:
    https://math.stackexchange.com/questions/3627267/fourier-transform-of-the-2d-coulomb-potential
    """
    def __init__(self, cell: PrimitiveCell,
                 energy_max: float, energy_step: int,
                 kmesh_size: Tuple[int, int, int],
                 mu=0.0, temperature=300.0, back_epsilon=1.0,
                 dimension=2, delta=0.005) -> None:
        """
        :param cell: instance of 'PrimitiveCell' class
            primitive cell under investigation
        :param energy_max: float
            upper bound of energy range for evaluating energy-dependent
            properties
        :param energy_step: integer
            resolution of energy grid
        :param kmesh_size: (nk_a, nk_b, nk_c)
            dimension of mesh grid in 1st Brillouin zone
        :param mu: float
            chemical potential in eV
        :param temperature: float
            temperature in Kelvin
        :param back_epsilon: float
            background dielectric constant
        :param dimension: int
            dimension of the system
        :param delta: float
            broadening parameter in eV
        """
        self.cell = cell
        self.cell.lock()
        self.cell.sync_array()
        self.omegas = np.linspace(0, energy_max, energy_step+1)
        self.kmesh_size = kmesh_size
        self.kmesh_grid = [(ka, kb, kc)
                           for ka in range(self.kmesh_size[0])
                           for kb in range(self.kmesh_size[1])
                           for kc in range(self.kmesh_size[2])]
        self.mu = mu
        self.beta = 1 / (KB * temperature)
        self.g_s = 2
        self.back_epsilon = back_epsilon
        self.dimension = dimension
        self.delta = delta

    @staticmethod
    def _wrap_k_frac(k_points: np.ndarray):
        """
        Wrap k-points back into 1st Brillouin zone.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        :return: k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of wrapped k_points
        """
        for i_k in range(k_points.shape[0]):
            for i_dim in range(3):
                k = k_points.item(i_k, i_dim)
                while k < 0.0:
                    k += 1.0
                while k >= 1.0:
                    k -= 1.0
                k_points[i_k, i_dim] = k
        return k_points

    def _wrap_k_grid(self, k_points):
        """
        Wrap k-points back into 1st Brillouin zone.

        :param k_points: list of (ik_a, ik_b, ik_c)
            GRID coordinates of k-points
        :return: k_points: list of (ik_a, ik_b, ik_c)
            GRID coordinates of wrapped k_points
        """
        k_points = [(ka % self.kmesh_size[0],
                     kb % self.kmesh_size[1],
                     kc % self.kmesh_size[2])
                    for (ka, kb, kc) in k_points]
        return k_points

    def frac2cart(self, frac_coord: np.ndarray, unit=NM):
        """
        Convert FRACTIONAL coordinates of k-points to CARTESIAN coordinates
        in 1/unit.

        :param frac_coord: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        :param unit: float
            scaling factor from unit to NANOMETER, e.g. unit=0.1 for ANGSTROM
        :return: cart_coord: (num_kpt, 3) float64 array
            CARTESIAN coordinates of k-points in 1/unit
        """
        recip_lat_vec = gen_reciprocal_vectors(self.cell.lat_vec)
        cart_coord_nm = frac2cart(recip_lat_vec, frac_coord)
        cart_coord = cart_coord_nm * unit
        return cart_coord

    def cart2frac(self, cart_coord: np.ndarray, unit=NM):
        """
        Convert CARTESIAN coordinates of k-points in 1/unit to FRACTIONAL
        coordinates.

        :param cart_coord: (num_kpt, 3) float64 array
            CARTESIAN coordinates in 1/unit
        :param unit: float
            scaling factor from unit to NANOMETER, e.g. unit=0.1 for ANGSTROM
        :return: frac_coord: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        """
        recip_lat_vec = gen_reciprocal_vectors(self.cell.lat_vec)
        cart_coord_nm = cart_coord / unit
        frac_coord = cart2frac(recip_lat_vec, cart_coord_nm)
        return frac_coord

    def grid2frac(self, grid_coord, wrap=True):
        """
        Convert GRID coordinates of k-points to FRACTIONAL coordinates.

        :param grid_coord: list of (ik_a, ik_b, ik_c)
            GRID coordinates of k-points
        :param wrap: boolean
            whether to wrap k-points if they fall out of 1st
            Brillouin zone
        :return: frac_coord: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        """
        if wrap:
            grid_coord = self._wrap_k_grid(grid_coord)
        frac_coord = [(ka / self.kmesh_size[0],
                       kb / self.kmesh_size[1],
                       kc / self.kmesh_size[2])
                      for (ka, kb, kc) in grid_coord]
        frac_coord = np.array(frac_coord, dtype=np.float64)
        return frac_coord

    def grid2cart(self, grid_coord, wrap=True, unit=NM):
        """
        Convert GRID coordinates of k-points to CARTESIAN coordinates in
        1/unit.

        :param grid_coord: list of (ik_a, ik_b, ik_c)
            GRID coordinates of k-points
        :param wrap: boolean
            whether to wrap k-points if they fall out of 1st
            Brillouin zone
        :param unit: float
            scaling factor from unit to NANOMETER, e.g. unit=0.1 for ANGSTROM
        :return: cart_coord: (num_kpt, 3) float64 array
            CARTESIAN coordinates of k-points in 1/unit
        """
        frac_coord = self.grid2frac(grid_coord, wrap=wrap)
        cart_coord = self.frac2cart(frac_coord, unit=unit)
        return cart_coord

    def _get_eigen_states(self, k_points: np.ndarray, convention=1):
        """
        Calculate eigenstates and eigenvalues for given k-points.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        :param convention: integer
            convention to construct Hamiltonian
        :return: bands: (num_kpt, num_orb) float64 array
            eigenvalues of states on k-points in eV
        :return: states: (num_kpt, num_orb, num_orb) complex128 array
            eigenstates on k-points
        :raises ValueError: if convention is neither 1 nor 2
        """
        num_kpt = k_points.shape[0]
        num_orb = self.cell.num_orb
        bands = np.zeros((num_kpt, num_orb), dtype=np.float64)
        states = np.zeros((num_kpt, num_orb, num_orb), dtype=np.complex128)
        ham_k = np.zeros((num_orb, num_orb), dtype=np.complex128)

        for i_k, k_point in enumerate(k_points):
            ham_k *= 0.0
            if convention == 1:
                core.set_ham(self.cell.orb_pos, self.cell.orb_eng,
                             self.cell.hop_ind, self.cell.hop_eng,
                             k_point, ham_k)
            elif convention == 2:
                core.set_ham2(self.cell.orb_eng,
                              self.cell.hop_ind, self.cell.hop_eng,
                              k_point, ham_k)
            else:
                raise ValueError(f"Illegal convention {convention}")
            eigenvalues, eigenstates, info = spla.zheev(ham_k)
            bands[i_k] = eigenvalues
            states[i_k] = eigenstates.T
        return bands, states

    def _gen_kq_map(self, q_point):
        """
        Map k-points on k+q grid to k-grid.

        :param q_point: (iq_a, iq_b, iq_c)
            GRID coordinate of q-point
        :return: kq_map: int64 array
            map of k-points on k+q grid to k-grid, e.g.  kq_map[0] = 3 means
            the 0th k-point on k+q grid is the 3rd k-point on k grid.
        """
        kq_map = []
        for k_point in self.kmesh_grid:
            kq_a = (k_point[0] + q_point[0]) % self.kmesh_size[0]
            kq_b = (k_point[1] + q_point[1]) % self.kmesh_size[1]
            kq_c = (k_point[2] + q_point[2]) % self.kmesh_size[2]
            kq_map.append(self.kmesh_grid.index((kq_a, kq_b, kq_c)))
        kq_map = np.array(kq_map, dtype=np.int64)
        return kq_map

    def _get_dyn_pol_factor(self):
        """
        Get prefactor for dynamic polarizability.

        :return: dyn_pol_factor: float
            prefactor for dynamic polarizability
        :raises NotImplementedError: if system dimension is not 2 or 3
        """
        g_vec = gen_reciprocal_vectors(self.cell.lat_vec)
        for i_dim in range(3):
            g_vec[i_dim] /= self.kmesh_size[i_dim]
        if self.dimension == 2:
            dk_area = np.linalg.norm(np.cross(g_vec[0], g_vec[1]))
            dyn_pol_factor = self.g_s * dk_area / (2 * math.pi)**2
        elif self.dimension == 3:
            dk_volume = get_lattice_volume(g_vec)
            dyn_pol_factor = self.g_s * dk_volume / (2 * math.pi)**3
        else:
            raise NotImplementedError(f"Dimension {self.dimension} not "
                                      f"implemented")
        return dyn_pol_factor

    def calc_dyn_pol_regular(self, q_points, use_fortran=True):
        """
        Calculate dynamic polarizability for regular q-points on k-mesh.

        :param q_points: list of (iq_a, iq_b, iq_c)
            GRID coordinates of q-points
        :param use_fortran: boolean
            whether to use FORTRAN backend, set to False to enable cython
            backend for debugging
        :return: omegas: (num_omega,) float64 array
            energies in eV
        :return: dyn_pol: (num_qpt, num_omega) complex128 array
            dynamic polarizability in 1/(eV*nm**2) or 1/(eV*nm**3)
        :raises NotImplementedError: if system dimension is not 2 or 3
        """
        # Prepare q-points and orbital positions
        q_points_cart = self.grid2cart(q_points, unit=NM)
        orb_pos = self.cell.orb_pos_nm

        # Allocate dyn_pol
        num_qpt = len(q_points)
        num_omega = self.omegas.shape[0]
        dyn_pol = np.zeros((num_qpt, num_omega), dtype=np.complex128)

        # Get eigenvalues and eigenstates
        kmesh_frac = self.grid2frac(self.kmesh_grid)
        bands, states = self._get_eigen_states(kmesh_frac, convention=2)

        # Transpose arrays for compatibility with FORTRAN backend
        if use_fortran:
            orb_pos = orb_pos.T
            bands = bands.T
            states = states.T
            dyn_pol = dyn_pol.T

        # Evaluate dyn_pol for all q-points
        for i_q, q_point in enumerate(q_points):
            # Remap k+q to k
            kq_map = self._gen_kq_map(q_point)

            # Evaluate dyn_pol for this q-point
            # FORTRAN array indices begin from 1. So we need to increase
            # kq_map and i_q by 1.
            if use_fortran:
                kq_map += 1
                f2py.dyn_pol_q(bands, states, kq_map,
                               self.beta, self.mu, self.omegas, self.delta,
                               i_q+1, q_points_cart[i_q], orb_pos,
                               dyn_pol)
            else:
                core.dyn_pol_q(bands, states, kq_map,
                               self.beta, self.mu, self.omegas, self.delta,
                               i_q, q_points_cart[i_q], orb_pos,
                               dyn_pol)

        # Multiply dyn_pol by prefactor
        dyn_pol *= self._get_dyn_pol_factor()

        # Transpose dyn_pol back
        if use_fortran:
            dyn_pol = dyn_pol.T
        return self.omegas, dyn_pol

    def calc_dyn_pol_arbitrary(self, q_points, use_fortran=True):
        """
        Calculate dynamic polarizability for arbitrary q-points.

        :param q_points: list of (q_x, q_y, q_z)
            CARTESIAN coordinates of q-points in 1/NM
        :param use_fortran: boolean
            whether to use FORTRAN backend, set to False to enable cython
            backend for debugging
        :return: omegas: (num_omega,) float64 array
            energies in eV
        :return: dyn_pol: (num_qpt, num_omega) complex128 array
            dynamic polarizability in 1/(eV*nm**2) or 1/(eV*nm**3)
        :raises NotImplementedError: if system dimension is not 2 or 3
        """
        # Prepare q-points and orbital positions
        q_points = np.array(q_points, dtype=np.float64)
        q_points_frac = self.cart2frac(q_points, unit=NM)
        orb_pos = self.cell.orb_pos_nm

        # Allocate dyn_pol
        num_qpt = len(q_points)
        num_omega = self.omegas.shape[0]
        dyn_pol = np.zeros((num_qpt, num_omega), dtype=np.complex128)

        # Get eigenvalues and eigenstates
        kmesh_frac = self.grid2frac(self.kmesh_grid)
        bands, states = self._get_eigen_states(kmesh_frac, convention=2)

        # Transpose arrays for compatibility with FORTRAN backend
        if use_fortran:
            orb_pos = orb_pos.T
            bands = bands.T
            states = states.T
            dyn_pol = dyn_pol.T

        # Evaluate dyn_pol for all q-points
        for i_q, q_point in enumerate(q_points):
            # Get eigenvalues and eigenstates on k+q grid
            kq_mesh_frac = kmesh_frac + q_points_frac[i_q]
            bands_kq, states_kq = self._get_eigen_states(kq_mesh_frac,
                                                         convention=2)
            if use_fortran:
                bands_kq = bands_kq.T
                states_kq = states_kq.T

            # Evaluate dyn_pol for this q-point
            # FORTRAN array indices begin from 1. So we need to increase i_q
            # by 1.
            if use_fortran:
                f2py.dyn_pol_q_arb(bands, states, bands_kq, states_kq,
                                   self.beta, self.mu, self.omegas, self.delta,
                                   i_q+1, q_point, orb_pos,
                                   dyn_pol)
            else:
                core.dyn_pol_q_arb(bands, states, bands_kq, states_kq,
                                   self.beta, self.mu, self.omegas, self.delta,
                                   i_q, q_point, orb_pos,
                                   dyn_pol)

        # Multiply dyn_pol by prefactor
        dyn_pol *= self._get_dyn_pol_factor()

        # Transpose dyn_pol back
        if use_fortran:
            dyn_pol = dyn_pol.T
        return self.omegas, dyn_pol

    def _get_coulomb_factor(self):
        """
        Get prefactor for Coulomb interaction in momentum space.

        factor = 10**9 * elementary_charge / (4 * pi * epsilon_0) in SI units

        :return: coulomb_factor: float
            prefactor for Coulomb interaction
        :raises NotImplementedError: if system dimension is not 2 or 3
        """
        factor = 1.439964548
        if self.dimension == 2:
            coulomb_factor = factor * 2 * math.pi / self.back_epsilon
        elif self.dimension == 3:
            coulomb_factor = factor * 4 * math.pi / self.back_epsilon
        else:
            raise NotImplementedError(f"Dimension {self.dimension} not "
                                      f"implemented")
        return coulomb_factor

    def _calc_epsilon(self, q_points_nm: np.ndarray, dyn_pol: np.ndarray):
        """
        Core function for evaluating dielectric function.

        :param q_points_nm: (num_qpt, 3) float64 array
            CARTESIAN coordinates of q-points in NM
        :param dyn_pol: (num_qpt, num_omega) complex128 array
            dynamic polarization from calc_dyn_pol_*
        :return: epsilon: (num_qpt, num_omega) complex128 array
            relative dielectric function
        :raises NotImplementedError: if system dimension is not 2 or 3
        """
        num_qpt = len(q_points_nm)
        num_omega = self.omegas.shape[0]
        epsilon = np.zeros((num_qpt, num_omega), dtype=np.complex128)
        prefactor = self._get_coulomb_factor()
        for i_q, q_point in enumerate(q_points_nm):
            q_norm = np.linalg.norm(q_point)
            if self.dimension == 2:
                v_q = prefactor / q_norm
            elif self.dimension == 3:
                v_q = prefactor / q_norm**2
            else:
                raise NotImplementedError(f"Dimension {self.dimension} not "
                                          f"implemented")
            epsilon[i_q] = 1.0 - v_q * dyn_pol[i_q]
        return epsilon

    def calc_epsilon_regular(self, q_points, use_fortran=True):
        """
        Calculate dielectric function (eps_r) for regular q-points on k-mesh.

        :param q_points: list of (iq_a, iq_b, iq_c)
            GRID coordinates of q-points
        :param use_fortran: boolean
            whether to use FORTRAN backend, set to False to enable cython
            backend for debugging
        :return: omegas: (num_omega,) float64 array
            energies in eV
        :return: epsilon: (num_qpt, num_omega) complex128 array
            relative dielectric function
        :raises NotImplementedError: if system dimension is not 2 or 3
        """
        omegas, dyn_pol = self.calc_dyn_pol_regular(q_points, use_fortran)
        q_points_nm = self.grid2cart(q_points, unit=NM)
        epsilon = self._calc_epsilon(q_points_nm, dyn_pol)
        return omegas, epsilon

    def calc_epsilon_arbitrary(self, q_points, use_fortran=True):
        """
        Calculate dielectric function (eps_r) for arbitrary q-points.

        :param q_points: list of (q_x, q_y, q_z)
            CARTESIAN coordinates of q-points in 1/NM
        :param use_fortran: boolean
            whether to use FORTRAN backend, set to False to enable cython
            backend for debugging
        :return: omegas: (num_omega,) float64 array
            energies in eV
        :return: epsilon: (num_qpt, num_omega) complex128 array
            relative dielectric function
        :raises NotImplementedError: if system dimension is not 2 or 3
        """
        omegas, dyn_pol = self.calc_dyn_pol_arbitrary(q_points, use_fortran)
        epsilon = self._calc_epsilon(q_points, dyn_pol)
        return omegas, epsilon
