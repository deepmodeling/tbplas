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

from .builder import (PrimitiveCell, gen_reciprocal_vectors,
                      frac2cart, cart2frac, NM)
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
        angular frequencies for which properties will be evaluated
        TODO: what unit? rename to energies? move it to functions?
    kmesh_size: (nk_a, nk_b, nk_c)
        dimension of mesh grid in 1st Brillouin zone
    kmesh_grid: list of (ik_a, ik_n, ik_c)
        grid coordinates of k-points on mesh grid
    mu: float
        chemical potential in eV
        TODO: check the unit
    beta: float
        value for 1/kT in 1/eV
        TODO: check the unit
    g_s: int
        spin degeneracy?
        TODO: check the meaning of this parameter
    dyn_pol_factor: float
        prefactor for dynamic polarizability
    eps_factor: float
        prefactor for dielectric function
    """
    def __init__(self, cell: PrimitiveCell,
                 energy_max: float, energy_step: int,
                 kmesh_size: Tuple[int, int, int],
                 mu=0.0, temperature=300.0, back_epsilon=1.0) -> None:
        """
        :param cell: instance of 'PrimitiveCell' class
            primitive cell under investigation
        :param energy_max: float
            upper bound of energy range for evaluating energy-dependent
            properties
            TODO: check the unit
        :param energy_step: integer
            resolution of energy grid
        :param kmesh_size: (nk_a, nk_b, nk_c)
            dimension of mesh grid in 1st Brillouin zone
        :param mu: float
            chemical potential in eV
            TODO: check the unit
        :param temperature: float
            temperature in Kelvin
            TODO: check the unit
        :param back_epsilon: float
            background dielectric constant
            TODO: what should the default value be? It has been 23.6 for TBPM.
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
        self.beta = 11604.505 / temperature
        self.g_s = 2
        recip_vectors = gen_reciprocal_vectors(self.cell.lat_vec)
        dk_a = recip_vectors[0] / self.kmesh_size[0]
        dk_b = recip_vectors[1] / self.kmesh_size[1]
        dk_area = np.linalg.norm(np.cross(dk_a, dk_b))
        self.dyn_pol_factor = self.g_s * dk_area / (2 * math.pi)**2
        self.eps_factor = 1.4399644 * 2 * math.pi / back_epsilon

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

    def _get_eigen_states(self, k_points: np.ndarray):
        """
        Calculate eigenstates and eigenvalues for given k-points.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        :return: bands: (num_kpt, num_orb) float64 array
            eigenvalues of states on k-points in eV
        :return: states: (num_kpt, num_orb, num_orb) complex128 array
            eigenstates on k-points
        """
        num_kpt = k_points.shape[0]
        num_orb = self.cell.num_orb
        bands = np.zeros((num_kpt, num_orb), dtype=np.float64)
        states = np.zeros((num_kpt, num_orb, num_orb), dtype=np.complex128)
        ham_k = np.zeros((num_orb, num_orb), dtype=np.complex128)

        for i_k, k_point in enumerate(k_points):
            ham_k *= 0.0
            core.set_ham2(self.cell.orb_eng,
                          self.cell.hop_ind, self.cell.hop_eng,
                          k_point, ham_k)
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

    def calc_dyn_pol_regular(self, q_points, use_fortran=True):
        """
        Calculate dynamic polarizability for regular q-points on k-mesh.

        :param q_points: list of (iq_a, iq_b, iq_c)
            GRID coordinates of q-points
        :param use_fortran: boolean
            whether to use FORTRAN backend, set to False to enable cython
            backend for debugging
        :return: omegas: (num_omega,) float64 array
            angular frequencies
            TODO: what unit? rename to energies?
        :return: dyn_pol: (num_qpt, num_omega) complex128 array
            dynamic polarizability
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
        bands, states = self._get_eigen_states(kmesh_frac)

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
                               self.beta, self.mu, self.omegas,
                               i_q+1, q_points_cart[i_q], orb_pos,
                               dyn_pol)
            else:
                core.dyn_pol_q(bands, states, kq_map,
                               self.beta, self.mu, self.omegas,
                               i_q, q_points_cart[i_q], orb_pos,
                               dyn_pol)

        # Multiply dyn_pol by prefactor
        dyn_pol *= self.dyn_pol_factor

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
            angular frequencies
            TODO: what unit? rename to energies?
        :return: dyn_pol: (num_qpt, num_omega) complex128 array
            dynamic polarizability
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
        bands, states = self._get_eigen_states(kmesh_frac)

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
            bands_kq, states_kq = self._get_eigen_states(kq_mesh_frac)
            if use_fortran:
                bands_kq = bands_kq.T
                states_kq = states_kq.T

            # Evaluate dyn_pol for this q-point
            # FORTRAN array indices begin from 1. So we need to increase i_q
            # by 1.
            if use_fortran:
                f2py.dyn_pol_q_arb(bands, states, bands_kq, states_kq,
                                   self.beta, self.mu, self.omegas,
                                   i_q+1, q_point, orb_pos,
                                   dyn_pol)
            else:
                core.dyn_pol_q_arb(bands, states, bands_kq, states_kq,
                                   self.beta, self.mu, self.omegas,
                                   i_q, q_point, orb_pos,
                                   dyn_pol)

        # Multiply dyn_pol by prefactor
        dyn_pol *= self.dyn_pol_factor

        # Transpose dyn_pol back
        if use_fortran:
            dyn_pol = dyn_pol.T
        return self.omegas, dyn_pol
