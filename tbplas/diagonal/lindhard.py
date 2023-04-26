"""Module for evaluating response properties from Lindhard function."""

from typing import Tuple, Any
import math

import numpy as np

from ..base import (gen_reciprocal_vectors, get_lattice_volume,
                    frac2cart, cart2frac, NM, KB, EPSILON0)
from ..fortran import f2py
from . import core
from .base import DiagSolver


__all__ = ["Lindhard"]


class Lindhard(DiagSolver):
    """
    Lindhard function calculator.

    Attributes
    ----------
    cell: 'PrimitiveCell' instance
        primitive cell for which properties will be evaluated, kept as an alias
        to the 'model' attribute of 'DiagSolver' class for compatibility
    omegas: (num_omega,) float64 array
        energy grid in eV
    kmesh_size: (3,) int64 array
        dimension of mesh grid in the 1st Brillouin zone
    kmesh_grid: (num_kpt, 3) int64 array
        grid coordinates of k-points on the mesh grid
    mu: float
        chemical potential in eV
    beta: float
        1/kBT in 1/eV
    g_s: int
        spin degeneracy
    back_epsilon: float
        relative background dielectric constant
    dimension: int
        dimension of the system
    delta: float
        broadening parameter in eV that appears in the denominator of Lindhard
        function

    NOTES
    -----
    1. Units

    'Lindhard' class uses eV for energies, elementary charge of electron for
    charges and nm for lengths. So the unit for dynamic polarization is
    1 / (eV * nm**2) in 2d case and 1 / (eV * nm**3) in 3d case. Relative
    dielectric function is dimensionless. The unit for AC conductivity is
    e**2/h_bar in 2d case and e**2/(h_bar*nm) in 3d case.

    2. Coulomb potential and dielectric function

    The Coulomb potential in Lindhard class takes the following form:
        V(r) = 1 / epsilon * e**2 / r
             = 1 / (epsilon_0 * epsilon_r) * e**2 / r
    where epsilon is the ABSOLUTE dielectric electric constant of background.

    Now we evaluate the absolute dielectric constant of vacuum in the units of
    eV/nm/q. Suppose that we have two electrons separated at the distance of
    1nm, then their potential energy in SI units is:
        V = 1 / (4 * pi * epsilon_0) * q**2 / 1e-9 [Joule]
          = 1 / (4 * pi * epsilon_0) * q**2 / 1e-9 * 1 / q [eV]
          = q * 1e9 / (4 * pi * epsilon_0) [eV]
    The counterpart in eV/nm/q units is:
        V = 1 / epsilon_0_arb [eV]
    So we have:
        1 / epsilon_0_arb = q * 1e9 / (4 * pi * epsilon_0)
    where the right part are the values in SI. Finally, we have:
        epsilon_0_arb = 0.6944615417149689
    in eV/nm/q units. That's EPSILON0 in constants module.

    3. Fourier transform of Coulomb potential

    The 3D Fourier transform of Coulomb potential V(r) as defined in section 2
    takes the following form:
        V(q) = 4 * pi * e**2 / (epsilon_0 * epsilon_r * q**2)
    For the derivation, see pp. 19 of Many-Particle Physics by Gerald D. Mahan.

    The 2D Fourier transform takes the form:
        V(q) = 2 * pi * e**2 / (epsilon_0 * epsilon_r * q)
    See the following url for the derivation:
    https://math.stackexchange.com/questions/3627267/fourier-transform-of-the-2d-coulomb-potential

    4. System dimension

    'Lindhard' class deals with system dimension in two approaches. The first
    approach is to treat all systems as 3-dimensional. Supercell technique is
    required in this approach, with vacuum layers added on non-periodic
    directions. Also, the component(s) of kmesh_size should be set to 1
    accordingly.

    The second approach utilizes dimension-specific formula whenever possible.
    For now, only 2-dimensional case has been implemented. This approach
    requires that the system should be periodic in xOy plane, i.e. the
    non-periodic direction should be along 'c' axis. Otherwise, the results
    will be wrong.

    The first approach suffers from the problem that dynamic polarization and
    AC conductivity scale inversely proportional to the product of supercell
    lengths, i.e. |c| in 2d case and |a|*|b| in 1d case. This is due to the
    elementary volume in reciprocal space (dnk) in the summation in Lindhard
    function. On the contrary, the second approach has no such issue. If the
    supercell lengths of non-periodic directions are set to 1 nm, then the first
    approach yields the same results as the second approach.

    For the dielectric function, the situation is more complicated. From the
    equation
        epsilon(q) = 1 - V(q) * dyn_pol(q)
    we can see it is also affected by the Coulomb potential V(q). Since
        V(q) = 4 * pi * e**2 / (epsilon_0 * epsilon_r * q**2) (3d)
    and
        V(q) = 2 * pi * e**2 / (epsilon_0 * epsilon_r * q) (2d)
    the influence of system dimension is q-dependent. Setting supercell lengths
    to 1 nm will NOT produce the same result as the second approach.
    """
    def __init__(self, cell: Any,
                 energy_max: float,
                 energy_step: int,
                 kmesh_size: Tuple[int, int, int],
                 mu: float = 0.0,
                 temperature: float = 300.0,
                 g_s: int = 1,
                 back_epsilon: float = 1.0,
                 dimension: int = 3,
                 delta: float = 0.005,
                 **kwargs) -> None:
        """
        :param cell: primitive cell for which properties will be evaluated
        :param energy_max: upper bound of energy grid for evaluating response
            properties
        :param energy_step: resolution of energy grid
        :param kmesh_size: dimension of mesh grid in 1st Brillouin zone
        :param mu: chemical potential of the cell in eV
        :param temperature: temperature in Kelvin
        :param g_s: spin degeneracy
        :param back_epsilon: relative background dielectric constant
        :param dimension: dimension of the system
        :param delta: broadening parameter in eV
        :param kwargs: parallelization arguments for DiagSolver.__init__
        :raises ValueError: if kmesh_size and dimension are not properly set
        """
        super().__init__(cell, **kwargs)
        self.cell = self.model
        self.omegas = np.linspace(0, energy_max, energy_step+1)
        if len(kmesh_size) != 3:
            raise ValueError("Length of kmesh_size should be 3")
        self.kmesh_size = np.array(kmesh_size, dtype=np.int64)
        self.kmesh_grid = core.build_kmesh_grid(self.kmesh_size)
        self.mu = mu
        self.beta = 1 / (KB * temperature)
        self.g_s = g_s
        self.back_epsilon = back_epsilon
        if dimension not in (2, 3):
            raise ValueError(f"Unsupported dimension: {dimension}")
        if dimension == 2 and kmesh_size[2] != 1:
            raise ValueError("2d specific algorithms require kmesh_size[2] == 1")
        self.dimension = dimension
        self.delta = delta

    @staticmethod
    def wrap_frac(k_points: np.ndarray) -> np.ndarray:
        """
        Wrap fractional coordinates of k-points back into 1st Brillouin zone.

        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        :return: (num_kpt, 3) float64 array
            wrapped FRACTIONAL coordinates of k-points
        """
        if not isinstance(k_points, np.ndarray):
            k_points = np.array(k_points, dtype=np.float64)
        k_wrap = k_points.copy()
        for i_k in range(k_wrap.shape[0]):
            for i_dim in range(3):
                k = k_wrap.item(i_k, i_dim)
                while k < 0.0:
                    k += 1.0
                while k >= 1.0:
                    k -= 1.0
                k_wrap[i_k, i_dim] = k
        return k_wrap

    def wrap_grid(self, k_points: np.ndarray) -> np.ndarray:
        """
        Wrap grid coordinates of k-points back into 1st Brillouin zone.

        :param k_points: (num_kpt, 3) int64 array
            GRID coordinates of k-points
        :return: (num_kpt, 3) int64 array
            wrapped GRID coordinates of k-points
        """
        if not isinstance(k_points, np.ndarray):
            k_points = np.array(k_points, dtype=np.int64)
        k_wrap = k_points.copy()
        for i_k in range(k_wrap.shape[0]):
            for i_dim in range(3):
                k = k_wrap.item(i_k, i_dim)
                k_wrap[i_k, i_dim] = k % self.kmesh_size.item(i_dim)
        return k_wrap

    def frac2cart(self, frac_coord: np.ndarray, unit: float = NM) -> np.ndarray:
        """
        Convert FRACTIONAL coordinates of k-points to CARTESIAN coordinates
        in 1/unit.

        :param frac_coord: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        :param unit: scaling factor from unit to NANOMETER,
            e.g. unit=0.1 for ANGSTROM
        :return: (num_kpt, 3) float64 array
            CARTESIAN coordinates of k-points in 1/unit
        """
        if not isinstance(frac_coord, np.ndarray):
            frac_coord = np.array(frac_coord, dtype=np.float64)
        recip_lat_vec = gen_reciprocal_vectors(self.cell.lat_vec)
        cart_coord_nm = frac2cart(recip_lat_vec, frac_coord)
        cart_coord = cart_coord_nm * unit
        return cart_coord

    def cart2frac(self, cart_coord: np.ndarray, unit: float = NM) -> np.ndarray:
        """
        Convert CARTESIAN coordinates of k-points in 1/unit to FRACTIONAL
        coordinates.

        :param cart_coord: (num_kpt, 3) float64 array
            CARTESIAN coordinates in 1/unit
        :param unit: scaling factor from unit to NANOMETER,
            e.g. unit=0.1 for ANGSTROM
        :return: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        """
        if not isinstance(cart_coord, np.ndarray):
            cart_coord = np.array(cart_coord, dtype=np.float64)
        recip_lat_vec = gen_reciprocal_vectors(self.cell.lat_vec)
        cart_coord_nm = cart_coord / unit
        frac_coord = cart2frac(recip_lat_vec, cart_coord_nm)
        return frac_coord

    def grid2frac(self, grid_coord: np.ndarray) -> np.ndarray:
        """
        Convert GRID coordinates of k-points to FRACTIONAL coordinates.

        :param grid_coord: (num_kpt, 3) int64 array
            GRID coordinates of k-points
        :return: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of k-points
        """
        if not isinstance(grid_coord, np.ndarray):
            grid_coord = np.array(grid_coord, dtype=np.int64)
        frac_coord = np.array(grid_coord, dtype=np.float64)
        for i_dim in range(3):
            frac_coord[:, i_dim] /= self.kmesh_size.item(i_dim)
        return frac_coord

    def grid2cart(self, grid_coord: np.ndarray, unit: float = NM) -> np.ndarray:
        """
        Convert GRID coordinates of k-points to CARTESIAN coordinates in
        1/unit.

        :param grid_coord: (num_kpt, 3) int64 array
            GRID coordinates of k-points
        :param unit: scaling factor from unit to NANOMETER,
            e.g. unit=0.1 for ANGSTROM
        :return: (num_kpt, 3) float64 array
            CARTESIAN coordinates of k-points in 1/unit
        """
        if not isinstance(grid_coord, np.ndarray):
            grid_coord = np.array(grid_coord, dtype=np.int64)
        frac_coord = self.grid2frac(grid_coord)
        cart_coord = self.frac2cart(frac_coord, unit=unit)
        return cart_coord

    def _get_dnk(self) -> float:
        """
        Get elementary area/volume in reciprocal space depending on system
        dimension.

        :return: dnk: elementary area/volume in 1/nm**2 or 1/nm**3
        """
        g_vec = gen_reciprocal_vectors(self.cell.lat_vec)
        for i_dim in range(3):
            g_vec[i_dim] /= self.kmesh_size.item(i_dim)
        if self.dimension == 2:
            dnk = np.linalg.norm(np.cross(g_vec[0], g_vec[1]))
        else:
            dnk = get_lattice_volume(g_vec)
        return dnk

    def _get_dyn_pol_factor(self) -> float:
        """
        Get prefactor for dynamic polarizability.

        :return: prefactor for dynamic polarizability
        """
        dnk = self._get_dnk()
        dyn_pol_factor = self.g_s * dnk / (2 * math.pi)**self.dimension
        return dyn_pol_factor

    def calc_dyn_pol_regular(self, q_points: np.ndarray,
                             use_fortran: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate dynamic polarizability for regular q-points on k-mesh.

        :param q_points: (num_qpt, 3) int64 array
            GRID coordinates of q-points
        :param use_fortran: choose between FORTRAN and Cython backends
        :return: (omegas, dyn_pol)
            omegas: (num_omega,) float64 array
            energies in eV
            dyn_pol: (num_qpt, num_omega) complex128 array
            dynamic polarizability in 1/(eV*nm**2) or 1/(eV*nm**3)
        """
        # Prepare q-points and orbital positions
        if not isinstance(q_points, np.ndarray):
            q_points = np.array(q_points, dtype=np.int64)
        q_points_cart = self.grid2cart(q_points, unit=NM)
        orb_pos = self.cell.orb_pos_nm

        # Allocate working arrays
        num_qpt = q_points.shape[0]
        num_kpt = self.kmesh_grid.shape[0]
        num_omega = self.omegas.shape[0]
        num_bands = self.cell.num_orb
        delta_eng = np.zeros((num_kpt, num_bands, num_bands), dtype=np.float64)
        prod_df = np.zeros((num_kpt, num_bands, num_bands), dtype=np.complex128)
        dyn_pol = np.zeros((num_qpt, num_omega), dtype=np.complex128)

        # Get eigenvalues and eigenstates
        kmesh_frac = self.grid2frac(self.kmesh_grid)
        bands, states = self.calc_states(kmesh_frac, convention=2)

        # Transpose arrays for compatibility with FORTRAN backend
        if use_fortran:
            orb_pos = orb_pos.T
            bands = bands.T
            states = states.T
            delta_eng = delta_eng.T
            prod_df = prod_df.T
            dyn_pol = dyn_pol.T

        # Distribute k-points
        k_min, k_max = self.dist_bound(num_kpt)

        # Evaluate dyn_pol for all q-points
        for i_q, q_point in enumerate(q_points):
            # Remap k+q to k
            kq_map = core.build_kq_map(self.kmesh_size, self.kmesh_grid,
                                       q_point)

            # Setup working arrays
            delta_eng *= 0.0
            prod_df *= 0.0
            if use_fortran:
                # NOTE: FORTRAN array index begins from 1!
                kq_map += 1
                f2py.prod_dp(bands, states, kq_map, self.beta, self.mu,
                             q_points_cart[i_q], orb_pos, k_min+1, k_max+1,
                             delta_eng, prod_df)
            else:
                core.prod_dp(bands, states, kq_map, self.beta, self.mu,
                             q_points_cart[i_q], orb_pos, k_min, k_max,
                             delta_eng, prod_df)
            delta_eng = self.all_reduce(delta_eng)
            prod_df = self.all_reduce(prod_df)

            # Evaluate dyn_pol
            if use_fortran:
                # NOTE: FORTRAN array index begins from 1!
                f2py.dyn_pol_f(delta_eng, prod_df, self.omegas, self.delta,
                               k_min+1, k_max+1, i_q+1, dyn_pol)
            else:
                core.dyn_pol(delta_eng, prod_df, self.omegas, self.delta,
                             k_min, k_max, i_q, dyn_pol)
            dyn_pol = self.all_reduce(dyn_pol)

        # Multiply dyn_pol by prefactor
        dyn_pol *= self._get_dyn_pol_factor()
        dyn_pol /= self.cell.extended

        # Transpose dyn_pol back
        if use_fortran:
            dyn_pol = dyn_pol.T
        return self.omegas, dyn_pol

    def calc_dyn_pol_arbitrary(self, q_points: np.ndarray,
                               use_fortran: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate dynamic polarizability for arbitrary q-points.

        :param q_points: (num_qpt, 3) float64 array
            CARTESIAN coordinates of q-points in 1/NM
        :param use_fortran: choose between FORTRAN and Cython backends
        :return: (omegas, dyn_pol)
            omegas: (num_omega,) float64 array
            energies in eV
            dyn_pol: (num_qpt, num_omega) complex128 array
            dynamic polarizability in 1/(eV*nm**2) or 1/(eV*nm**3)
        """
        # Prepare q-points and orbital positions
        if not isinstance(q_points, np.ndarray):
            q_points = np.array(q_points, dtype=np.float64)
        q_points_frac = self.cart2frac(q_points, unit=NM)
        orb_pos = self.cell.orb_pos_nm

        # Allocate working arrays
        num_qpt = q_points.shape[0]
        num_kpt = self.kmesh_grid.shape[0]
        num_omega = self.omegas.shape[0]
        num_bands = self.cell.num_orb
        delta_eng = np.zeros((num_kpt, num_bands, num_bands), dtype=np.float64)
        prod_df = np.zeros((num_kpt, num_bands, num_bands), dtype=np.complex128)
        dyn_pol = np.zeros((num_qpt, num_omega), dtype=np.complex128)

        # Get eigenvalues and eigenstates
        # It is important not to synchronize the wave functions on each process.
        # Otherwise, OverflowError may be raised if the kmesh_grid is too dense.
        kmesh_frac = self.grid2frac(self.kmesh_grid)
        bands, states = self.calc_states(kmesh_frac, convention=2,
                                         all_reduce=False)

        # Transpose arrays for compatibility with FORTRAN backend
        if use_fortran:
            orb_pos = orb_pos.T
            bands = bands.T
            states = states.T
            delta_eng = delta_eng.T
            prod_df = prod_df.T
            dyn_pol = dyn_pol.T

        # Distribute k-points
        k_min, k_max = self.dist_bound(num_kpt)

        # Evaluate dyn_pol for all q-points
        for i_q, q_point in enumerate(q_points):
            # Get eigenvalues and eigenstates on k+q grid
            kq_mesh_frac = kmesh_frac + q_points_frac[i_q]
            bands_kq, states_kq = self.calc_states(kq_mesh_frac, convention=2,
                                                   all_reduce=False)

            # Setup working arrays
            delta_eng *= 0.0
            prod_df *= 0.0
            if use_fortran:
                # NOTE: FORTRAN array index begins from 1!
                bands_kq = bands_kq.T
                states_kq = states_kq.T
                f2py.prod_dp_arb(bands, states, bands_kq, states_kq,
                                 self.beta, self.mu, q_point, orb_pos,
                                 k_min+1, k_max+1, delta_eng, prod_df)
            else:
                core.prod_dp_arb(bands, states, bands_kq, states_kq,
                                 self.beta, self.mu, q_point, orb_pos,
                                 k_min, k_max, delta_eng, prod_df)

            # Evaluate dyn_pol
            if use_fortran:
                # NOTE: FORTRAN array index begins from 1!
                f2py.dyn_pol_f(delta_eng, prod_df, self.omegas, self.delta,
                               k_min+1, k_max+1, i_q+1, dyn_pol)
            else:
                core.dyn_pol(delta_eng, prod_df, self.omegas, self.delta,
                             k_min, k_max, i_q, dyn_pol)
            dyn_pol = self.all_reduce(dyn_pol)

        # Multiply dyn_pol by prefactor
        dyn_pol *= self._get_dyn_pol_factor()
        dyn_pol /= self.cell.extended

        # Transpose dyn_pol back
        if use_fortran:
            dyn_pol = dyn_pol.T
        return self.omegas, dyn_pol

    def _get_vq(self, q_point: np.ndarray) -> float:
        """
        Get Coulomb interaction in momentum space for given q-point.

        :param q_point: (3,) float64 array
            CARTESIAN coordinate of q-point in 1/NM
        :return: Coulomb interaction in momentum space in eV
        """
        factor = 1 / (EPSILON0 * self.back_epsilon)
        q_norm = np.linalg.norm(q_point)
        if self.dimension == 2:
            vq = factor * 2 * math.pi / q_norm
        else:
            vq = factor * 4 * math.pi / q_norm**2
        return vq

    def calc_epsilon(self, q_points: np.ndarray,
                     dyn_pol: np.ndarray) -> np.ndarray:
        """
        Calculate dielectric function for given q-points from dynamic
        polarization.

        :param q_points: (num_qpt, 3) float64 array
            CARTESIAN coordinates of q-points in 1/NM
        :param dyn_pol: (num_qpt, num_omega) complex128 array
            dynamic polarization from calc_dyn_pol_*
        :return: (num_qpt, num_omega) complex128 array
            relative dielectric function
        """
        if not isinstance(q_points, np.ndarray):
            q_points = np.ndarray(q_points, dtype=np.float64)
        num_qpt = q_points.shape[0]
        num_omega = self.omegas.shape[0]
        epsilon = np.zeros((num_qpt, num_omega), dtype=np.complex128)
        for i_q, q_point in enumerate(q_points):
            vq = self._get_vq(q_point)
            epsilon[i_q] = 1.0 - vq * dyn_pol[i_q]
        return epsilon

    def calc_ac_cond(self, component: str = "xx",
                     use_fortran: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate AC conductivity using Kubo-Greenwood formula.

        Reference: section 12.2 of Wannier90 user guide.
        NOTE: there is not such g_s factor in the reference.

        :param component: which component of conductivity to evaluate
            should be in "xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy" and "zz"
        :param use_fortran: choose between FORTRAN and Cython backends
        :return: (omegas, ac_cond)
            omegas: (num_omega,) float64 array
            energies in eV
            ac_cond: (num_omega,) complex128 array
            AC conductivity in e**2/(h_bar*nm) in 3d case and e**2/h_bar
            in 2d case
        :raises ValueError: if component is illegal
        """
        # Aliases for variables
        kmesh = self.grid2cart(self.kmesh_grid, unit=NM)
        hop_eng = self.cell.hop_eng
        if component not in [a+b for a in "xyz" for b in "xyz"]:
            raise ValueError(f"Illegal component {component}")
        comp = np.array(["xyz".index(_) for _ in component], dtype=np.int32)

        # Allocate working arrays
        num_kpt = self.kmesh_grid.shape[0]
        num_omega = self.omegas.shape[0]
        num_bands = self.cell.num_orb
        delta_eng = np.zeros((num_kpt, num_bands, num_bands), dtype=np.float64)
        prod_df = np.zeros((num_kpt, num_bands, num_bands), dtype=np.complex128)
        ac_cond = np.zeros(num_omega, dtype=np.complex128)

        # Build hopping indices and distances
        hop_ind = self.cell.hop_ind[:, 3:5].copy()
        hop_dr = self.cell.hop_dr_nm

        # Get eigenvalues and eigenstates
        # It is important not to synchronize the wave functions on each process.
        # Otherwise, OverflowError may be raised if the kmesh_grid is too dense.
        kmesh_frac = self.grid2frac(self.kmesh_grid)
        bands, states = self.calc_states(kmesh_frac, convention=1,
                                         all_reduce=False)

        # Transpose arrays for compatibility with FORTRAN backed
        # hop_eng, ac_cond are vectors and need no transposing.
        if use_fortran:
            bands = bands.T
            states = states.T
            hop_ind = hop_ind.T
            hop_dr = hop_dr.T
            kmesh = kmesh.T
            delta_eng = delta_eng.T
            prod_df = prod_df.T

        # Distribute k-points
        k_min, k_max = self.dist_bound(num_kpt)

        # Setup working arrays
        if use_fortran:
            # NOTE: FORTRAN array index begins from 1!
            hop_ind += 1
            comp += 1
            f2py.prod_ac(bands, states, hop_ind, hop_eng, hop_dr, kmesh,
                         self.beta, self.mu, comp, k_min+1, k_max+1,
                         delta_eng, prod_df)
        else:
            core.prod_ac(bands, states, hop_ind, hop_eng, hop_dr, kmesh,
                         self.beta, self.mu, comp, k_min, k_max,
                         delta_eng, prod_df)

        # Evaluate ac_cond
        if use_fortran:
            # NOTE: FORTRAN array index begins from 1!
            f2py.ac_cond_f(delta_eng, prod_df, self.omegas, self.delta,
                           k_min+1, k_max+1, ac_cond)
        else:
            core.ac_cond(delta_eng, prod_df, self.omegas, self.delta,
                         k_min, k_max, ac_cond)
        ac_cond = self.all_reduce(ac_cond)

        # Multiply prefactor
        # NOTE: there is not such g_s factor in the reference.
        if self.dimension == 2:
            area = self.cell.get_lattice_area("c")
            prefactor = self.g_s * 1j / (area * len(self.kmesh_grid))
        elif self.dimension == 3:
            volume = self.cell.get_lattice_volume()
            prefactor = self.g_s * 1j / (volume * len(self.kmesh_grid))
        else:
            raise NotImplementedError(f"Dimension {self.dimension} not "
                                      f"implemented")
        ac_cond *= prefactor
        ac_cond /= self.cell.extended
        return self.omegas, ac_cond
