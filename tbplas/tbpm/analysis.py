"""Functions and classes for analyzing correlation functions."""

from math import cos, sin, exp, pi
from typing import Callable, Tuple

import numpy as np
from scipy.signal import hilbert
from scipy.integrate import trapz

from ..base import EPSILON0
from ..builder import Sample
from ..fortran import f2py
from ..parallel import MPIEnv
from .config import Config


__all__ = ["window_hanning", "window_exp", "window_exp_ten", "Analyzer"]


def window_hanning(i: int, tnr: int) -> float:
    """
    Hanning window function.

    :param i: summation index
    :param tnr: total length of summation
    :return: Hanning window value
    """
    return 0.5 * (1 + cos(pi * i / tnr))


def window_exp(i: int, tnr: int) -> float:
    """
    Exponential window function.

    :param i: summation index
    :param tnr: total length of summation
    :return: exponential window value
    """
    return exp(-2. * (i / tnr)**2)


def window_exp_ten(i: int, tnr: int) -> float:
    """
    Exponential window function with base 10.

    :param i: summation index
    :param tnr: total length of summation
    :return: exponential window value
    """
    power = -2 * (1. * i / tnr)**2
    return 10.**power


class Analyzer(MPIEnv):
    """
    Class for analyzing correlation functions.

    Attributes
    ----------
    _sample: 'Sample' instance
        sample for which TBPM calculations will be performed
    _config: 'Config' instance
        parameters controlling TBPM calculation
    _dimension: int
        dimension of the system
    """
    def __init__(self, sample: Sample,
                 config: Config,
                 dimension: int = 3,
                 enable_mpi: bool = False,
                 echo_details: bool = False) -> None:
        """
        :param sample: sample for which TBPM calculations will be performed
        :param config: parameters controlling TBPM calculations
        :param dimension: dimension of the sample
        :param enable_mpi: whether to enable parallelism using MPI
        :param echo_details: whether to output parallelization details
        :raises ValueError: if dimension is neither 2 nor 3, or if illegal
            parameters are detected in config
        """
        super().__init__(enable_mpi=enable_mpi, echo_details=echo_details)
        self._sample = sample
        self._config = config
        self._config.check_params()
        if dimension not in (2, 3):
            raise ValueError(f"Unsupported dimension: {dimension}")
        self._dimension = dimension

    def calc_dos(self, corr_dos: np.ndarray,
                 window: Callable = window_hanning) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate DOS from correlation function.

        Reference: eqn. 16-17 of feature article.

        The unit of dos follows:
            [dos] = [C_DOS] [dt] = h_bar / eV
        So possibly the formula misses a h_bar on the denominator.
        Anyway, the DOS is correct since it is explicitly normalized to 1.

        :param corr_dos: (nr_time_steps+1,) complex128 array
            dimensionless DOS correlation function
        :param window: window function for integral
        :return: (energies, dos)
            energies: (2*nr_time_steps,) float64 array
            energies in eV
            dos: (2*nr_time_steps,) float64 array
            DOS in 1/eV
        """
        # Get parameters
        tnr = self._config.generic['nr_time_steps']
        en_range = self._sample.energy_range
        en_step = 0.5 * en_range / tnr

        # Allocate working arrays
        energies = np.array([0.5 * i * en_range / tnr - en_range / 2.
                             for i in range(tnr * 2)], dtype=float)
        dos = np.zeros(tnr * 2, dtype=float)

        if self.is_master:
            # Get negative time correlation
            corr_neg_time = np.empty(tnr * 2, dtype=complex)
            corr_neg_time[tnr - 1] = corr_dos.item(0)
            corr_neg_time[2 * tnr - 1] = window(tnr - 1, tnr) * corr_dos.item(tnr)
            for i in range(tnr - 1):
                corr_neg_time[tnr + i] = window(i, tnr) * corr_dos.item(i + 1)
                corr_neg_time[tnr-i-2] = window(i, tnr) * corr_dos.item(i + 1).conjugate()

            # Fourier transform
            corr_fft = np.fft.ifft(corr_neg_time)
            for i in range(tnr):
                dos[i + tnr] = abs(corr_fft.item(i))
            for i in range(tnr, 2 * tnr):
                dos[i - tnr] = abs(corr_fft.item(i))

            # Normalise and correct for spin
            dos = dos / (np.sum(dos) * en_step)
            if self._config.generic['correct_spin']:
                dos = 2. * dos
        else:
            pass
        self.bcast(dos)
        return energies, dos

    def calc_ldos(self, corr_ldos: np.ndarray,
                  window: Callable = window_hanning) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate LDOS from correlation function.

        :param corr_ldos: (nr_time_steps+1,) complex128 array
            dimensionless LDOS correlation function
        :param window: window function for integral
        :return: (energies, ldos)
            energies: (2*nr_time_steps,) float64 array
            energies in eV
            ldos: (2*nr_time_steps,) float64 array
            LDOS in 1/eV
        """
        return self.calc_dos(corr_ldos, window)

    def calc_ac_cond(self, corr_ac: np.ndarray,
                     window: Callable = window_exp) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate optical (AC) conductivity from correlation function.

        Reference: eqn. 300-301 of graphene note.

        The unit of AC conductivity in 2d case follows:
        [sigma] = [1/(h_bar * omega * A)] * [j^2] * [dt]
                = 1/(eV*nm^2) * e^2/h_bar^2 * (eV)^2 * nm^2 * h_bar/eV
                = e^2/h_bar
        which is consistent with the results from Lindhard function.

        The reason for nr_orbitals in the prefactor is that every electron
        contribute freely to the conductivity, and we have to take the number
        of electrons into consideration. See eqn. 222-223 of the note for more
        details.

        :param corr_ac: (4, nr_time_steps) complex128 array
            AC correlation function in 4 directions:
            xx, xy, yx, yy, respectively
            Unit should be e^2/h_bar^2 * (eV)^2 * nm^2.
        :param window: window function for integral
        :return: (omegas, ac_cond)
            omegas: (nr_time_steps,) float64 array
            frequencies in eV
            ac_cond: (4, nr_time_steps) complex128 array
            ac conductivity values corresponding to omegas for 4 directions
            (xx, xy, yx, yy, respectively)
            The unit is e^2/(h_bar*nm) in 3d case and e^2/h_bar in 2d case.
        """
        # Get parameters
        tnr = self._config.generic['nr_time_steps']
        en_range = self._sample.energy_range
        t_step = pi / en_range
        beta = self._config.generic['beta']
        if self._dimension == 2:
            ac_prefactor = self._sample.nr_orbitals \
                / (self._sample.area_unit_cell * self._sample.extended)
        else:
            ac_prefactor = self._sample.nr_orbitals \
                / (self._sample.volume_unit_cell * self._sample.extended)

        # Allocate working arrays
        omegas = np.array([i * en_range / tnr for i in range(tnr)],
                          dtype=float)
        ac_cond = np.zeros((4, tnr), dtype=complex)

        if self.is_master:
            # Get real part of AC conductivity
            ac_real = np.zeros((4, tnr), dtype=float)
            for j in range(4):
                for i in range(tnr):
                    omega = omegas.item(i)
                    acv = 0.
                    for k in range(tnr):
                        acv += 2. * window(k + 1, tnr) \
                            * sin(omega * k * t_step) * corr_ac.item(j, k).imag
                    if omega == 0.:
                        acv = 0.
                    else:
                        acv = ac_prefactor * t_step * acv \
                            * (exp(-beta * omega) - 1) / omega
                    ac_real[j, i] = acv

            # Get imaginary part of AC conductivity via Kramers-Kronig relations
            # (Hilbert transformation).
            ac_imag = np.zeros((4, tnr), dtype=float)
            for j in range(4):
                sigma = np.zeros(2 * tnr, dtype=float)
                for i in range(tnr):
                    sigma[tnr + i] = ac_real.item(j, i)
                    sigma[tnr - i] = ac_real.item(j, i)
                ac_imag[j, :] = np.imag(hilbert(sigma))[tnr:2 * tnr]
            ac_cond = ac_real + 1j * ac_imag

            # Correct for spin
            if self._config.generic['correct_spin']:
                ac_cond = 2. * ac_cond
        else:
            pass
        self.bcast(ac_cond)
        return omegas, ac_cond

    def calc_dyn_pol(self, corr_dyn_pol: np.ndarray,
                     window: Callable = window_exp_ten) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate dynamical polarization from correlation function.

        Reference:
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.035439

        The unit of dp in 2d case follows:
        [dp] = [1/A] * [C_DP] * [dt]
             = 1/nm^2 * 1 * h_bar/eV
             = h_bar/(eV*nm^2)
        which is inconsistent with the output of Lindhard! So, possibly the
        formula misses a h_bar on the denominator.

        :param corr_dyn_pol: (n_q_points, nr_time_steps) float64 array
            dimensionless dynamical polarization correlation function
        :param window: window function for integral
        :return: (q_points, omegas, dyn_pol)
            q_points: (n_q_points, 3) float64 array
            Cartesian coordinates of q-points in 1/nm
            omegas: (nr_time_steps,) float64 array
            frequencies in eV
            dyn_pol: (n_q_points, nr_time_steps) complex128 array
            dynamical polarization values corresponding to q-points and omegas
            The unit is 1/(eV*nm^2) or 1/(eV*nm^3) depending on the dimension.
        """
        # Get parameters
        tnr = self._config.generic['nr_time_steps']
        en_range = self._sample.energy_range
        t_step = pi / en_range
        q_points = np.array(self._config.dyn_pol['q_points'], dtype=float)
        if self._dimension == 2:
            dyn_pol_prefactor = 2. * self._sample.nr_orbitals \
                / (self._sample.area_unit_cell * self._sample.extended)
        else:
            dyn_pol_prefactor = 2. * self._sample.nr_orbitals \
                / (self._sample.volume_unit_cell * self._sample.extended)

        # Allocate working arrays
        n_q_points = len(q_points)
        n_omegas = tnr
        omegas = np.array([i * en_range / tnr for i in range(tnr)],
                          dtype=float)
        dyn_pol = np.zeros((n_q_points, n_omegas), dtype=complex)

        if self.is_master:
            for i_q in range(n_q_points):
                for i in range(n_omegas):
                    omega = omegas.item(i)
                    dpv = 0.0j
                    for k in range(tnr):
                        phi = k * t_step * omega
                        dpv += window(k + 1, tnr) * corr_dyn_pol.item(i_q, k) \
                            * (cos(phi) + 1j * sin(phi))
                    dyn_pol[i_q, i] = dyn_pol_prefactor * t_step * dpv

            # correct for spin
            if self._config.generic['correct_spin']:
                dyn_pol = 2. * dyn_pol
        else:
            pass
        self.bcast(dyn_pol)
        return q_points, omegas, dyn_pol

    def _get_vq(self, q_point: np.ndarray) -> float:
        """
        Get Coulomb interaction in momentum space for given q-point.

        :param q_point: (3,) float64 array
            CARTESIAN coordinate of q-point in 1/NM
        :return: Coulomb interaction in momentum space in eV
        """
        factor = self._config.dyn_pol['coulomb_constant']
        back_epsilon = self._config.dyn_pol['background_dielectric_constant']
        factor /= (EPSILON0 * back_epsilon)
        q_norm = np.linalg.norm(q_point)
        if q_norm == 0.0:
            vq = 0.0
        else:
            if self._dimension == 2:
                vq = factor * 2 * pi / q_norm
            else:
                vq = factor * 4 * pi / q_norm**2
        return vq

    def calc_epsilon(self, dyn_pol: np.ndarray) -> np.ndarray:
        """
        Calculate dielectric function from dynamical polarization.

        :param dyn_pol: (n_q_points, nr_time_steps) complex128 array
            dynamical polarization
        :return: (n_q_points, nr_time_steps) complex128 array
            dielectric function
        """
        # Get parameters
        tnr = self._config.generic['nr_time_steps']
        q_points = self._config.dyn_pol['q_points']
        n_q_points = len(q_points)
        n_omegas = tnr
        epsilon = np.zeros((n_q_points, n_omegas), dtype=complex)

        if self.is_master:
            for i, q_point in enumerate(q_points):
                vq = self._get_vq(q_point)
                epsilon[i] = 1 - vq * dyn_pol[i]
        else:
            pass
        self.bcast(epsilon)
        return epsilon

    def calc_dc_cond(self, corr_dos: np.ndarray,
                     corr_dc: np.ndarray,
                     window_dos: Callable = window_hanning,
                     window_dc: Callable = window_exp) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate electronic (DC) conductivity at zero temperature from its
        correlation function.

        Reference: eqn. 381 of graphene note.

        The unit of DC conductivity in 2d case follows:
        [sigma] = [D/A] * [j^2] * [dt]
                = 1/(eV*nm^2) * e^2/h_bar^2 * (eV)^2 * nm^2 * h_bar/eV
                = e^2/h_bar
        which is consistent with the result of Lindhard function.

        NOTE: the xy and yx components of conductivity are not accurate. So
        they will not be evaluated.

        NOTE: Here we need to call analyze_corr_dos to obtain DOS, which is
        intended to analyze the result of calc_corr_dos by design. As in
        the fortran extension, the results of calc_corr_dos and calc_corr_ldos
        have the length of nr_time_steps+1, while that of calc_corr_dc has
        length of nr_time_steps. This is due to incomplete update of the
        source code. tbpm_dos and tbpm_ldos have been update, while other
        subroutines are not. So here we need to insert 1.0 to the head of
        corr_dos by calc_corr_dc before calling analyze_corr_dos.

        :param corr_dos: (nr_time_steps,) complex128 array
            dimensionless DOS correlation function
        :param corr_dc: (2, n_energies, nr_time_steps) complex128 array
            DC conductivity correlation function in e^2/h_bar^2 * (eV)^2 * nm^2
        :param window_dos: window function for DOS integral
        :param window_dc: window function for DC integral
        :return: (energies, dc)
            energies: (n_energies,) float64 array
            energies in eV
            dc: (2, n_energies) float64 array
            dc conductivity values for xx and yy directions in the same unit
            as ac conductivity
        """
        # Get DOS
        corr_dos = np.insert(corr_dos, 0, 1.0)
        energies_dos, dos = self.calc_dos(corr_dos, window_dos)
        energies_dos = np.array(energies_dos)
        dos = np.array(dos)

        # Get parameters
        tnr = self._config.generic['nr_time_steps']
        en_range = self._sample.energy_range
        t_step = 2 * pi / en_range
        en_limit = self._config.DC_conductivity['energy_limits']
        qe_indices = np.where((energies_dos >= en_limit[0]) &
                              (energies_dos <= en_limit[1]))[0]
        n_energies = len(qe_indices)
        energies = energies_dos[qe_indices]
        if self._dimension == 2:
            dc_prefactor = self._sample.nr_orbitals / \
                           self._sample.area_unit_cell
        else:
            dc_prefactor = self._sample.nr_orbitals / \
                           self._sample.volume_unit_cell
        dc = np.zeros((2, n_energies))

        if self.is_master:
            for i in range(2):
                for j in range(n_energies):
                    en = energies.item(j)
                    dos_val = dos.item(qe_indices.item(j))
                    dc_val = 0.
                    for k in range(tnr):
                        w = window_dc(k + 1, tnr)
                        phi = k * t_step * en
                        c_exp = cos(phi) - 1j * sin(phi)
                        add_dcv = w * (c_exp * corr_dc.item(i, j, k)).real
                        dc_val += add_dcv
                    dc[i, j] = dc_prefactor * t_step * dos_val * dc_val
            # correct for spin
            if self._config.generic['correct_spin']:
                dc = 2. * dc
        else:
            pass
        self.bcast(dc)
        return energies, dc

    def calc_diff_coeff(self, corr_dc: np.ndarray,
                        window_dc: Callable = window_exp) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate diffusion coefficient form DC correlation function.

        Reference: eqn. 43-44 of feature article.

        The unit of diff_coeff follows:
        [diff_coeff] = [1/e^2] * [j^2] * [dt]
                     = 1/e^2 * e^2/h_bar^2 * (eV)^2 * nm^2 * h_bar/eV
                     = eV*nm^2/h_bar
        which does not depend on system dimension.

        :param corr_dc: (2, n_energies, nr_time_steps) complex128 array
            DC conductivity correlation function in e^2/h_bar^2 * (eV)^2 * nm^2
        :param window_dc: window function for DC integral
        :return: (time, diff_coeff)
            time: (nr_time_steps,) float64 array
            time for diffusion coefficient in h_bar/eV
            diff_coeff: (2, n_energies, nr_time_steps) complex128 array
            diffusion coefficient in nm^2/(h_bar/eV)
        """
        # Get parameters
        tnr = self._config.generic['nr_time_steps']
        en_range = self._sample.energy_range
        t_step = 2 * pi / en_range
        en_limit = self._config.DC_conductivity['energy_limits']
        energies = np.array([0.5 * i * en_range / tnr - en_range / 2.
                             for i in range(tnr * 2)], dtype=float)
        qe_indices = np.where((energies >= en_limit[0]) &
                              (energies <= en_limit[1]))[0]
        n_energies = len(qe_indices)
        energies = energies[qe_indices]
        time = np.linspace(0, tnr - 1, tnr) * t_step
        diff_coeff = np.zeros((2, n_energies, tnr))

        if self.is_master:
            for i in range(2):
                for j in range(n_energies):
                    en = energies.item(j)
                    temp = np.zeros(tnr)
                    for k in range(tnr):
                        w = window_dc(k + 1, tnr)
                        phi = k * t_step * en
                        c_exp = cos(phi) - 1j * sin(phi)
                        temp[k] = w * (c_exp * corr_dc.item(i, j, k)).real
                    for k2 in range(tnr):
                        diff_coeff[i, j, k2] = trapz(temp[:k2 + 1], time[:k2 + 1])
        else:
            pass
        self.bcast(diff_coeff)
        return time, diff_coeff

    def calc_hall_cond(self, mu_mn: np.ndarray,
                       unit: str = "h_bar") -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Hall conductivity according to Kubo-Bastin formula mu_mn.

        Reference:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.116602

        The unit of hall_cond in 2d case from eqn. 1 of the reference follows:
        [hall_cond] = [h_bar*e^2/Omega] * [dE * Tr<...>]
                    = h_bar*e^2/nm^2 * 1/eV * nm^2/h_bar^2 * eV
                    = e^2/h_bar
        which is consistent with AC and DC conductivity.
        Note that the delta function in the formula has the unit of 1/eV.

        The unit can also be determined from enq. 4 as:
        [hall_cond] = [e^2*h_bar/(Omega*delaE^2)] * [mu]
                    = h_bar/(nm^2*V^2) * nm^2 * (eV)^2 / h_bar^2
                    = e^2/h_bar
        Note that the scaled energy is dimensionless.

        :param mu_mn: (n_kernel, n_kernel) complex128 array
            output of solver.calc_hall_mu in nm^2/h_bar^2 * (eV)^2
        :param unit: unit of Hall conductivity, set to 'h_bar' to use
            'e^2/h_bar' and 'h' to use 'e^2/h'
        :return: (energies, conductivity)
            energies: float64 array
            chemical potentials specified in config.dckb['energies']
            conductivity: float64 array
            Hall conductivity according to energies
        :raise ValueError: if unit is neither 'h_bar' nor 'h'
        """
        if unit not in ("h_bar", "h"):
            raise ValueError(f"Illegal unit {unit}")
        energies = np.array(self._config.dckb['energies'])
        if self.is_master:
            dckb_prefactor = 16 * self._sample.nr_orbitals / \
                             (pi * self._sample.energy_range ** 2)
            if self._dimension == 2:
                dckb_prefactor /= self._sample.area_unit_cell
            else:
                dckb_prefactor /= self._sample.volume_unit_cell
            if unit == "h":
                dckb_prefactor *= (2 * pi)
            conductivity = f2py.cond_from_trace(
                mu_mn,
                self._config.dckb['energies'],
                self._sample.rescale,
                self._config.generic['beta'],
                self._config.dckb['ne_integral'],
                self._config.generic['Fermi_cheb_precision'],
                self.rank)
            conductivity *= dckb_prefactor

            # correct for spin
            if self._config.generic['correct_spin']:
                conductivity *= 2
        else:
            conductivity = np.zeros(len(energies), dtype=float)
        self.bcast(conductivity)
        return energies, conductivity
