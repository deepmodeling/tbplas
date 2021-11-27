"""
Functions and classes for analyzing correlation functions.

Functions
----------
    window_hanning: user function
        Hanning window
    window_exp: user function
        Exponential window
    window_exp_ten: user function
        Window function given by exponential of 10

Classes
-------
    Analyzer: user class
        wrapper over analyzing tools
"""

from math import cos, sin, exp

import numpy as np
import numpy.linalg as npla
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from .builder import Sample
from .config import Config
from .solver import BaseSolver
from .fortran import f2py


def window_hanning(i, tnr):
    """
    Hanning window function.

    :param i: integer
        summation index
    :param tnr: integer
        total length of summation
    :return: float
        Hanning window value
    """
    return 0.5 * (1 + cos(np.pi * i / tnr))


def window_exp(i, tnr):
    """
    Exponential window function.

    :param i: integer
        summation index
    :param tnr: integer
        total length of summation
    :return: float
        exponential window value
    """
    return exp(-2. * (i / tnr)**2)


def window_exp_ten(i, tnr):
    """
    Exponential window function with base 10.

    :param i: integer
        summation index
    :param tnr: integer
        total length of summation
    :return: float
        exponential window value
    """
    power = -2 * (1. * i / tnr)**2
    return 10.**power


class Analyzer(BaseSolver):
    """Class for analyzing correlation functions."""
    def __init__(self, sample: Sample, config: Config, enable_mpi=False):
        """
        :param sample: instance of 'Sample' class
            sample for which TBPM calculations will be performed
        :param config: instance of 'Config' class
            parameters controlling TBPM calculations
        :param enable_mpi: boolean
            whether to enable parallelism using MPI
        """
        super().__init__(sample, config, enable_mpi)

    def calc_dos(self, corr_dos, window=window_hanning):
        """
        Calculate DOS from correlation function.

        :param corr_dos: (nr_time_steps+1,) complex128 array
            DOS correlation function
        :param window: function, optional
            window function for integral
        :return: energies: (2*nr_time_steps,) float64 array
            energy values
        :return: dos: (2*nr_time_steps,) float64 array
            DOS values corresponding to energies
        """
        if self.rank == 0:
            # Get parameters
            tnr = self.config.generic['nr_time_steps']
            en_range = self.sample.energy_range
            en_step = 0.5 * en_range / tnr
            energies = np.array([0.5 * i * en_range / tnr - en_range / 2.
                                 for i in range(tnr * 2)], dtype=float)

            # Get negative time correlation
            corr_neg_time = np.empty(tnr * 2, dtype=complex)
            corr_neg_time[tnr - 1] = corr_dos.item(0)
            corr_neg_time[2 * tnr - 1] = window(tnr - 1, tnr) * corr_dos.item(tnr)
            for i in range(tnr - 1):
                corr_neg_time[tnr + i] = window(i, tnr) * corr_dos.item(i + 1)
                corr_neg_time[tnr-i-2] = window(i, tnr) * corr_dos.item(i + 1).conjugate()

            # Fourier transform
            corr_fft = np.fft.ifft(corr_neg_time)
            dos = np.empty(tnr * 2)
            for i in range(tnr):
                dos[i + tnr] = abs(corr_fft.item(i))
            for i in range(tnr, 2 * tnr):
                dos[i - tnr] = abs(corr_fft.item(i))

            # Normalise and correct for spin
            dos = dos / (np.sum(dos) * en_step)
            if self.config.generic['correct_spin']:
                dos = 2. * dos
        else:
            energies, dos = None, None
        return energies, dos

    def calc_ldos(self, corr_ldos, window=window_hanning):
        """
        Calculate LDOS from correlation function.

        :param corr_ldos: (nr_time_steps+1,) complex128 array
            LDOS correlation function
        :param window: function, optional
            window function for integral
        :return: energies: (2*nr_time_steps,) float64 array
            energy values
        :return: ldos: (2*nr_time_steps,) float64 array
            LDOS values corresponding to energies
        """
        return self.calc_dos(corr_ldos, window)

    def calc_ac_cond(self, corr_ac, window=window_exp):
        """
        Calculate AC conductivity from correlation function.

        :param corr_ac: (4, nr_time_steps) complex128 array
            AC correlation function in 4 directions:
            xx, xy, yx, yy, respectively
        :param window: function, optional
            window function for integral
        :return: omegas: (nr_time_steps,) float64 array
            omega values
        :return: ac: (4, nr_time_steps) float64 array
            ac conductivity values corresponding to omegas for 4 directions
            (xx, xy, yx, yy, respectively)
        """
        if self.rank == 0:
            # Get parameters
            tnr = self.config.generic['nr_time_steps']
            en_range = self.sample.energy_range
            t_step = np.pi / en_range
            beta = self.config.generic['beta']
            ac_prefactor = 4. * self.sample.nr_orbitals \
                / (self.sample.area_unit_cell * self.sample.extended)
            omegas = np.array([i * en_range / tnr for i in range(tnr)],
                              dtype=float)

            # Get AC conductivity
            ac = np.zeros((4, tnr), dtype=float)
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
                    ac[j, i] = acv

            # Correct for spin
            if self.config.generic['correct_spin']:
                ac = 2. * ac
        else:
            omegas, ac = None, None
        return omegas, ac

    def calc_ac_cond_imag(self, ac_cond_real):
        """
        Calculate imaginary part of the AC conductivity from real part
        using Kramers-Kronig relations (Hilbert transformation).

        :param ac_cond_real: (num_data,) float64 array
            real part of AC conductivity
        :return: ac_imag: (num_data,) float64 array
            imaginary part of AC conductivity
        """
        if self.rank == 0:
            num_data = len(ac_cond_real)
            sigma = np.zeros(2 * num_data, dtype=float)
            for i in range(num_data):
                sigma[num_data + i] = ac_cond_real.item(i)
                sigma[num_data - i] = ac_cond_real.item(i)
            ac_imag = np.imag(hilbert(sigma))[num_data:2 * num_data]
        else:
            ac_imag = None
        return ac_imag

    def calc_dyn_pol(self, corr_dyn_pol, window=window_exp_ten):
        """
        Calculate dynamical polarization from correlation function.

        :param corr_dyn_pol: (n_q_points, nr_time_steps) float64 array
            dynamical polarization correlation function
        :param window: function, optional
            window function for integral
        :return: q_points: (n_q_points, 3) float64 array
            coordinates of q-points
        :return: omegas: (nr_time_steps,) float64 array
            omega values
        :return: dyn_pol: (n_q_points, nr_time_steps) complex128 array
            dynamical polarization values corresponding to q-points and omegas
        """
        if self.rank == 0:
            # Get parameters
            tnr = self.config.generic['nr_time_steps']
            en_range = self.sample.energy_range
            t_step = np.pi / en_range
            q_points = np.array(self.config.dyn_pol['q_points'], dtype=float)
            n_q_points = len(q_points)
            # do we need to divide the prefactor by 1.5??
            dyn_pol_prefactor = -2. * self.sample.nr_orbitals \
                / (self.sample.area_unit_cell * self.sample.extended)
            n_omegas = tnr
            omegas = np.array([i * en_range / tnr for i in range(tnr)],
                              dtype=float)

            # get dynamical polarization
            dyn_pol = np.zeros((n_q_points, n_omegas), dtype=complex)
            for i_q in range(n_q_points):
                for i in range(n_omegas):
                    omega = omegas.item(i)
                    dpv = 0.0j
                    for k in range(tnr):
                        phi = k * t_step * omega
                        dpv += window(k + 1, tnr) * corr_dyn_pol.item(i_q, k) \
                            * (cos(phi) + 1j * sin(phi))
                    dyn_pol[i_q, i] = -dyn_pol_prefactor * t_step * dpv

            # correct for spin
            if self.config.generic['correct_spin']:
                dyn_pol = 2. * dyn_pol
        else:
            q_points, omegas, dyn_pol = None, None, None
        return q_points, omegas, dyn_pol

    def calc_epsilon(self, dyn_pol):
        """
        Calculate dielectric function from dynamical polarization.

        :param dyn_pol: (n_q_points, nr_time_steps) complex128 array
            dynamical polarization
        :return: epsilon: (n_q_points, nr_time_steps) complex128 array
            dielectric function
        """
        if self.rank == 0:
            # Get parameters
            tnr = self.config.generic['nr_time_steps']
            q_points = self.config.dyn_pol['q_points']
            n_q_points = len(q_points)
            epsilon_prefactor = self.config.dyn_pol['coulomb_constant'] \
                / self.config.dyn_pol['background_dielectric_constant'] \
                / self.sample.extended
            n_omegas = tnr

            # declare arrays
            epsilon = np.ones((n_q_points, n_omegas)) \
                + np.zeros((n_q_points, n_omegas)) * 0j
            v0 = epsilon_prefactor * np.ones(n_q_points)
            v = np.zeros(n_q_points)

            # calculate epsilon
            for i, q_point in enumerate(q_points):
                k = npla.norm(q_point)
                if k == 0.0:
                    v[i] = 0.
                else:
                    v[i] = v0.item(i) / k
                    epsilon[i, :] -= v.item(i) * dyn_pol[i, :]
        else:
            epsilon = None
        return epsilon

    def calc_dc_cond(self, corr_dos, corr_dc, window_dos=window_hanning,
                     window_dc=window_exp):
        """
        Calculate DC conductivity from its correlation function.

        :param corr_dos: (nr_time_steps,) complex128 array
            DOS correlation function
        :param corr_dc: (2, n_energies, nr_time_steps) complex128 array
            DC conductivity correlation function
        :param window_dos: function, optional
            window function for DOS integral
        :param window_dc: function, optional
            window function for DC integral
        :return: energies: (n_energies,) float64 array
            energy values
        :return: dc: (2, n_energies) float64 array
            dc conductivity values
        """
        # NOTE: Here we need to call analyze_corr_dos to obtain DOS, which is
        # intended to analyze the result of calc_corr_dos by design. As in
        # fortran/f2py.pyf, the results of calc_corr_dos and calc_corr_ldos
        # have the length of nr_time_steps+1, while that of calc_corr_dc has
        # length of nr_time_steps. This is due to incomplete update of the
        # source code. tbpm_dos and tbpm_ldos have been update, while other
        # subroutines are not. So here we need to insert 1.0 to the head of
        # corr_dos by calc_corr_dc before calling analyze_corr_dos.
        if self.rank == 0:
            corr_dos = np.insert(corr_dos, 0, 1.0)
            energies_dos, dos = self.calc_dos(corr_dos, window_dos)
            energies_dos = np.array(energies_dos)
            dos = np.array(dos)

            # Get parameters
            tnr = self.config.generic['nr_time_steps']
            en_range = self.sample.energy_range
            t_step = 2 * np.pi / en_range
            en_limit = self.config.DC_conductivity['energy_limits']
            qe_indices = np.where((energies_dos >= en_limit[0]) &
                                  (energies_dos <= en_limit[1]))[0]
            n_energies = len(qe_indices)
            energies = energies_dos[qe_indices]
            dc_prefactor = self.sample.nr_orbitals / self.sample.area_unit_cell

            # get DC conductivity
            dc = np.zeros((2, n_energies))
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
            if self.config.generic['correct_spin']:
                dc = 2. * dc
        else:
            energies, dc = None, None
        return energies, dc

    def calc_hall_cond(self, mu_mn):
        """
        Calculate Hall conductivity according to Kubo-Bastin formula mu_mn.

        :param mu_mn: (n_kernel, n_kernel) complex128 array
            output of self.calc_hall_mu
        :return: energies: float64 array
            energies
        :return: conductivity: float64 array
            Hall conductivity according to energies
        """
        if self.rank == 0:
            energies = np.array(self.config.dckb['energies'])
            dckb_prefactor = 16 * self.sample.nr_orbitals / \
                self.sample.area_unit_cell
            conductivity = f2py.cond_from_trace(
                mu_mn,
                self.config.dckb['energies'],
                self.sample.rescale,
                self.config.generic['beta'],
                self.config.dckb['ne_integral'],
                self.config.generic['Fermi_cheb_precision'],
                dckb_prefactor)
        else:
            energies, conductivity = None, None
        return energies, conductivity
