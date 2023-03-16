"""
Functions and classes for TBPM calculation.

Functions
---------
    None

Classes
-------
    Solver: user class
        wrapper class over FORTRAN TBPM subroutines
"""

import time
import os
import pickle
import math

import numpy as np
import scipy.special as spec

from .builder import Sample
from .builder.constants import H_BAR_EV
from .config import Config
from .fortran import f2py
from .parallel import MPIEnv


class Solver(MPIEnv):
    """
    Wrapper class over FORTRAN TBPM subroutines.

    Attributes
    ----------
    sample: instance of 'Sample' class
        sample for which TBPM calculations will be performed
    config: instance of 'Config' class
        parameters controlling TBPM calculation
    output['directory'] : string
        Output directory.
        Default value: "sim_data".
    output['prefix'] : string
        Prefix prepended to the output files.
        Default value: timestamp.
    output['corr_AC'] : string
        AC conductivity correlation output file.
        Default value: f"sim_data/{timestamp}.corr_AC".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    output['corr_DC'] : string
        DC conductivity correlation output file.
        Default value: f"sim_data/{timestamp}.corr_DC".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    output['corr_DOS'] : string
        DOS correlation output file.
        Default value: f"sim_data/{timestamp}.corr_DOS".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    output['corr_LDOS'] : string
        LDOS correlation output file.
        Default value: f"sim_data/{timestamp}.corr_LDOS".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    output['corr_dyn_pol'] : string
        AC conductivity correlation output file.
        Default value: f"sim_data/{timestamp}.corr_dyn_pol".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    output['hall_mu'] : string
        mu_{mn} output file.
        Default value: f"sim_data/{timestamp}.hall_mu".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    output['qe'] : string
        Quasi-eigenstates output file.
        Default value: f"sim_data/{timestamp}.qe".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    output['ldos_haydock'] : string
        LDOS output file from Haydock method.
        Default value: f"sim_data/{timestamp}.ldos_haydock".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    output['psi_t'] : string
        Time-dependent wave function output file.
        Default value: f"sim_data/{timestamp}.psi_t".
        The actual output file will have a suffix of 'npy' or 'h5' depending on
        the output format.
    """
    def __init__(self, sample: Sample, config: Config, enable_mpi=False,
                 **kwargs):
        """
        :param sample: instance of 'Sample' class
            sample for which TBPM calculations will be performed
        :param config: instance of 'Config' class
            parameters controlling TBPM calculations
        :param enable_mpi: boolean
            whether to enable parallelism using MPI
        :param kwargs: dictionary
            arguments for method 'set_output'
        :raises ValueError: if illegal parameters are detected in config
        """
        super().__init__(enable_mpi=enable_mpi, echo_details=True)
        self.sample = sample
        self.config = config
        self.config.check_params()
        self.output = dict()
        self.set_output(**kwargs)

    def set_output(self, directory="sim_data", prefix=None):
        """
        Setup directory and file names for output.

        :param directory: string
            directory for writing files
        :param prefix: string
             prefix prepended to output files
        :return: None
            self.output is modified.
        """
        # Set directory and prefix
        self.output["directory"] = directory
        if prefix is not None:
            self.output["prefix"] = prefix
        else:
            self.output["prefix"] = str(int(time.time()))

        # Create directory if not exits
        # NOTE: On some file systems, e.g. btrfs, creating directory may be
        # slow. Other processes may try to access the directory before it is
        # created and run into I/O errors. So we need to put a barrier here.
        if self.is_master:
            if not os.path.exists(self.output["directory"]):
                os.mkdir(self.output["directory"])
        self.barrier()

        # Determine file names
        for key in ("corr_DOS", "corr_LDOS", "corr_AC", "corr_dyn_pol",
                    "corr_DC", "hall_mu", "qe", "ldos_haydock", "psi_t"):
            self.output[key] = "%s/%s.%s" % (self.output["directory"],
                                             self.output["prefix"], key)

        # Print output details
        self.print("Output details:")
        self.print("%11s: %s" % ("Directory", self.output["directory"]))
        self.print("%11s: %s" % ("Prefix", self.output["prefix"]))
        self.print()

        # Force saving config
        self.save_config()

    def save_config(self, filename="config.pkl"):
        """
        Save self.config to a .pkl file.

        :param filename: string
            file name of the .pkl file
        :return: None
        """
        if self.is_master:
            pickle_name = "%s/%s.%s" % (self.output["directory"],
                                        self.output["prefix"], filename)
            with open(pickle_name, 'wb') as f:
                pickle.dump(self.config, f, pickle.HIGHEST_PROTOCOL)

    def __dist_sample(self):
        """
        Common interface to distribute samples among MPI processes.

        :return: num_sample: integer
            number of random samples assigned to this process
            self.config.generic["nr_random_samples"] is updated.
        """
        num_sample_opt = self.config.generic["nr_random_samples"]
        if self.mpi_enabled:
            while num_sample_opt % self.size != 0:
                num_sample_opt += 1
            if num_sample_opt != self.config.generic["nr_random_samples"]:
                self.print("\nWARNING:\n  nr_random_samples adjusted to %d for"
                           " optimal balance" % num_sample_opt)
                self.config.generic["nr_random_samples"] = num_sample_opt
            num_sample = num_sample_opt // self.size
        else:
            num_sample = num_sample_opt
        return num_sample

    def __get_time_step(self):
        """
        Get the time step for TBPM calculation via:
            time_step = 2 * pi / sample.energy_range

        :return: time_step: float
            time step in h_bar/eV
        """
        return 2 * math.pi / self.sample.energy_range

    def __echo_time_step_fs(self, time_step):
        """
        Convert time step from h_bar/eV to femto-second and report.

        :param time_step: float
            time step in h_bar/eV
        :return: None.
            Result are printed to stdout.
        """
        time_step_fs = time_step * H_BAR_EV * 1e15
        self.print("Time step for propagation: %7.3f fs\n" % time_step_fs)

    def __get_bessel_series(self, time_step):
        """
        Get the values of Bessel functions up to given order.

        :param time_step: float
            time step
        :return: bes: list of floats
            values of Bessel functions
        :raises ValueError: if self.config.generic["Bessel_max"] is too low
        """
        bessel_max = self.config.generic["Bessel_max"]
        bessel_precision = self.config.generic["Bessel_precision"]
        time_scaled = time_step * self.sample.rescale

        # Collect bessel function values
        bessel_series = []
        converged = False
        for i in range(bessel_max):
            bes_val = spec.jv(i, time_scaled)
            if np.abs(bes_val) > bessel_precision:
                bessel_series.append(bes_val)
            else:
                bes_val_up = spec.jv(i + 1, time_scaled)
                if np.abs(bes_val_up) > bessel_precision:
                    bessel_series.append(bes_val)
                else:
                    converged = True
                    break

        # Check and return results
        if not converged:
            raise ValueError("Bessel_max too low")
        return bessel_series

    def __get_beta_mu_re(self):
        """
        Get beta_re and mu_re for ac, dyn_pol and dc calculations.

        :return: beta_re: float
        :return: mu_re: float
        """
        beta_re = self.config.generic['beta'] * self.sample.rescale
        mu_re = self.config.generic['mu'] / self.sample.rescale
        return beta_re, mu_re

    def __save_data(self, data, file_name, output_format="numpy"):
        """
        Save array data to file.

        :param data: numpy array
            data to save
        :param file_name: string
            name of file to which data will be saved WITHOUT suffix.
            Suffix will be added automatically depending on the format.
        :param output_format: string
            format of output data
            For now only numpy is supported.
        :return: None
        :raises NotImplementedError: if output_format is not "numpy" or "npy"
        """
        if output_format in ("numpy", "npy"):
            suffix = "npy"
        else:
            raise NotImplementedError(f"Unsupported format {output_format}")
        if self.is_master:
            np.save(f"{file_name}.{suffix}", data)

    def calc_corr_dos(self):
        """
        Calculate correlation function of density of states (DOS).

        :return: corr_dos: (nr_time_steps+1,) complex128 array
            dimensionless DOS correlation function
        """
        # Get parameters
        time_step = self.__get_time_step()
        bessel_series = self.__get_bessel_series(time_step)
        ham_csr = self.sample.build_ham_csr()
        num_sample = self.__dist_sample()
        self.__echo_time_step_fs(time_step)

        # Call FORTRAN subroutine
        corr_dos = f2py.tbpm_dos(
            bessel_series,
            ham_csr.indptr, ham_csr.indices, ham_csr.data,
            self.config.generic['seed'],
            self.config.generic['nr_time_steps'],
            num_sample,
            self.output['corr_DOS'],
            self.rank,
            self.config.generic['wfn_check_steps'],
            self.config.generic['wfn_check_thr']
        )
        corr_dos = self.average(corr_dos)
        self.__save_data(corr_dos, self.output["corr_DOS"])
        return corr_dos

    def calc_corr_ldos(self):
        """
        Calculate correlation function of local density of states (DOS).

        :return: corr_ldos: (nr_time_steps+1,) complex128 array
            dimensionless LDOS correlation function
        """
        # Get parameters
        time_step = self.__get_time_step()
        bessel_series = self.__get_bessel_series(time_step)
        ham_csr = self.sample.build_ham_csr()
        num_sample = self.__dist_sample()
        self.__echo_time_step_fs(time_step)

        # Call FORTRAN subroutine
        corr_ldos = f2py.tbpm_ldos(
            self.config.LDOS['site_indices'],
            bessel_series,
            ham_csr.indptr, ham_csr.indices, ham_csr.data,
            self.config.generic['seed'],
            self.config.generic['nr_time_steps'],
            num_sample,
            self.output['corr_LDOS'],
            self.rank,
            self.config.generic['wfn_check_steps'],
            self.config.generic['wfn_check_thr']
        )
        corr_ldos = self.average(corr_ldos)
        self.__save_data(corr_ldos, self.output["corr_LDOS"])
        return corr_ldos

    def calc_corr_ac_cond(self):
        """
        Calculate correlation function of optical (AC) conductivity.

        :return: corr_ac: (4, nr_time_steps) complex128 array
            AC correlation function in 4 directions:
            xx, xy, yx, yy, respectively
            Unit is e^2/h_bar^2 * (eV)^2 * nm^2.
        """
        # Get parameters
        time_step = 0.5 * self.__get_time_step()
        bessel_series = self.__get_bessel_series(time_step)
        beta_re, mu_re = self.__get_beta_mu_re()
        indptr, indices, hop, dx, dy = self.sample.build_ham_dxy()
        num_sample = self.__dist_sample()
        self.__echo_time_step_fs(time_step)

        # Call FORTRAN subroutine
        corr_ac = f2py.tbpm_accond(
            bessel_series, beta_re, mu_re,
            indptr, indices, hop,
            self.sample.rescale,
            dx, dy,
            self.config.generic['seed'],
            self.config.generic['nr_time_steps'],
            num_sample,
            self.config.generic['nr_Fermi_fft_steps'],
            self.config.generic['Fermi_cheb_precision'],
            self.output['corr_AC'],
            self.rank,
            self.config.generic['wfn_check_steps'],
            self.config.generic['wfn_check_thr']
        )
        corr_ac = self.average(corr_ac)
        self.__save_data(corr_ac, self.output["corr_AC"])
        return corr_ac

    def calc_corr_dyn_pol(self):
        """
        Calculate correlation function of dynamical polarization.

        :return: corr_dyn_pol: (n_q_points, nr_time_steps) float64 array
            domensionless Dynamical polarization correlation function.
        """
        # Get parameters
        time_step = 0.5 * self.__get_time_step()
        bessel_series = self.__get_bessel_series(time_step)
        beta_re, mu_re = self.__get_beta_mu_re()
        indptr, indices, hop, dx, dy = self.sample.build_ham_dxy()
        site_x = self.sample.orb_pos[:, 0]
        site_y = self.sample.orb_pos[:, 1]
        site_z = self.sample.orb_pos[:, 2]
        num_sample = self.__dist_sample()
        self.__echo_time_step_fs(time_step)

        # Call FORTRAN subroutine
        corr_dyn_pol = f2py.tbpm_dyn_pol(
            bessel_series, beta_re, mu_re,
            indptr, indices, hop,
            self.sample.rescale,
            dx, dy, site_x, site_y, site_z,
            self.config.generic['seed'],
            self.config.generic['nr_time_steps'],
            num_sample,
            self.config.generic['nr_Fermi_fft_steps'],
            self.config.generic['Fermi_cheb_precision'],
            self.config.dyn_pol['q_points'],
            self.output['corr_dyn_pol'],
            self.rank,
            self.config.generic['wfn_check_steps'],
            self.config.generic['wfn_check_thr']
        )
        corr_dyn_pol = self.average(corr_dyn_pol)
        self.__save_data(corr_dyn_pol, self.output["corr_dyn_pol"])
        return corr_dyn_pol

    def calc_corr_dc_cond(self):
        """
        Calculate correlation function of electronic (DC) conductivity.

        :return: corr_dos: (nr_time_steps,) complex128 array
            dimensionless DOS correlation function
        :return: corr_dc: (2, n_energies, nr_time_steps) complex128 array
            DC conductivity correlation function in e^2/h_bar^2 * (eV)^2 * nm^2
        """
        # Get parameters
        tnr = self.config.generic['nr_time_steps']
        en_range = self.sample.energy_range
        energies_dos = np.array([0.5 * i * en_range / tnr - en_range / 2.
                                 for i in range(tnr * 2)])

        en_limit = self.config.DC_conductivity['energy_limits']
        qe_indices = np.where((energies_dos >= en_limit[0]) &
                              (energies_dos <= en_limit[1]))[0]

        beta_re, mu_re = self.__get_beta_mu_re()

        time_step = self.__get_time_step()
        bessel_series = self.__get_bessel_series(time_step)
        indptr, indices, hop, dx, dy = self.sample.build_ham_dxy()
        num_sample = self.__dist_sample()
        self.__echo_time_step_fs(time_step)

        # Call FORTRAN subroutine
        corr_dos, corr_dc = f2py.tbpm_dccond(
            bessel_series, beta_re, mu_re,
            indptr, indices, hop,
            self.sample.rescale,
            dx, dy,
            self.config.generic['seed'],
            self.config.generic['nr_time_steps'],
            num_sample,
            time_step, energies_dos, qe_indices,
            self.output['corr_DOS'],
            self.output['corr_DC'],
            self.rank,
            self.config.generic['wfn_check_steps'],
            self.config.generic['wfn_check_thr']
        )
        corr_dos = self.average(corr_dos)
        corr_dc = self.average(corr_dc)
        self.__save_data(corr_dos, self.output["corr_DOS"])
        self.__save_data(corr_dc, self.output["corr_DC"])
        return corr_dos, corr_dc

    def calc_hall_mu(self):
        """
        Calculate mu_{mn} required for the evaluation of Hall conductivity
        using Kubo-Bastin formula.

        Reference:
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.114.116602

        :return: mu_mn: (n_kernel, n_kernel) complex128 array
            mu_{mn} in eqn. 4 in the reference paper, which will be utilized
            to evaluate Hall conductivity using Kubo-Bastin formula
            n_kernel = self.config.dckb['n_kernel']
            Unit is nm^2/h_bar^2 * (eV)^2.
        """
        indptr, indices, hop, dx, dy = self.sample.build_ham_dxy()
        num_sample = self.__dist_sample()
        mu_mn = f2py.tbpm_kbdc(
            self.config.generic['seed'],
            indptr, indices, hop,
            self.sample.rescale,
            dx, dy,
            num_sample,
            self.config.dckb['n_kernel'],
            self.config.dckb['direction'],
            self.rank)
        mu_mn = self.average(mu_mn)
        self.__save_data(mu_mn, self.output['hall_mu'])
        return mu_mn

    def calc_quasi_eigenstates(self, save_data=False):
        """
        Calculate quasi-eigenstates.

        :param save_data: boolean
            whether to save quasi-eigenstates
        :return: states: (n_energies, n_indptr-1) float64 array
            Quasi-eigenstates of the sample
            states[i, :] is a quasi-eigenstate at energy of
            config.quasi_eigenstates['energies'][i].
        """
        # Get parameters
        time_step = self.__get_time_step()
        bessel_series = self.__get_bessel_series(time_step)
        num_sample = self.__dist_sample()
        self.__echo_time_step_fs(time_step)

        # Call FORTRAN subroutine
        ham_csr = self.sample.build_ham_csr()
        states = f2py.tbpm_eigenstates(
            bessel_series,
            ham_csr.indptr, ham_csr.indices, ham_csr.data,
            self.config.generic['seed'],
            self.config.generic['nr_time_steps'],
            num_sample,
            time_step,
            self.config.quasi_eigenstates['energies'],
            self.rank,
            self.config.generic['wfn_check_steps'],
            self.config.generic['wfn_check_thr']
        )
        states = self.average(states)
        if save_data:
            self.__save_data(states, self.output['qe'])
        return states

    def calc_ldos_haydock(self):
        """
        Calculate local density of states (LDOS) using Haydock recursion method.

        CAUTION: this method works for only one site. Although it can be adopted
        to deal with multiple sites, the time usage will be unaffordable.
        Use TBPM instead if you want to calculate LDOS for multiple sites.

        Ref: https://journals.jps.jp/doi/10.1143/JPSJ.80.054710

        :return: energies: (2*nr_time_steps+1,) float64 array
            energies in eV
        :return: ldos: (2*nr_time_steps+1,) float64 array
            LDOS value to corresponding energies in 1/eV
        :raises RuntimeError: if more than 1 mpi process is used
        """
        if self.size != 1:
            raise RuntimeError("Using more than 1 mpi process is not allowed"
                               " for ldos_haydock")

        ham_csr = self.sample.build_ham_csr()
        energies, ldos = f2py.ldos_haydock(
            self.config.LDOS['site_indices'],
            self.config.LDOS['delta'],
            self.sample.energy_range,
            ham_csr.indptr, ham_csr.indices, ham_csr.data,
            self.sample.rescale,
            self.config.generic['seed'],
            self.config.LDOS['recursion_depth'],
            self.config.generic['nr_time_steps'],
            self.config.generic['nr_random_samples'],
            self.output['ldos_haydock'])
        self.__save_data(ldos, self.output["ldos_haydock"])
        return energies, ldos

    def calc_psi_t(self, psi_0: np.ndarray, time_log: np.ndarray, dt_scale=1.0,
                   save_data=False):
        """
        Calculate propagation of wave function from given initial state.

        NOTES: a demo of nr_time_steps and time_log
        nr_time_steps: 1, 2, 3, 4, 5, 6, 7, 8
             time_log: 0, 1, 2, 3, 4, 5, 6, 7, 8

        :param psi_0: (num_orb_sample,) complex128 array
            expansion coefficients of initial wave function
        :param time_log: (num_time,) int64 array
            steps on which time the time-dependent wave function will be logged
            For example, t=0 stands for the initial wave function, while t=1
            indicates the wave function AFTER the 1st propagation.
        :param dt_scale: float
            scale factor for the time step with respect to the default value
        :param save_data: boolean
            whether to save time-dependent wave function
        :return: psi_t: (num_time, num_orb_sample) complex128 array
            time-dependent wave function according to time_log
        :raises RuntimeError: if more than 1 mpi process is used
        :raises ValueError: if any time in time_log not in [0, nr_time_steps].
        """
        if self.size != 1:
            raise RuntimeError("Using more than 1 mpi process is not allowed"
                               " for wave function propagation")

        # Check and convert parameters
        psi_0 = np.array(psi_0, dtype=np.complex128)
        psi_0 /= np.linalg.norm(psi_0)
        time_log = np.array(list(set(time_log)), dtype=np.int64)
        time_log.sort()
        for it in time_log:
            if it not in range(self.config.generic['nr_time_steps']+1):
                raise ValueError(f"time {it} out of range")

        # Get quantities for propagation
        time_step = self.__get_time_step() * dt_scale
        bessel_series = self.__get_bessel_series(time_step)
        ham_csr = self.sample.build_ham_csr()
        self.__echo_time_step_fs(time_step)

        # Propagate the wave function
        psi_t = f2py.tbpm_psi_t(
            bessel_series,
            ham_csr.indptr, ham_csr.indices, ham_csr.data,
            psi_0, time_log,
            self.config.generic['nr_time_steps'],
            self.config.generic['wfn_check_steps'],
            self.config.generic['wfn_check_thr']
        )
        psi_t = psi_t.T
        if save_data:
            self.__save_data(psi_t, self.output['psi_t'])
        return psi_t
