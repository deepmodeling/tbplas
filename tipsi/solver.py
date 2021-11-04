"""
Functions and classes for TBPM calculation.

Functions
---------
    None

Classes
-------
    BaseSolver: developer class
        base class for Solver and Analyzer
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
from .config import Config
from .fortran import f2py


class BaseSolver:
    """
    Base class for Solver and Analyzer.

    Attributes
    ----------
    sample: instance of 'Sample' class
        sample for which TBPM calculations will be performed
    config: instance of 'Config' class
        parameters controlling TBPM calculation
    mpi_env: instance of 'MPIEnv' class
        mpi parallel environment
        None if mpi is not enabled.
    """
    def __init__(self, sample: Sample, config: Config, enable_mpi=False):
        """
        :param sample: instance of 'Sample' class
            sample for which TBPM calculations will be performed
        :param config: instance of 'Config' class
            parameters controlling TBPM calculations
        :param enable_mpi: boolean
            whether to enable parallelism using MPI
        """
        self.sample = sample
        self.config = config

        # Setup parallel environment
        if enable_mpi:
            from .parallel import MPIEnv
            self.mpi_env = MPIEnv()
        else:
            self.mpi_env = None

        # Print simulation details
        self.print("\nParallelization details:")
        self.print("%17s:%4d" % ("MPI processes", self.size))
        for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            if env_name in os.environ.keys():
                self.print("%17s:%4s" % (env_name, os.environ[env_name]))
            else:
                self.print("%17s: n/a" % env_name)
        self.print()

    @property
    def rank(self):
        """
        Common interface to obtain the rank of mpi process.

        :return: rank: integer
            self.mpi_env.rank if mpi is enabled, otherwise 0.
        """
        if self.mpi_env is not None:
            rank = self.mpi_env.rank
        else:
            rank = 0
        return rank

    @property
    def size(self):
        """
        Common interface to obtain total number of mpi processes.

        :return: size: integer
            self.mpi_env.size if mpi is enabled, otherwise 1.
        """
        if self.mpi_env is not None:
            size = self.mpi_env.size
        else:
            size = 1
        return size

    @property
    def is_master(self):
        """
        Common interface to check if this is the master process.

        :return: is_master: boolean
            True if self.rank == 0, otherwise False.
        """
        return self.rank == 0

    def barrier(self):
        """
        Common interface to call MPI_Barrier.

        :return: None
        """
        if self.mpi_env is not None:
            self.mpi_env.barrier()
        else:
            pass

    def average(self, data_local):
        """
        Common interface to average data among mpi processes.

        :param data_local: numpy array
            local data on each mpi process
        :return: data: numpy array with same shape and data type as data_local
            averaged data from data_local
            data_local itself is returned if mpi is not enabled.
        """
        if self.mpi_env is not None:
            return self.mpi_env.average(data_local)
        else:
            return data_local

    def print(self, text=""):
        """
        Print text on master process.

        NOTE: flush=True is essential for some MPI implementations,
        e.g. MPICH3.
        """
        if self.rank == 0:
            print(text, flush=True)


class Solver(BaseSolver):
    """
    Wrapper class over FORTRAN TBPM subroutines.

    Attributes
    ----------
    output['directory'] : string
        Output directory.
        Default value: "sim_data".
    output['prefix'] : string
        Prefix prepended to the output files.
        Default value: timestamp.
    output['corr_AC'] : string
        AC conductivity correlation output file.
        Default value: "sim_data/" + timestamp + "corr_AC.dat".
    output['corr_DC'] : string
        DC conductivity correlation output file.
        Default value: "sim_data/" + timestamp + "corr_DC.dat".
    output['corr_DOS'] : string
        DOS correlation output file.
        Default value: "sim_data/" + timestamp + "corr_DOS.dat".
    output['corr_LDOS'] : string
        LDOS correlation output file.
        Default value: "sim_data/" + timestamp + "corr_LDOS.dat".
    output['corr_dyn_pol'] : string
        AC conductivity correlation output file.
        Default value: "sim_data/" + timestamp + "corr_dyn_pol.dat".
    """
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
        self.output = dict()
        self.set_output()

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
        if self.rank == 0:
            if not os.path.exists(self.output["directory"]):
                os.mkdir(self.output["directory"])
        self.barrier()

        # Determine file names
        full_prefix = "%s/%s" % (self.output["directory"],
                                 self.output["prefix"])
        for key in ("corr_DOS", "corr_LDOS", "corr_AC", "corr_dyn_pol",
                    "corr_DC"):
            self.output[key] = "%s%s.dat.%s" % (full_prefix, key, self.rank)

        # Print output details
        self.print("Output details:")
        self.print("%11s: %s" % ("Directory", self.output["directory"]))
        self.print("%11s: %s" % ("Prefix", self.output["prefix"]))
        self.print()

    def save_config(self, filename="config.pkl"):
        """
        Save self.config to a .pkl file.

        :param filename: string
            file name of the .pkl file
        :return: None
        """
        if self.rank == 0:
            pickle_name = "%s/%s%s" % (self.output["directory"],
                                       self.output["prefix"], filename)
            with open(pickle_name, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def __dist_sample(self):
        """
        Common interface to distribute samples among MPI processes.

        :return: num_sample: integer
            number of random samples assigned to this process
            self.config.generic["nr_random_samples"] is updated.
        """
        num_sample_opt = self.config.generic["nr_random_samples"]
        if self.mpi_env is not None:
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
        Get the time step for TBPM calculation.

        :return: time_step: float
            time step
        """
        return 2 * math.pi / self.sample.energy_range

    def __get_bessel_series(self):
        """
        Get the values of Bessel functions up to given order.

        :return: bes: list of floats
            values of Bessel functions
        :raises ValueError: if self.config.generic["Bessel_max"] is too low
        """
        bessel_max = self.config.generic["Bessel_max"]
        bessel_precision = self.config.generic["Bessel_precision"]
        time_step = self.__get_time_step()
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

    def calc_corr_dos(self):
        """
        Calculate correlation function of density of states (DOS).

        :return: corr_dos: (nr_time_steps+1,) complex128 array
            DOS correlation function
        """
        # Get parameters
        bessel_series = self.__get_bessel_series()
        ham_csr = self.sample.build_ham_csr()
        num_sample = self.__dist_sample()

        # Call FORTRAN subroutine
        corr_dos = f2py.tbpm_dos(
            bessel_series,
            ham_csr.indptr, ham_csr.indices, ham_csr.data,
            self.config.generic['seed'],
            self.config.generic['nr_time_steps'],
            num_sample,
            self.output['corr_DOS'],
            self.rank)
        corr_dos = self.average(corr_dos)
        return corr_dos

    def calc_corr_ldos(self):
        """
        Calculate correlation function of local density of states (DOS).

        :return: corr_ldos: (nr_time_steps+1,) complex128 array
            LDOS correlation function
        """
        # Get parameters
        bessel_series = self.__get_bessel_series()
        ham_csr = self.sample.build_ham_csr()
        num_sample = self.__dist_sample()

        # Call FORTRAN subroutine
        corr_ldos = f2py.tbpm_ldos(
            self.config.LDOS['site_indices'],
            bessel_series,
            ham_csr.indptr, ham_csr.indices, ham_csr.data,
            self.config.generic['seed'],
            self.config.generic['nr_time_steps'],
            num_sample,
            self.output['corr_LDOS'],
            self.rank)
        corr_ldos = self.average(corr_ldos)
        return corr_ldos

    def calc_corr_ac_cond(self):
        """
        Calculate correlation function of AC conductivity.

        :return: corr_ac: (4, nr_time_steps) complex128 array
            AC correlation function in 4 directions:
            xx, xy, yx, yy, respectively
        """
        # Get parameters
        bessel_series = self.__get_bessel_series()
        beta_re, mu_re = self.__get_beta_mu_re()
        indptr, indices, hop, dx, dy = self.sample.build_ham_dxy()
        num_sample = self.__dist_sample()

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
            self.rank)
        corr_ac = self.average(corr_ac)
        return corr_ac

    def calc_corr_dyn_pol(self):
        """
        Calculate correlation function of dynamical polarization.

        :return: corr_dyn_pol: (n_q_points, nr_time_steps) float64 array
            Dynamical polarization correlation function.
        """
        # Get parameters
        bessel_series = self.__get_bessel_series()
        beta_re, mu_re = self.__get_beta_mu_re()
        indptr, indices, hop, dx, dy = self.sample.build_ham_dxy()
        site_x = self.sample.orb_pos[:, 0]
        site_y = self.sample.orb_pos[:, 1]
        site_z = self.sample.orb_pos[:, 2]
        num_sample = self.__dist_sample()

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
            self.rank)
        corr_dyn_pol = self.average(corr_dyn_pol)
        return corr_dyn_pol

    def calc_corr_dc_cond(self):
        """
        Calculate correlation function of DC conductivity.

        :return: corr_dos: (nr_time_steps,) complex128 array
            DOS correlation function
        :return: corr_dc: (2, n_energies, nr_time_steps) complex128 array
            DC conductivity correlation function
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
        bessel_series = self.__get_bessel_series()
        indptr, indices, hop, dx, dy = self.sample.build_ham_dxy()
        num_sample = self.__dist_sample()

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
            self.rank)
        corr_dos = self.average(corr_dos)
        corr_dc = self.average(corr_dc)
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
        return mu_mn

    def calc_quasi_eigenstates(self):
        """
        Calculate quasi-eigenstates.

        :return: states: (n_energies, n_indptr-1) float64 array
            Quasi-eigenstates of the sample
            states[i, :] is a quasi-eigenstate at energy of
            config.quasi_eigenstates['energies'][i].
        """
        # Get parameters
        time_step = self.__get_time_step()
        bessel_series = self.__get_bessel_series()
        num_sample = self.__dist_sample()

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
            self.rank)
        states = self.average(states)
        return states

    def calc_ldos_haydock(self):
        """
        Calculate local density of states (LDOS) using Haydock recursion method.

        :return: energies: (2*nr_time_steps+1,) float64 array
            energy list
        :return: ldos: (2*nr_time_steps+1,) float64 array
            LDOS value to corresponding energies
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
            self.output['corr_LDOS'])
        return energies, ldos
