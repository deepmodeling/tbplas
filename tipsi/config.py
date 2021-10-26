"""config.py contains the class Config, which keeps track of
parameters for the TBPM calculation.

Classes
----------
    Config
        Contains TBPM parameters.
"""

################
# dependencies
################
import time
import os

# numerics & math
import numpy as np
import pickle

################
# Config class
################


class Config:
    """Class for TBPM parameters.

    Attributes
    ----------
    sample['area_unit_cell'] : float
        Area of the unit cell.
    sample['energy_range'] : float
        Energy range in eV, centered at 0.
    sample['extended'] : integer
        Number of times the unit cell has been extended.
    sample['nr_orbitals'] : integer
        Degrees of freedom per unit cell.
    sample['volume_unit_cell'] : float
        Volume of the unit cell.
    scmple['H_rescale'] : float
        rescale value for Hamiltonian
    generic['Bessel_max'] : int
        Maximum number of Bessel functions. Default value: 100
    generic['Bessel_precision'] : float
        Bessel function precision cut-off. Default value: 1.0e-13
    generic['beta'] : float
        Value for 1/kT.
        Default value: 11604.505/300 (room temperature, using eV)
    generic['correct_spin'] : bool
        If True, results are corrected for spin. Default value: False.
    generic['Fermi_cheb_precision'] : float
        Precision cut-off of Fermi-Dirac distribution.
        Default value: 1.0e-10
    generic['mu'] : float
        Chemical potential. Default value: 0.
    generic['nr_Fermi_fft_steps'] : int
        Maximum number of Fermi-Dirac distribution FFT steps,
        must be power of two. Default value: 2**15
    generic['nr_ran_samples'] : int
        Number of random initial wave functions. Default value: 1
    generic['nr_time_steps'] : int
        Number of time steps. Default value: 1024
    generic['seed'] : int
        Seed for random wavefunction generation. Default value: 1337.
    generic['rank'] : int
        Rank of mpi process. Default value: 0.
    LDOS['site_indices'] : int
        Site indices for LDOS calculation.
    LDOS['wf_weights'] : int
        Wave function weights for LDOS calculation.
        Default value: equal weights for all sites.
    LDOS['delta'] : float
        Parameter of infinitesimal. Default value: 0.01.
    LDOS['recursion_depth'] : int
        Recursion depth of Haydock method. Default value: 2000
    dyn_pol['background_dielectric_constant'] : float
        Background dielectric constant. Default value: 23.6.
    dyn_pol['coulomb_constant'] : float
        Coulomb constant. Default value: 1.0
    dyn_pol['q_points'] : (n_q_points, 3) list of floats
        List of q-points. Default value: [[0.1, 0., 0.]].
    DC_conductivity['energy_limits'] : 2-tuple of floats
        Minimum and maximum of energy window for DC conductivity.
        Default value: [-0.5, 0.5].
    dckb['energies'] : list of floats
        List of chemical potentials to calculate Hall conductivity.
        It must be in [-1, 1], unit is H_rescale.
    dckb['n_kernel'] : int
        Number of kernels in Kernel Polynomial Method(KPM). Default value: 2048
    dckb['direction'] : int
        1 gives XX, 2 gives XY conductivity. Default value: 1
    dckb['ne_integral'] : int
        Number of integral steps. Default value: 2048
    quasi_eigenstates['energies'] : list of floats
        List of energies of quasi-eigenstates. Default value: [-0.1, 0., 0.1].
    output['prefix'] : string
        Prefix prepended to the output files. Default value: timestamp.
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
    output['directory'] : string
        Output directory. Default value: "sim_data".
    """

    # initialize
    def __init__(self, sample=False, read_from_file=False,
                 directory=None, prefix=None, mpi_env=None):
        """Initialize.

        Parameters
        ----------
        sample : Sample object
            Sample object of which to take sample parameters.
        read_from_file : bool
            set to True if you are reading a config object from file
        directory: string
            Directory for writing/reading files.
        prefix: string
            Prefix prepended to output files under directory.
        mpi_env: MPIEnv object
           MPI environment.
        """

        # declare dicts
        self.sample = {}
        self.generic = {}
        self.LDOS = {}
        self.dyn_pol = {}
        self.DC_conductivity = {}
        self.quasi_eigenstates = {}
        self.output = {}
        self.dckb = {}

        # sample parameters
        if sample:
            self.sample['nr_orbitals'] = sample.nr_orbitals
            self.sample['energy_range'] = sample.energy_range
            self.sample['area_unit_cell'] = sample.area_unit_cell
            self.sample['volume_unit_cell'] = sample.volume_unit_cell
            self.sample['extended'] = sample.extended
            self.sample['H_rescale'] = sample.rescale

        # generic standard values
        self.generic['Bessel_max'] = 100
        self.generic['Bessel_precision'] = 1.0e-13
        self.generic['correct_spin'] = False
        self.generic['nr_time_steps'] = 1024
        self.generic['nr_random_samples'] = 1
        self.generic['beta'] = 11604.505 / 300
        self.generic['mu'] = 0.
        self.generic['nr_Fermi_fft_steps'] = 2**15
        self.generic['Fermi_cheb_precision'] = 1.0e-10
        self.generic['seed'] = 1337
        if mpi_env is None:
            self.generic['rank'] = 0
        else:
            self.generic['rank'] = mpi_env.rank

        # LDOS
        self.LDOS['site_indices'] = 0
        self.LDOS['wf_weights'] = False
        self.LDOS['delta'] = 0.01
        self.LDOS['recursion_depth'] = 2000

        # DC conductivity
        self.DC_conductivity['energy_limits'] = (-0.5, 0.5)

        # quasi-eigenstates
        self.quasi_eigenstates['energies'] = [-0.1, 0., 0.1]

        # dynamical polarization
        self.dyn_pol['q_points'] = [[1., 0., 0.]]
        self.dyn_pol['coulomb_constant'] = 1.0
        self.dyn_pol['background_dielectric_constant'] = 2 * np.pi * 3.7557757

        # dckb, Hall conductivity
        self.dckb['energies'] = [i * 0.01 - 0.2 for i in range(0, 41)]
        self.dckb['n_kernel'] = 2048
        self.dckb['direction'] = 1  # 1 gives XX, 2 gives XY conductivity
        self.dckb['ne_integral'] = 2048

        # output settings
        # TODO: what if read_from_file?
        if not read_from_file:
            self.set_output(directory, prefix)

    def set_output(self, directory=None, prefix=None):
        """Function to set data output options.

        This function will set self.output['directory'] and correlation
        file names for automised data output.

        Parameters
        ----------
        directory : string, optional
            output directory, set to False if you don't want to specify
            an output directory
        prefix : string, optional
            prefix for filenames, set to False for standard (timestamp) prefix
        """
        # Update names of directory and files
        if directory is not None:
            self.output['directory'] = directory
        else:
            self.output['directory'] = 'sim_data'
        if prefix is not None:
            self.output['prefix'] = prefix
        else:
            self.output['prefix'] = str(int(time.time()))
        full_prefix = "%s/%s" % (self.output['directory'], self.output['prefix'])
        for key in ('corr_DOS', 'corr_LDOS', 'corr_AC', 'corr_dyn_pol', 'corr_DC'):
            self.output[key] = "%s%s.dat.%s" % (full_prefix, key, self.generic['rank'])
        if self.generic['rank'] == 0:
            print("Output details:")
            print("%11s: %s" % ("Directory", self.output['directory']))
            print("%11s: %s" % ("Prefix", self.output['prefix']))
            print(flush=True)

        # Create directory if not exits
        if self.generic['rank'] == 0:
            if not os.path.exists(self.output['directory']):
                os.mkdir(self.output['directory'])

    def save(self, filename="config.pkl", directory=None, prefix=None):
        """Function to save config parameters to a .pkl file.

        Parameters
        ----------
        filename : string, optional
            file name
        directory : string, optional
            output directory, set to False if you don't want to specify
            an output directory
        prefix : string, optional
            prefix for filenames, set to False for standard (timestamp) prefix
        """
        if self.generic['rank'] == 0:
            if directory is None:
                directory = self.output['directory']
            if prefix is None:
                prefix = self.output['prefix']
            if not os.path.exists(directory):
                os.mkdir(directory)
            pickle_name = "%s/%s%s" % (directory, prefix, filename)
            with open(pickle_name, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def dckb_prefactor(self):
        return 16 * self.sample['nr_orbitals'] / self.sample['area_unit_cell']
