"""Functions and classes for managing the parameters of TBPM calculation."""

import pickle

from ..base import KB


__all__ = ["Config", "read_config"]


class Config:
    """
    Class for representing TBPM parameters.

    Attributes
    ----------
    generic['Bessel_max'] : int
        Maximum number of Bessel functions. Default value: 100
    generic['Bessel_precision'] : float
        Bessel function precision cut-off. Default value: 1.0e-14
    generic['beta'] : float
        Value for 1/kBT in 1/eV. Default value: 1 / (kB * 300).
    generic['mu'] : float
        Chemical potential in eV. Default value: 0.
    generic['correct_spin'] : bool
        If True, results are corrected for spin-degeneracy.
        Default value: False.
    generic['nr_ran_samples'] : int
        Number of random initial wave functions. Default value: 1
    generic['nr_time_steps'] : int
        Number of time steps. Default value: 1024
    generic['seed'] : int
        Seed for random wave function generation. Default value: 1337.
    generic['wfn_check_steps']: int
        Check the wave function for divergence after each number of steps.
        Default value: 128.
    generic['wfn_check_thr']: float
        Threshold for checking divergence of wave function. If the difference
        of norm is larger than this value, errors will be raised.
        Default value: 1.0e-9
    generic['Fermi_cheb_precision'] : float
        Precision cut-off of Fermi-Dirac distribution.
        Default value: 1.0e-12
    generic['nr_Fermi_fft_steps'] : int
        Maximum number of Fermi-Dirac distribution FFT steps,
        must be power of two. Default value: 2**15
    LDOS['site_indices'] : List[int]
        Site indices for LDOS calculation. Default value: [0].
        There is no limit for the number of sites for TBPM.
        However, the Haydock recursion method accepts only one site.
    LDOS['wf_weights'] : List[float]
        Wave function weights for LDOS calculation. Default value: [1.0].
        It seems that this parameter is no longer in use.
    LDOS['delta'] : float
        Parameter of infinitesimal in eV. Default value: 0.01.
    LDOS['recursion_depth'] : int
        Recursion depth of Haydock method. Default value: 2000
    dyn_pol['background_dielectric_constant'] : float
        Relative background dielectric constant. Default value: 1.0.
    dyn_pol['coulomb_constant'] : float
        Scaling factor for Coulomb potential. Default value: 1.0
    dyn_pol['q_points'] : (n_q_points, 3) list of floats
        Cartesian coordinates of q-points in 1/nm.
        Default value: [[1., 0., 0.]].
    DC_conductivity['energy_limits'] : 2-tuple of floats
        Minimum and maximum of energy window for DC conductivity in eV.
        Default value: [-0.5, 0.5].
    dckb['energies'] : list of floats
        List of chemical potentials to calculate Hall conductivity in eV.
        Default value: [-0.2, 0.2] with energy step of 0.01 eV
    dckb['n_kernel'] : int
        Number of kernels in Kernel Polynomial Method(KPM). Default value: 2048
    dckb['direction'] : int
        1 gives XX, 2 gives XY conductivity. Default value: 1
    dckb['ne_integral'] : int
        Number of integral steps. Default value: 2048
    quasi_eigenstates['energies'] : list of floats
        List of energies of quasi-eigenstates in eV.
        Default value: [-0.1, 0., 0.1].
    _legal_params: dict of sets
        names of legal parameters, reserved for checking purposes.
        DO NOT CHANGE IT!
    """
    def __init__(self):
        # generic standard values
        self.generic = {'Bessel_max': 250,
                        'Bessel_precision': 1.0e-14,
                        'beta': 1.0 / (KB * 300),
                        'mu': 0.,
                        'correct_spin': False,
                        'nr_time_steps': 1024,
                        'nr_random_samples': 1,
                        'seed': 1337,
                        'wfn_check_steps': 128,
                        'wfn_check_thr': 1.0e-9,
                        'nr_Fermi_fft_steps': 2**15,
                        'Fermi_cheb_precision': 1.0e-12}

        # LDOS
        self.LDOS = {'site_indices': [0],
                     'wf_weights': [1.0],
                     'delta': 0.01,
                     'recursion_depth': 2000}

        # DC conductivity
        self.DC_conductivity = {'energy_limits': (-0.5, 0.5)}

        # quasi-eigenstates
        self.quasi_eigenstates = {'energies': [-0.1, 0., 0.1]}

        # dynamical polarization
        self.dyn_pol = {'q_points': [[1., 0., 0.]],
                        'coulomb_constant': 1.0,
                        'background_dielectric_constant': 1.0}

        # dckb, Hall conductivity
        self.dckb = {'energies': [i * 0.01 - 0.2 for i in range(0, 41)],
                     'n_kernel': 2048,
                     'direction': 1,
                     'ne_integral': 2048}

        # Set legal parameter names
        self._legal_params = {
            'generic': set(self.generic.keys()),
            'LDOS': set(self.LDOS.keys()),
            'dyn_pol': set(self.dyn_pol.keys()),
            'dckb': set(self.dckb.keys())
        }

    def set_temperature(self, temperature=300):
        """
        Set temperature.

        :param temperature: float
            temperature in Kelvin
        """
        self.generic['beta'] = 1.0 / (KB * temperature)

    def check_params(self):
        """
        Check the sanity of parameters.

        :return: None
        :raises ValueError: if illegal parameters are detected.
        """
        def _check(attr, attr_name):
            set_check = set(attr.keys())
            set_ref = self._legal_params[attr_name]
            set_diff = set_check.difference(set_ref)
            for key in set_diff:
                raise ValueError(f"Undefined parameter {key} in"
                                 f" config.{attr_name}")

        _check(self.generic, 'generic')
        _check(self.LDOS, 'LDOS')
        _check(self.dyn_pol, 'dyn_pol')
        _check(self.dckb, 'dckb')


def read_config(filename):
    """
    Read configuration from a .pkl file.

    :param filename: string
            file name of the .pkl file
    :return: config: instance of 'Config' class
        config object read from file
    """
    with open(filename, 'rb') as f:
        config_dict = pickle.load(f)
    config = Config()
    config.generic = config_dict.generic
    config.LDOS = config_dict.LDOS
    config.dyn_pol = config_dict.dyn_pol
    config.DC_conductivity = config_dict.DC_conductivity
    config.dckb = config_dict.dckb
    config.quasi_eigenstates = config_dict.quasi_eigenstates
    return config
