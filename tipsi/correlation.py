"""correlation.py is the interface between python and fortran.
Most importantly, it returns correlation functions for given
Sample and Config objects.

Functions
----------
    Bessel
        Get Bessel functions. (Move this function to fortran?)
    corr_DOS
        Get DOS correlation function.
    corr_AC
        Get AC conductivity correlation function.
    corr_dyn_pol
        Get dynamical polarization correlation function.
    corr_DC
        Get DC conductivity correlation function.
    quasi_eigenstates
        Get quasi-eigenstates.
"""

################
# dependencies
################

# numerics & math
import numpy as np
import scipy.special as spec

# fortran tbpm
from .fortran import f2py


def Bessel(t_step, H_rescale, Bessel_precision, Bessel_max):
    """Get Bessel functions of the first kind.

    Parameters
    ----------
    t_step : float
        time step
    H_rescale : float
        Hamiltonian rescale parameter; the Bessel function
        argument is given by t_step * H_rescale
    Bessel_precision : float
        get Bessel functions above this cut-off
    Bessel_max : int
        maximum order

    Returns
    ----------
    Bes : list of floats
        list of Bessel functions; returns False if Bessel_max is
        too low
    """

    Bes = []
    for i in range(Bessel_max):
        besval = spec.jv(i, t_step * H_rescale)
        if (np.abs(besval) > Bessel_precision):
            Bes.append(besval)
        else:
            besval_up = spec.jv(i + 1, t_step * H_rescale)
            if (np.abs(besval_up) > Bessel_precision):
                Bes.append(besval)
            else:
                return Bes
    return False


def corr_DOS(sample, config):
    """Get density of states correlation function

    Parameters
    ----------
    sample : Sample object
        sample information
    config : Config object
        tbpm parameters

    Returns
    ----------
    corr_DOS : list of complex floats
        DOS correlation function
    """

    # get Bessel functions
    t_step = 2 * np.pi / config.sample['energy_range']
    Bes = Bessel(t_step, sample.rescale,
                 config.generic['Bessel_precision'],
                 config.generic['Bessel_max'])

    # pass to FORTRAN
    corr_DOS = f2py.tbpm_dos(Bes,
                             sample.indptr, sample.indices, sample.hop,
                             config.generic['seed'],
                             config.generic['nr_time_steps'],
                             config.generic['nr_random_samples'],
                             config.output['corr_DOS'])

    return corr_DOS


def corr_LDOS(sample, config):
    """Get local density of states correlation function

    Parameters
    ----------
    sample : Sample object
        sample information
    config : Config object
        tbpm parameters

    Returns
    ----------
    corr_LDOS : list of complex floats
        LDOS correlation function
    """

    # get Bessel functions
    t_step = 2 * np.pi / config.sample['energy_range']
    Bes = Bessel(t_step, sample.rescale,
                 config.generic['Bessel_precision'],
                 config.generic['Bessel_max'])

    # get wf_weights:
    if not config.LDOS['wf_weights']:
        N = len(config.LDOS['site_indices'])
        wf_weights = [1 for i in range(N)]
    else:
        wf_weights = config.LDOS['wf_weights']

    # pass to FORTRAN
    corr_LDOS = f2py.tbpm_ldos(config.LDOS['site_indices'], wf_weights,
                               Bes, sample.indptr, sample.indices,
                               sample.hop,
                               config.generic['seed'],
                               config.generic['nr_time_steps'],
                               config.generic['nr_random_samples'],
                               config.output['corr_LDOS'])

    return corr_LDOS


def corr_AC(sample, config):
    """Get AC conductivity

    Parameters
    ----------
    sample : Sample object
        sample information
    config : Config object
        tbpm parameters

    Returns
    ----------
    corr_AC : (4, n) list of complex floats
        AC correlation function in 4 directions:
        xx, xy, yx, yy, respectively.
    """

    # get Bessel functions
    t_step = np.pi / config.sample['energy_range']
    Bes = Bessel(t_step, sample.rescale,
                 config.generic['Bessel_precision'],
                 config.generic['Bessel_max'])

    # get rescaled simulation parameters
    beta_re = config.generic['beta'] * sample.rescale
    mu_re = config.generic['mu'] / sample.rescale

    # pass to FORTRAN
    corr_AC = f2py.tbpm_accond(Bes, beta_re, mu_re,
                               sample.indptr, sample.indices,
                               sample.hop,
                               sample.rescale, sample.dx, sample.dy,
                               config.generic['seed'],
                               config.generic['nr_time_steps'],
                               config.generic['nr_random_samples'],
                               config.generic['nr_Fermi_fft_steps'],
                               config.generic['Fermi_cheb_precision'],
                               config.output['corr_AC'])

    return corr_AC


def corr_dyn_pol(sample, config):
    """Get dynamical polarization correlation function

    Parameters
    ----------
    sample : Sample object
        sample information
    config : Config object
        tbpm parameters

    Returns
    ----------
    corr_dyn_pol : (n_q_points, n_t_steps) list of complex floats
        Dynamical polarization correlation function.
    """

    # get Bessel functions
    t_step = np.pi / config.sample['energy_range']
    Bes = Bessel(t_step, sample.rescale,
                 config.generic['Bessel_precision'],
                 config.generic['Bessel_max'])

    # get rescaled simulation parameters
    beta_re = config.generic['beta'] * sample.rescale
    mu_re = config.generic['mu'] / sample.rescale

    # pass to FORTRAN
    corr_dyn_pol = f2py.tbpm_dyn_pol(Bes, beta_re, mu_re,
                                     sample.indptr, sample.indices,
                                     sample.hop, sample.rescale,
                                     sample.dx, sample.dy,
                                     sample.site_x, sample.site_y,
                                     sample.site_z,
                                     config.generic['seed'],
                                     config.generic['nr_time_steps'],
                                     config.generic['nr_random_samples'],
                                     config.generic['nr_Fermi_fft_steps'],
                                     config.generic['Fermi_cheb_precision'],
                                     config.dyn_pol['q_points'],
                                     config.output['corr_dyn_pol'])

    return corr_dyn_pol


def corr_DC(sample, config):
    """Get DC conductivity correlation function

    Parameters
    ----------
    sample : Sample object
        sample information
    config : Config object
        tbpm parameters

    Returns
    ----------
    corr_DOS : (n_t_steps) list of complex floats
        DOS correlation function.
    corr_DC : (2, n_energies, n_t_steps) list of complex floats
        DC conductivity correlation function.
    """

    # Get parameters
    tnr = config.generic['nr_time_steps']
    en_range = config.sample['energy_range']
    t_step = 2 * np.pi / en_range
    Bes = Bessel(t_step, sample.rescale,
                 config.generic['Bessel_precision'],
                 config.generic['Bessel_max'])
    energies_DOS = np.array([0.5 * i * en_range / tnr - en_range / 2.
                             for i in range(tnr * 2)])
    lims = config.DC_conductivity['energy_limits']
    QE_indices = np.where((energies_DOS >= lims[0]) &
                          (energies_DOS <= lims[1]))[0]
    beta_re = config.generic['beta'] * sample.rescale
    mu_re = config.generic['mu'] / sample.rescale

    # pass to FORTRAN
    corr_DOS, corr_DC = f2py.tbpm_dccond(Bes, beta_re, mu_re,
                                         sample.indptr, sample.indices,
                                         sample.hop,
                                         sample.rescale, sample.dx, sample.dy,
                                         config.generic['seed'],
                                         config.generic['nr_time_steps'],
                                         config.generic['nr_random_samples'],
                                         t_step, energies_DOS, QE_indices,
                                         config.output['corr_DOS'],
                                         config.output['corr_DC'])

    return corr_DOS, corr_DC


def mu_Hall(sample, config):
    """Get correlation for Hall conductivity

    Parameters
    ----------
    sample: fortran_sample object
        sample information
    config : tbpm_config object
        config parameters

    Returns
    ----------
    energies : list of floats
        energy list with rank (2*nr_time_steps+1)
    mu_mn    : ???

    """

    # if config.dckb['output_correlation']:
    #     output_int = 2
    #     dckb_corr_filename = config.output['dckb_corr']
    # else:
    #     output_int = 1
    #     dckb_corr_filename = ''

    print(" -- Calculating mu_mn for Hall conductivity")

    from .fortran import f2py as fortran_f2py

    # call fortran function
    mu_mn = fortran_f2py.tbpm_kbdc(
        config.generic['seed'], sample.indptr, sample.indices, sample.hop,
        sample.rescale, sample.dx, sample.dy,
        config.generic['nr_random_samples'], config.dckb['n_kernel'],
        config.dckb['direction'])

    return mu_mn


def quasi_eigenstates(sample, config):
    """Get quasi-eigenstates

    Parameters
    ----------
    sample : Sample object
        sample information
    config : Config object
        tbpm parameters

    Returns
    ----------
    states : list of list of complex floats
        Quasi-eigenstates of the sample; states[i,:] is a quasi-eigenstate
        at energy config.quasi_eigenstates['energies'][i].
    """

    # get Bessel functions
    t_step = 2 * np.pi / config.sample['energy_range']
    Bes = Bessel(t_step, sample.rescale,
                 config.generic['Bessel_precision'],
                 config.generic['Bessel_max'])

    # pass to FORTRAN
    states = f2py.tbpm_eigenstates(Bes,
                                   sample.indptr, sample.indices, sample.hop,
                                   config.generic['seed'],
                                   config.generic['nr_time_steps'],
                                   config.generic['nr_random_samples'], t_step,
                                   config.quasi_eigenstates['energies'])

    return states
