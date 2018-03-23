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
    corr_KB_DC
        Get Kubo-Bastin DC conductivity correlation function.
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
from .fortran import tbpm_f2py as fortran_tbpm

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
    Bes = Bessel(t_step, sample.rescale, \
                 config.generic['Bessel_precision'], \
                 config.generic['Bessel_max'])

    # pass to FORTRAN
    corr_DOS = fortran_tbpm.tbpm_dos(Bes, \
        sample.indptr, sample.indices, sample.hop, \
        config.generic['seed'], config.generic['nr_time_steps'], \
        config.generic['nr_random_samples'], config.output['corr_DOS'])
    
    return corr_DOS
    
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
    Bes = Bessel(t_step, sample.rescale, \
                 config.generic['Bessel_precision'], \
                 config.generic['Bessel_max'])
                 
    # get rescaled simulation parameters
    beta_re = config.generic['beta'] * sample.rescale
    mu_re = config.generic['mu'] / sample.rescale
    
    # pass to FORTRAN
    corr_AC = fortran_tbpm.tbpm_accond(Bes, beta_re, mu_re, \
        sample.indptr, sample.indices, sample.hop, \
        sample.rescale, sample.dx, sample.dy, \
        config.generic['seed'], config.generic['nr_time_steps'], \
        config.generic['nr_random_samples'], \
        config.generic['nr_Fermi_fft_steps'], \
        config.generic['Fermi_cheb_precision'], config.output['corr_AC'])

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
    Bes = Bessel(t_step, sample.rescale, \
                 config.generic['Bessel_precision'], \
                 config.generic['Bessel_max'])
                 
    # get rescaled simulation parameters
    beta_re = config.generic['beta'] * sample.rescale
    mu_re = config.generic['mu'] / sample.rescale
    
    # pass to FORTRAN
    corr_dyn_pol = fortran_tbpm.tbpm_dyn_pol(Bes, beta_re, mu_re, \
        sample.indptr, sample.indices, sample.hop, \
        sample.rescale, sample.dx, sample.dy, \
        sample.site_x, sample.site_y, sample.site_z, \
        config.generic['seed'], config.generic['nr_time_steps'], \
        config.generic['nr_random_samples'], \
        config.generic['nr_Fermi_fft_steps'], \
        config.generic['Fermi_cheb_precision'], \
        config.dyn_pol['q_points'], config.output['corr_dyn_pol'])

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
    corr_DC : (2, n_energies, n_t_steps) list of complex floats
        DC conductivity correlation function.
    """ 
    
    return
    
def corr_KB_DC(sample, config):
    """Get Kubo-Bastin DC conductivity correlation function
 
    Parameters
    ----------
    sample : Sample object
        sample information
    config : Config object
        tbpm parameters
        
    Returns
    ----------
    corr_KB_DC : (n_kernel, n_kernel) list of complex floats
        Kubo-Bastin DC conductivity correlation function.
    """ 
    
    return
    
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
    Bes = Bessel(t_step, sample.rescale, \
                 config.generic['Bessel_precision'], \
                 config.generic['Bessel_max'])

    # pass to FORTRAN
    states = fortran_tbpm.tbpm_eigenstates(Bes, \
        sample.indptr, sample.indices, sample.hop, \
        config.generic['seed'], config.generic['nr_time_steps'], \
        t_step, config.quasi_eigenstates['energies'])
    
    return states
