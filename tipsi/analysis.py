"""analysis.py contains tools to analyze correlation functions.

Functions
----------
    window_Hanning
        Hanning window
    window_exp
        Exponential window
    window_exp_ten
        Window function given by exponential of 10
    analyze_corr_DOS
        Analyze DOS correlation function
    analyze_corr_LDOS
        Analyze LDOS correlation function
    analyze_corr_AC
        Analyze AC correlation function
    AC_imag
        Calculate the imaginary part of the AC conductivity
    analyze_corr_dyn_pol
        Analyze dynamical polarization correlation function
    get_dielectric_function
        Get dielectric function from dynamical polarization
    analyze_corr_DC
        Analyze DC correlation function
"""

################
# dependencies
################

# numerics & math
import numpy as np
import numpy.linalg as npla
from scipy.signal import hilbert

################
# window functions
################


# Hanning window
def window_Hanning(i, N):
    """Hanning window.

    Parameters
    ----------
    i : integer
        summation index
    N : integer
        total length of summation

    Returns
    ----------
    float
        Hanning window value
    """

    return 0.5 * (1 + np.cos(np.pi * i / N))


# Exponential window
def window_exp(i, N):
    """Exponential window.

    Parameters
    ----------
    i : integer
        summation index
    N : integer
        total length of summation

    Returns
    ----------
    float
        exponential window value
    """

    return np.exp(-2. * (i / N)**2)


# Exponential of 10 window
def window_exp_ten(i, N):
    """Window function given by exponential of 10.

    Parameters
    ----------
    i : integer
        summation index
    N : integer
        total length of summation

    Returns
    ----------
    float
        exponential window value
    """

    power = -2 * (1. * i / N)**2
    return 10.**power


################
# correlation function analysis
################


def analyze_corr_DOS(config, corr_DOS, window=window_Hanning):
    """Function for analyzing the DOS correlation function.

    Parameters
    ----------
    config : Config object
        contains TBPM configuration parameters
    corr_DOS : list of complex floats
        DOS correlation function
    window : function, optional
        window function for integral; default: window_Hanning

    Returns
    ----------
    energies : list of floats
        energy values
    DOS : list of floats
        DOS values corresponding to energies
    """

    # get useful things
    tnr = config.generic['nr_time_steps']
    en_range = config.sample['energy_range']
    energies = [0.5 * i * en_range / tnr - en_range / 2. \
                for i in range(tnr * 2)]
    en_step = 0.5 * en_range / tnr

    # Get negative time correlation
    corr_negtime = np.zeros(tnr * 2, dtype=complex)
    corr_negtime[tnr - 1] = 1.
    corr_negtime[2 * tnr - 1] = window(tnr - 1, tnr) \
                                * corr_DOS[tnr - 1]
    for i in range(tnr - 1):
        corr_negtime[tnr + i] = window(i, tnr) \
                                * corr_DOS[i]
        corr_negtime[tnr - i - 2] = window(i, tnr) \
                                    * np.conjugate(corr_DOS[i])

    # Fourier transform
    corr_fft = np.fft.ifft(corr_negtime)
    DOS = np.zeros(tnr * 2)
    for i in range(tnr):
        DOS[i + tnr] = np.abs(corr_fft[i])
    for i in range(tnr, 2 * tnr):
        DOS[i - tnr] = np.abs(corr_fft[i])

    # Normalise and correct for spin
    DOS = DOS / (np.sum(DOS) * en_step)
    if config.generic['correct_spin']:
        DOS = 2. * DOS

    return energies, DOS


def analyze_corr_LDOS(config, corr_LDOS, window=window_Hanning):
    """Function for analyzing the LDOS correlation function -
    exactly the same as DOS analysis function.

    Parameters
    ----------
    config : Config object
        contains TBPM configuration parameters
    corr_LDOS : list of complex floats
        LDOS correlation function
    window : function, optional
        window function for integral; default: window_Hanning

    Returns
    ----------
    energies : list of floats
        energy values
    LDOS : list of floats
        LDOS values corresponding to energies
    """
    return analyze_corr_DOS(config, corr_LDOS, window)


def analyze_corr_AC(config, corr_AC, window=window_exp):
    """Function for analyzing the AC conductivity correlation function.

    Parameters
    ----------
    config : Config object
        contains TBPM configuration parameters
    corr_AC : (4,n) list of complex floats
        AC conductivity correlation function
    window : function, optional
        window function for integral; default: window_exp

    Returns
    ----------
    omegas : list of floats
        omega values
    AC : (4,n) list of floats
        AC conductivity values corresponding to omegas, for
        4 directions (xx, xy, yx, yy, respectively)
    """

    # get useful things
    tnr = config.generic['nr_time_steps']
    en_range = config.sample['energy_range']
    t_step = np.pi / en_range
    beta = config.generic['beta']
    omegas = [i * en_range / tnr for i in range(tnr)]
    ac_prefactor = 4. * config.sample['nr_orbitals'] \
                        / config.sample['area_unit_cell'] \
                        / config.sample['extended']

    # get AC conductivity
    AC = np.zeros((4, tnr))
    for j in range(4):
        for i in range(tnr):
            omega = omegas[i]
            acv = 0.
            for k in range(tnr):
                acv += 2. * window(k + 1, tnr) \
                       * np.sin(omega * k * t_step) \
                       * corr_AC[j,k].imag
            if omega == 0.:
                acv = 0.
            else:
                acv = ac_prefactor * t_step * acv \
                      * (np.exp(-beta * omega) - 1) / omega
            AC[j, i] = acv

    # correct for spin
    if config.generic['correct_spin']:
        AC = 2. * AC

    return omegas, AC


def AC_imag(AC_real):
    """Get the imaginary part of the AC conductivity
    from the real part using the Kramers-Kronig relations
    (the Hilbert transform).

    Parameters
    ----------
    AC_real : array of floats
        Re(sigma)

    Returns
    ----------
    array of floats
        Im(sigma)
    """

    N = len(AC_real)
    sigma = np.zeros(2 * N)
    for i in range(N):
        sigma[N + i] = AC_real[i]
        sigma[N - i] = AC_real[i]
    return np.imag(hilbert(sigma))[N:2 * N]

def analyze_corr_dyn_pol(config, corr_dyn_pol, \
                         window = window_exp_ten):
    """Function for analyzing the dynamical polarization correlation function.

    Parameters
    ----------
    config : Config object
        contains TBPM configuration parameters
    corr_dyn_pol : (n_q_points, n_t_steps) list of floats
        dynamical polarization correlation function
    window : function, optional
        window function for integral; default: window_exp_ten

    Returns
    ----------
    q_points : list of floats
        q-point values
    omegas : list of floats
        omega values
    dyn_pol : (n_q_points, n_omegas) list of complex floats
        dynamical polarization values corresponding to q-points and omegas
    """

    # get useful things
    tnr = config.generic['nr_time_steps']
    en_range = config.sample['energy_range']
    t_step = np.pi / en_range
    beta = config.generic['beta']
    q_points = config.dyn_pol['q_points']
    n_q_points = len(q_points)
    omegas = [i * en_range / tnr for i in range(tnr)]
    n_omegas = tnr
    # do we need to divide the prefac by 1.5??
    dyn_pol_prefactor = -2. * config.sample['nr_orbitals'] \
                        / config.sample['area_unit_cell'] \
                        / config.sample['extended']

    # get dynamical polarization
    dyn_pol = np.zeros((n_q_points, n_omegas), dtype=complex)
    for i_q in range(n_q_points):
        for i in range(n_omegas):
            omega = omegas[i]
            dpv = 0.0j
            for k in range(tnr):
                tau = k * t_step
                dpv += window(k + 1, tnr) * corr_dyn_pol[i_q, k] \
                       * np.exp(1j * omega * tau)
            dyn_pol[i_q, i] = dyn_pol_prefactor * t_step * dpv

    # correct for spin
    if config.generic['correct_spin']:
        dyn_pol = 2. * dyn_pol

    return q_points, omegas, dyn_pol


def get_dielectric_function(config, dyn_pol):
    """Function for analyzing the DOS correlation function.

    Parameters
    ----------
    sample : Sample object
        contains sample information
    config : Config object
        contains TBPM configuration parameters
    dyn_pol : (n_q_points, n_t_steps) list of complex floats
        dynamical polarization values

    Returns
    ----------
    epsilon : (n_q_points, n_omegas) list of complex floats
        dielectric function
    """

    # get useful things
    tnr = config.generic['nr_time_steps']
    en_range = config.sample['energy_range']
    t_step = np.pi / en_range
    beta = config.generic['beta']
    q_points = config.dyn_pol['q_points']
    n_q_points = len(q_points)
    omegas = [i * en_range / tnr for i in range(tnr)]
    n_omegas = tnr
    epsilon_prefactor = config.dyn_pol['coulomb_constant'] \
                        / config.dyn_pol['background_dielectric_constant'] \
                        / config.sample['extended']

    # declare arrays
    epsilon = np.ones((n_q_points, n_omegas)) \
              + np.zeros((n_q_points, n_omegas)) * 0j
    V0 = epsilon_prefactor * np.ones(n_q_points)
    V = np.zeros(n_q_points)

    # calculate epsilon
    for i, q_point in enumerate(q_points):
        k = npla.norm(q_point)
        if k == 0.0:
            V[i] = 0.
        else:
            V[i] = V0[i] / k
            epsilon[i, :] -= V[i] * dyn_pol[i, :]

    return q_points, omegas, epsilon

def analyze_corr_DC(config, corr_DOS, corr_DC, \
                    window_DOS = window_Hanning, window_DC = window_exp):
    """Function for analyzing the DC correlation function.

    Parameters
    ----------
    config : Config object
        contains TBPM configuration parameters
    corr_DOS : (n_t_steps) list of floats
        DOS correlation function
    corr_DC : (2, n_energies, n_t_steps) list of floats
        DC conductivity correlation function
    window_DOS : function, optional
        window function for DOS integral; default: window_Hanning
    window_DC : function, optional
        window function for DC integral; default: window_exp

    Returns
    ----------
    energies : list of floats
        energy values
    DC : (2, n_energies) list of floats
        DC conductivity values
    """

    # get DOS
    energies_DOS, DOS = analyze_corr_DOS(config, corr_DOS, window_DOS)
    energies_DOS = np.array(energies_DOS)
    DOS = np.array(DOS)

    # get useful things
    tnr = config.generic['nr_time_steps']
    en_range = config.sample['energy_range']
    t_step = 2 * np.pi / en_range
    lims = config.DC_conductivity['energy_limits']
    QE_indices = np.where((energies_DOS >= lims[0]) &
                          (energies_DOS <= lims[1]))[0]
    n_energies = len(QE_indices)
    energies = energies_DOS[QE_indices]
    dc_prefactor = config.sample['nr_orbitals'] \
                   / config.sample['area_unit_cell'] \
                   / (2 * np.pi)

    # get DC conductivity
    DC = np.zeros((2, n_energies))
    for i in range(2):
        for j in range(n_energies):

            en = energies[j]
            dosval = DOS[QE_indices[j]]
            dcval = 0.
            for k in range(tnr):
                W = window_DC(k + 1, tnr)
                cexp = np.exp(-1j * k * t_step * en)
                add_dcv = W * (cexp * corr_DC[i, j, k]).real
                dcval += add_dcv
            DC[i, j] = dc_prefactor * t_step * dosval * dcval

    # correct for spin
    if config.generic['correct_spin']:
        DC = 2. * DC

    return energies, DC
