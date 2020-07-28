"""input.py contains tools for reading data.

Functions
----------
    read_sample
        Read sample object from file.
    read_config
        Read config object from file.
    read_corr_DOS
        Read DOS correlation function from file.
    read_corr_LDOS
        Read LDOS correlation function from file.
    read_corr_AC
        Read AC correlation function from file.
    read_corr_dyn_pol
        Read dynamical polarization correlation function from file.
    read_corr_DC
        Read DC correlation function from file.
    read_wannier90
        Read Wannier90 files.
"""

################
# dependencies
################

# numerics & math
import numpy as np

from .builder import *
from .config import *


def read_sample(filename, lattice=None, bc_func=bc_default, nr_processes=1):
    """Read Sample object from file

    Parameters
    ----------
    filename : string
        read Sample object from this file
    lattice : Lattice object
        lattice corresponding to the sample; default: None
    bc_func : function
        boundary conditions function; default: bc_default
    nr_processes : integer
        number of processes to use for numerically
        expensive Sample methods; default: 1

    Returns
    ----------
    Sample object
    """
    return Sample(lattice, bc_func=bc_func,
                  nr_processes=nr_processes,
                  read_from_file=filename)


def read_config(filename):
    """Read Config object from file

    Parameters
    ----------
    filename : string
        read Config object from this file

    Returns
    ----------
    Config object
    """

    with open(filename, 'rb') as f:
        dict = pickle.load(f)
    config = Config(read_from_file=True)
    config.sample = dict.sample
    config.generic = dict.generic
    config.LDOS = dict.LDOS
    config.dyn_pol = dict.dyn_pol
    config.DC_conductivity = dict.DC_conductivity
    config.quasi_eigenstates = dict.quasi_eigenstates
    config.output = dict.output
    return config


def read_corr_DOS(filename):
    """Read DOS correlation from file

    Parameters
    ----------
    filename : string
        read correlation values from this file

    Returns
    ----------
    corr_DOS : (n_timesteps) list of complex floats
        the DOS correlation function
    """

    f = open(filename, 'r')

    n_samples = int(f.readline().split()[-1])
    n_timesteps = int(f.readline().split()[-1])
    corr_DOS = np.zeros(n_timesteps + 1, dtype=complex)

    for i in range(n_samples):
        temp_string = f.readline().split()
        for j in range(n_timesteps + 1):
            line = f.readline().split()
            corr_DOS[j] += float(line[1]) + 1j * float(line[2])

    return corr_DOS / n_samples


def read_corr_LDOS(filename):
    """Read LDOS correlation from file

    Parameters
    ----------
    filename : string
        read correlation values from this file

    Returns
    ----------
    corr_LDOS : (n_timesteps) list of complex floats
        the LDOS correlation function
    """

    f = open(filename, 'r')

    n_samples = int(f.readline().split()[-1])
    n_timesteps = int(f.readline().split()[-1])
    corr_LDOS = np.zeros(n_timesteps + 1, dtype=complex)

    for i in range(n_samples):
        temp_string = f.readline().split()
        for j in range(n_timesteps + 1):
            line = f.readline().split()
            corr_LDOS[j] = float(line[1]) + 1j * float(line[2])

    return corr_LDOS / n_samples


def read_corr_AC(filename):
    """Read AC correlation from file

    Parameters
    ----------
    filename : string
        read correlation values from this file

    Returns
    ----------
    corr_AC : (4, n_timesteps) list of complex floats
        the AC correlation function
    """

    f = open(filename, 'r')

    n_samples = int(f.readline().split()[-1])
    n_timesteps = int(f.readline().split()[-1])
    corr_AC = np.zeros((4, n_timesteps), dtype=complex)

    for i in range(n_samples):
        temp_string = f.readline().split()
        for j in range(n_timesteps):
            line = f.readline().split()
            corr_AC[0, j] += float(line[1]) + 1j * float(line[2])
            corr_AC[1, j] += float(line[3]) + 1j * float(line[4])
            corr_AC[2, j] += float(line[5]) + 1j * float(line[6])
            corr_AC[3, j] += float(line[7]) + 1j * float(line[8])

    return corr_AC / n_samples


def read_corr_dyn_pol(filename):
    """Read dynamical polarization correlation from file

    Parameters
    ----------
    filename : string
        read correlation values from this file

    Returns
    ----------
    corr_dyn_pol : (n_q_points, n_timesteps) list of complex floats
        the dynamical polarization  correlation function
    """

    f = open(filename, 'r')

    n_q_points = int(f.readline().split()[-1])
    n_samples = int(f.readline().split()[-1])
    n_timesteps = int(f.readline().split()[-1])
    corr_dyn_pol = np.zeros((n_q_points, n_timesteps), dtype=complex)

    for i_q in range(n_q_points):
        temp_string = f.readline().split()
        for i in range(n_samples):
            temp_string = f.readline().split()
            for j in range(n_timesteps):
                line = f.readline().split()
                corr_dyn_pol[i_q, j] += float(line[1])

    return corr_dyn_pol / n_samples


def read_corr_DC(filename):
    """Read DC conductivity correlation from file

    Parameters
    ----------
    filename : string
        read DC conductivity correlation values from this file

    Returns
    ----------
    corr_DC : (2, n_energies, n_t_steps) list of complex floats
        the dynamical polarization  correlation function
    """

    f = open(filename, 'r')

    n_samples = int(f.readline().split()[-1])
    n_energies = int(f.readline().split()[-1])
    n_t_steps = int(f.readline().split()[-1])
    corr_DC = np.zeros((2, n_energies, n_t_steps), dtype=complex)

    for i in range(n_samples):
        temp_string = f.readline().split()
        for j in range(n_energies):
            temp_string = f.readline().split()
            for k in range(n_t_steps):
                line = f.readline().split()
                corr_DC[0, j, k] += float(line[1]) + 1j * float(line[2])
                corr_DC[1, j, k] += float(line[3]) + 1j * float(line[4])

    return corr_DC / n_samples


def read_wannier90(lat_file,
                   coord_file,
                   ham_file,
                   correct_file=False,
                   en_cutoff=0.0):
    r"""Read Lattice and HopDict information from Wannier90 file

    Parameters
    ----------
    lat_file : string
        read lattice vectors from this file,
        usually named "\*.win"
    coord_file : string
        read wannier centres from this file,
        usually named "\*_centres.xyz"
    ham_file : string
        read hopping terms from this file,
        usually named "\*_hr.dat"
    correct_file : string, optional
        correction terms for hoppings, available since Wannier90 2.1,
        usually named "\*_wsvec.dat"
    en_cutoff : float, optional
        cut-off energy that absolute value of hopping less than en_cutoff \
        will be treated as 0
        Default: 0 which means no cut-off

    Returns
    ----------
    lat : Lattice object
        contains geometric information
    hop_dict : HopDict object
        contains electric information
    """

    #######################
    # MAKE LATTICE OBJECT #
    #######################

    # open lattice file
    with open(lat_file, 'r') as f:
        lat_content = f.readlines()

    # prepare
    parsing_lattice = False
    parsing_unitcell = False
    lattice_vectors = []

    # parse
    for line in lat_content:

        # lattice vectors
        if line.startswith("begin unit_cell_cart"):
            parsing_lattice = True
        elif line.startswith("end unit_cell_cart"):
            parsing_lattice = False
        elif parsing_lattice:
            data = line.split()
            lattice_vectors.append([float(x) for x in data])

    # open coordinate file
    with open(coord_file, 'r') as f:
        coord_content = f.readlines()

    # prepare
    orbital_coords = []

    # parse
    for line in coord_content[2:]:
        data = line.split()
        if data[0] == 'X':
            orbital_coords.append([float(x) for x in data[1:]])

    # init lattice object
    lat = Lattice(lattice_vectors, orbital_coords)

    #######################
    # MAKE HOPDICT OBJECT #
    #######################

    # open hopping file
    with open(ham_file, 'r') as f:
        ham_content = f.readlines()
    nr_wann = int(ham_content[1])
    nr_hoppings = int(ham_content[2])
    skip_lines = 3 + int(np.ceil(nr_hoppings / 15))

    # prepare
    hop_dict = HopDict()
    unit_cell_coord_old = False

    # parse
    for line in ham_content[skip_lines:]:

        # read data
        data = line.split()
        x = int(data[0])
        y = int(data[1])
        z = int(data[2])
        unit_cell_coord_new = (x, y, z)
        orb0 = int(data[3]) - 1
        orb1 = int(data[4]) - 1
        hop = float(data[5]) + 1j * float(data[6])
        if np.abs(hop) < en_cutoff:
            hop = 0 + 0j

        # add hopping terms
        # if new (x,y,z), fill hop_dict with old one
        if unit_cell_coord_new != unit_cell_coord_old:
            if unit_cell_coord_old:
                hop_dict.set(unit_cell_coord_old, hop_matrix)
            hop_matrix = np.zeros((nr_wann, nr_wann), dtype=complex)
            unit_cell_coord_old = unit_cell_coord_new

        # if same (x,y,z), fill hopping matrix
        hop_matrix[orb0, orb1] = hop

    # enter last hop_matrix
    hop_dict.set(unit_cell_coord_old, hop_matrix)

    if correct_file:
        # read correction terms
        cor = {}
        with open(correct_file, 'r') as iterator:
            next(iterator)  # skip comment line
            for first_line in iterator:
                data = first_line.split()
                x0 = int(data[0])
                y0 = int(data[1])
                z0 = int(data[2])
                orb0 = int(data[3]) - 1
                orb1 = int(data[4]) - 1
                N = int(next(iterator))
                sites_cor = []
                for i in range(N):
                    data = next(iterator).split()
                    x = int(data[0])
                    y = int(data[1])
                    z = int(data[2])
                    sites_cor.append((x, y, z))
                cor[(x0, y0, z0, orb0, orb1)] = sites_cor

        # apply correction terms
        for x0, y0, z0, orb0, orb1 in cor.keys():
            N = len(cor[(x0, y0, z0, orb0, orb1)])
            hop = hop_dict.dict[(x0, y0, z0)][orb0, orb1]
            hop_dict.dict[(x0, y0, z0)][orb0, orb1] = 0

            for i in range(N):
                x, y, z = cor[(x0, y0, z0, orb0, orb1)][i]
                x, y, z = x + x0, y + y0, z + z0
                if (x, y, z) in hop_dict.dict.keys():
                    hop_dict.dict[(x, y, z)][orb0, orb1] = hop / N
                else:
                    hop_matrix = np.zeros((nr_wann, nr_wann), dtype=complex)
                    hop_matrix[orb0, orb1] = hop / N
                    hop_dict.set((x, y, z), hop_matrix)

    return lat, hop_dict
