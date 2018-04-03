"""input.py contains tools for reading data.

Functions
----------
    read_sample
        Read sample object from file.
    read_config
        Read config object from file.
    read_corr_DOS
        Read DOS correlation function from file.
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

# this is ugly
from .builder import *
from .config import *

def read_sample(filename, lattice = None, bc_func = bc_default, \
                nr_processes = 1):
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
    return Sample(lattice, bc_func = bc_func, \
                  nr_processes = nr_processes, \
                  read_from_file = filename)

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
    config = Config()
    config.sample = dict.sample
    config.generic = dict.generic
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

    f = open(filename,'r')

    n_samples = int(f.readline().split()[-1])
    n_timesteps = int(f.readline().split()[-1])
    corr_DOS = np.zeros(n_timesteps, dtype = complex)

    for i in range(n_samples):
        temp_string = f.readline().split()
        for j in range(n_timesteps):
            line = f.readline().split()
            corr_DOS[j] += float(line[1]) + 1j * float(line[2])

    return corr_DOS / n_samples

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

    f = open(filename,'r')

    n_samples = int(f.readline().split()[-1])
    n_timesteps = int(f.readline().split()[-1])
    corr_AC = np.zeros((4, n_timesteps), dtype = complex)

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

    f = open(filename,'r')

    n_q_points = int(f.readline().split()[-1])
    n_samples = int(f.readline().split()[-1])
    n_timesteps = int(f.readline().split()[-1])
    corr_dyn_pol = np.zeros((n_q_points, n_timesteps), dtype = complex)

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

    f = open(filename,'r')

    n_samples = int(f.readline().split()[-1])
    n_energies = int(f.readline().split()[-1])
    n_t_steps = int(f.readline().split()[-1])
    corr_DC = np.zeros((2, n_energies, n_t_steps), dtype = complex)

    for i in range(n_samples):
        temp_string = f.readline().split()
        for j in range(n_energies):
            temp_string = f.readline().split()
            for k in range(n_t_steps):
                line = f.readline().split()
                corr_DC[0, j, k] += float(line[1]) + 1j * float(line[2])
                corr_DC[1, j, k] += float(line[3]) + 1j * float(line[4])

    return corr_DC / n_samples

def read_wannier90(lat_file, coord_file, ham_file):
    """Read Lattice and HopDict information from Wannier90 file

    Parameters
    ----------
    lat_file : string
        read lattice vectors and atom numbers from this file
        usually named "*.win"
    coord_file : string
        read coordinate information from this file
        usually named "*_centres.xyz"
    ham_file : string
        read Hamiltonian information from this file
        usually named "*_hr.dat"

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
    with open(lat_file) as f:
        lat_content = f.readlines()

    # prepare
    parsing_lattice = False
    parsing_unitcell = False
    lattice_vectors = []
    # orbital_coords = []
    nr_atoms = 0

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

        # atoms numbers within unit cell
        if line.startswith("begin atoms_cart"):
            parsing_unitcell = True
        elif line.startswith("end atoms_cart"):
            parsing_unitcell = False
        elif parsing_unitcell:
            nr_atoms += 1
            # data = line.split()
            # orbital_coords.append([float(x) for x in data[1:]])

    # open coordinate file
    with open(lat_file) as f:
        coord_content = f.readlines()

    # prepare
    orbital_coords = []

    # parse
    for line in coord_content[2:-nr_atoms]:
        data = line.split()
        orbital_coords.append([float(x) for x in data[1:]])

    # init lattice object
    lat = Lattice(lattice_vectors, orbital_coords)

    #######################
    # MAKE HOPDICT OBJECT #
    #######################

    # open hamiltonian file
    with open(ham_file) as f:
        ham_content = f.readlines()
    nr_orbitals = int(ham_content[1])
    nr_hoppings = int(ham_content[2])
    skip_lines = 3+int(np.ceil(nr_hoppings/15))

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
        unit_cell_coord_new = (x,y,z)
        orb0 = int(data[3])-1
        orb1 = int(data[4])-1
        hop = float(data[5])+1j*float(data[6])

        # if new (x,y,z), fill hop_dict with old one
        if unit_cell_coord_new != unit_cell_coord_old:
            if unit_cell_coord_old:
                hop_dict.set(unit_cell_coord_old, hop_matrix)
            hop_matrix = np.zeros((nr_orbitals, nr_orbitals), \
                                  dtype = complex)
            unit_cell_coord_old = unit_cell_coord_new

        # if same (x,y,z), fill hopping matrix
        hop_matrix[orb0, orb1] = hop

    # enter last hop_matrix
    hop_dict.set(unit_cell_coord_old, hop_matrix)

    return lat, hop_dict
