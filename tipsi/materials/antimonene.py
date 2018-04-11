import sys
sys.path.append("../..")
import tipsi
import numpy as np
        
def lattice(SOC = True, a = 0.411975806, z = 0.16455347):
    """Antimonene lattice.
    
    Parameters
    ----------
    SOC : bool
        set to True to include spin orbit coupling
    a : float
        lattice constant
    z : float
        vertical displacement
        
    Returns
    ----------
    tipsi.Lattice object
        Antimonene lattice.
    """
    
    b = a / np.sqrt(3.)
    vectors = [[1.5 * b, -0.5 * a, 0.], 
               [1.5 * b, 0.5 * a, 0.]]
    if SOC:
        n_orbitals_per_site = 6
    else:
        n_orbitals_per_site = 3
    orbital_coords = \
        [[-b / 2., 0., -z / 2.] for i in range(n_orbitals_per_site)] + \
        [[b / 2., 0., z / 2.] for i in range(n_orbitals_per_site)]
    return tipsi.Lattice(vectors, orbital_coords)
             
def SOC_matrix(SOC_lambda):
    """On-site SOC matrix.
    
    Parameters
    ----------
    SOC_lambda : float
        strength of spin orbit coupling
        
    Returns
    ----------
    M : (6, 6) numpy array
        on-site SOC matrix
    """
    
    M = 0.5 * SOC_lambda * np.array([ \
        [   0.0000 + 0.0000j,   0.0000 + 0.5854j,   0.0000 - 0.5854j,  -0.0000 + 0.0000j,   0.7020 - 0.4053j,  -0.0000 - 0.8107j ], \
        [   0.0000 - 0.5854j,   0.0000 + 0.0000j,   0.0000 + 0.5854j,  -0.7020 + 0.4053j,   0.0000 + 0.0000j,  -0.7020 - 0.4053j ], \
        [   0.0000 + 0.5854j,   0.0000 - 0.5854j,   0.0000 + 0.0000j,  -0.0000 + 0.8107j,   0.7020 + 0.4053j,  -0.0000 + 0.0000j ], \
        [   0.0000 + 0.0000j,  -0.7020 - 0.4053j,   0.0000 - 0.8107j,   0.0000 + 0.0000j,   0.0000 - 0.5854j,   0.0000 + 0.5854j ], \
        [   0.7020 + 0.4053j,  -0.0000 + 0.0000j,   0.7020 - 0.4053j,   0.0000 + 0.5854j,   0.0000 + 0.0000j,   0.0000 - 0.5854j ], \
        [   0.0000 + 0.8107j,  -0.7020 + 0.4053j,   0.0000 + 0.0000j,   0.0000 - 0.5854j,   0.0000 + 0.5854j,   0.0000 + 0.0000j ]  \
        ])
    return M

def hop_dict(SOC = True, SOC_lambda = 0.34):
    """Antimonene hopping dictionary.
    
    Parameters
    ----------
    SOC : bool
        set to True to include spin orbit coupling
    SOC_lambda : float
        strength of spin orbit coupling
        
    Returns
    ----------
    hops : tipsi.HopDict object
        antimonene HopDict
    """
    
    # hopping parameters
    t_01 = -2.09
    t_02 =  0.47
    t_03 =  0.18
    t_04 = -0.50
    t_05 = -0.11
    t_06 =  0.21
    t_07 =  0.08
    t_08 = -0.07
    t_09 =  0.07
    t_10 =  0.07
    t_11 = -0.06
    t_12 = -0.06
    t_13 = -0.03
    t_14 = -0.03
    t_15 = -0.04
    
    # first construct HopDict for SOC = False
    hop_dict = tipsi.HopDict()
    rel_unit_cells = [(0,0,0), (1,0,0), (0,1,0), \
        (1,-1,0), (1,1,0), (1,-2,0), (0,2,0), \
        (2,-1,0), (2,0,0), (2,-2,0), (-1,1,0), \
        (-1,0,0), (0,-1,0), (-1,2,0), (0,-2,0), \
        (-2,1,0), (-2,0,0), (-2,2,0), (-1,-1,0)]
    for uc in rel_unit_cells:
        hop_dict.empty(uc, (6, 6))
    
    ####################
    # NEAREST NEIGHBOURS
    ####################
    
    hop_dict.set_element((0,-1,0), (0,3), t_01)
    hop_dict.set_element((0,0,0), (1,4), t_01)
    hop_dict.set_element((-1,0,0), (2,5), t_01)
    hop_dict.set_element((0,1,0), (3,0), t_01)
    hop_dict.set_element((0,0,0), (4,1), t_01)
    hop_dict.set_element((1,0,0), (5,2), t_01)
    
    hop_dict.set_element((0,0,0), (0,3), t_02)
    hop_dict.set_element((-1,0,0), (0,3), t_02)
    hop_dict.set_element((-1,0,0), (1,4), t_02)
    hop_dict.set_element((0,-1,0), (1,4), t_02)
    hop_dict.set_element((0,-1,0), (2,5), t_02)
    hop_dict.set_element((0,0,0), (2,5), t_02)
    hop_dict.set_element((0,0,0), (3,0), t_02)
    hop_dict.set_element((1,0,0), (3,0), t_02)
    hop_dict.set_element((1,0,0), (4,1), t_02)
    hop_dict.set_element((0,1,0), (4,1), t_02)
    hop_dict.set_element((0,1,0), (5,2), t_02)
    hop_dict.set_element((0,0,0), (5,2), t_02)
    
    hop_dict.set_element((0,0,0), (0,4), t_07)
    hop_dict.set_element((-1,0,0), (0,5), t_07)
    hop_dict.set_element((-1,0,0), (1,5), t_07)
    hop_dict.set_element((0,-1,0), (1,3), t_07)
    hop_dict.set_element((0,-1,0), (2,3), t_07)
    hop_dict.set_element((0,0,0), (2,4), t_07)
    hop_dict.set_element((0,0,0), (3,1), t_07)
    hop_dict.set_element((1,0,0), (3,2), t_07)
    hop_dict.set_element((1,0,0), (4,2), t_07)
    hop_dict.set_element((0,1,0), (4,0), t_07)
    hop_dict.set_element((0,1,0), (5,0), t_07)
    hop_dict.set_element((0,0,0), (5,1), t_07)
    
    ####################
    # NEXT-NEAREST NEIGHBOURS
    ####################
    
    hop_dict.set_element((-1,1,0), (0,0), t_03)
    hop_dict.set_element((0,1,0), (0,0), t_03)
    hop_dict.set_element((1,-1,0), (0,0), t_03)
    hop_dict.set_element((0,-1,0), (0,0), t_03)
    hop_dict.set_element((0,-1,0), (1,1), t_03)
    hop_dict.set_element((-1,0,0), (1,1), t_03)
    hop_dict.set_element((0,1,0), (1,1), t_03)
    hop_dict.set_element((1,0,0), (1,1), t_03)
    hop_dict.set_element((-1,0,0), (2,2), t_03)
    hop_dict.set_element((-1,1,0), (2,2), t_03)
    hop_dict.set_element((1,0,0), (2,2), t_03)
    hop_dict.set_element((1,-1,0), (2,2), t_03)
    hop_dict.set_element((-1,1,0), (3,3), t_03)
    hop_dict.set_element((0,1,0), (3,3), t_03)
    hop_dict.set_element((1,-1,0), (3,3), t_03)
    hop_dict.set_element((0,-1,0), (3,3), t_03)
    hop_dict.set_element((0,-1,0), (4,4), t_03)
    hop_dict.set_element((-1,0,0), (4,4), t_03)
    hop_dict.set_element((0,1,0), (4,4), t_03)
    hop_dict.set_element((1,0,0), (4,4), t_03)
    hop_dict.set_element((-1,0,0), (5,5), t_03)
    hop_dict.set_element((-1,1,0), (5,5), t_03)
    hop_dict.set_element((1,0,0), (5,5), t_03)
    hop_dict.set_element((1,-1,0), (5,5), t_03)
    
    hop_dict.set_element((0,1,0), (0,1), t_04)
    hop_dict.set_element((-1,1,0), (0,2), t_04)
    hop_dict.set_element((-1,0,0), (1,2), t_04)
    hop_dict.set_element((0,-1,0), (1,0), t_04)
    hop_dict.set_element((1,-1,0), (2,0), t_04)
    hop_dict.set_element((1,0,0), (2,1), t_04)
    hop_dict.set_element((0,-1,0), (3,4), t_04)
    hop_dict.set_element((1,-1,0), (3,5), t_04)
    hop_dict.set_element((1,0,0), (4,5), t_04)
    hop_dict.set_element((0,1,0), (4,3), t_04)
    hop_dict.set_element((-1,1,0), (5,3), t_04)
    hop_dict.set_element((-1,0,0), (5,4), t_04)
    
    hop_dict.set_element((0,-1,0), (0,1), t_06)
    hop_dict.set_element((1,-1,0), (0,2), t_06)
    hop_dict.set_element((1,0,0), (1,2), t_06)
    hop_dict.set_element((0,1,0), (1,0), t_06)
    hop_dict.set_element((-1,1,0), (2,0), t_06)
    hop_dict.set_element((-1,0,0), (2,1), t_06)
    hop_dict.set_element((0,1,0), (3,4), t_06)
    hop_dict.set_element((-1,1,0), (3,5), t_06)
    hop_dict.set_element((-1,0,0), (4,5), t_06)
    hop_dict.set_element((0,-1,0), (4,3), t_06)
    hop_dict.set_element((1,-1,0), (5,3), t_06)
    hop_dict.set_element((1,0,0), (5,4), t_06)
    
    hop_dict.set_element((-1,0,0), (0,0), t_11)
    hop_dict.set_element((1,0,0), (0,0), t_11)
    hop_dict.set_element((-1,1,0), (1,1), t_11)
    hop_dict.set_element((1,-1,0), (1,1), t_11)
    hop_dict.set_element((0,-1,0), (2,2), t_11)
    hop_dict.set_element((0,1,0), (2,2), t_11)
    hop_dict.set_element((-1,0,0), (3,3), t_11)
    hop_dict.set_element((1,0,0), (3,3), t_11)
    hop_dict.set_element((-1,1,0), (4,4), t_11)
    hop_dict.set_element((1,-1,0), (4,4), t_11)
    hop_dict.set_element((0,-1,0), (5,5), t_11)
    hop_dict.set_element((0,1,0), (5,5), t_11)
    
    ####################
    # NEXT-NEXT-NEAREST NEIGHBOURS
    # ACROSS THE HEXAGON
    ####################
    
    hop_dict.set_element((-1,1,0), (0,4), t_08)
    hop_dict.set_element((-1,1,0), (0,5), t_08)
    hop_dict.set_element((-1,-1,0), (1,5), t_08)
    hop_dict.set_element((-1,-1,0), (1,3), t_08)
    hop_dict.set_element((1,-1,0), (2,3), t_08)
    hop_dict.set_element((1,-1,0), (2,4), t_08)
    hop_dict.set_element((1,-1,0), (3,1), t_08)
    hop_dict.set_element((1,-1,0), (3,2), t_08)
    hop_dict.set_element((1,1,0), (4,2), t_08)
    hop_dict.set_element((1,1,0), (4,0), t_08)
    hop_dict.set_element((-1,1,0), (5,0), t_08)
    hop_dict.set_element((-1,1,0), (5,1), t_08)
    
    hop_dict.set_element((1,-1,0), (0,4), t_12)
    hop_dict.set_element((-1,-1,0), (0,5), t_12)
    hop_dict.set_element((-1,1,0), (1,5), t_12)
    hop_dict.set_element((1,-1,0), (1,3), t_12)
    hop_dict.set_element((-1,-1,0), (2,3), t_12)
    hop_dict.set_element((-1,1,0), (2,4), t_12)
    hop_dict.set_element((-1,1,0), (3,1), t_12)
    hop_dict.set_element((1,1,0), (3,2), t_12)
    hop_dict.set_element((1,-1,0), (4,2), t_12)
    hop_dict.set_element((-1,1,0), (4,0), t_12)
    hop_dict.set_element((1,1,0), (5,0), t_12)
    hop_dict.set_element((1,-1,0), (5,1), t_12)
    
    ####################
    # NEXT-NEXT-NEAREST NEIGHBOURS
    # ACROSS THE ZIGZAG
    ####################
    
    hop_dict.set_element((1,-2,0), (0,3), t_05)
    hop_dict.set_element((0,-2,0), (0,3), t_05)
    hop_dict.set_element((0,1,0), (1,4), t_05)
    hop_dict.set_element((1,0,0), (1,4), t_05)
    hop_dict.set_element((-2,1,0), (2,5), t_05)
    hop_dict.set_element((-2,0,0), (2,5), t_05)
    hop_dict.set_element((-1,2,0), (3,0), t_05)
    hop_dict.set_element((0,2,0), (3,0), t_05)
    hop_dict.set_element((-1,0,0), (4,1), t_05)
    hop_dict.set_element((0,-1,0), (4,1), t_05)
    hop_dict.set_element((2,-1,0), (5,2), t_05)
    hop_dict.set_element((2,0,0), (5,2), t_05)
    
    hop_dict.set_element((-2,1,0), (0,3), t_09)
    hop_dict.set_element((0,1,0), (0,3), t_09)
    hop_dict.set_element((0,-2,0), (1,4), t_09)
    hop_dict.set_element((-2,0,0), (1,4), t_09)
    hop_dict.set_element((1,0,0), (2,5), t_09)
    hop_dict.set_element((1,-2,0), (2,5), t_09)
    hop_dict.set_element((2,-1,0), (3,0), t_09)
    hop_dict.set_element((0,-1,0), (3,0), t_09)
    hop_dict.set_element((0,2,0), (4,1), t_09)
    hop_dict.set_element((2,0,0), (4,1), t_09)
    hop_dict.set_element((-1,2,0), (5,2), t_09)
    hop_dict.set_element((-1,0,0), (5,2), t_09)
    
    hop_dict.set_element((0,1,0), (0,4), t_10)
    hop_dict.set_element((-2,1,0), (0,5), t_10)
    hop_dict.set_element((-2,0,0), (1,5), t_10)
    hop_dict.set_element((0,-2,0), (1,3), t_10)
    hop_dict.set_element((1,-2,0), (2,3), t_10)
    hop_dict.set_element((1,0,0), (2,4), t_10)
    hop_dict.set_element((0,-1,0), (3,1), t_10)
    hop_dict.set_element((2,-1,0), (3,2), t_10)
    hop_dict.set_element((2,0,0), (4,2), t_10)
    hop_dict.set_element((0,2,0), (4,0), t_10)
    hop_dict.set_element((-1,2,0), (5,0), t_10)
    hop_dict.set_element((-1,0,0), (5,1), t_10)
    
    hop_dict.set_element((1,0,0), (0,3), t_13)
    hop_dict.set_element((-2,0,0), (0,3), t_13)
    hop_dict.set_element((-2,1,0), (1,4), t_13)
    hop_dict.set_element((1,-2,0), (1,4), t_13)
    hop_dict.set_element((0,-2,0), (2,5), t_13)
    hop_dict.set_element((0,1,0), (2,5), t_13)
    hop_dict.set_element((-1,0,0), (3,0), t_13)
    hop_dict.set_element((2,0,0), (3,0), t_13)
    hop_dict.set_element((-1,2,0), (4,1), t_13)
    hop_dict.set_element((2,-1,0), (4,1), t_13)
    hop_dict.set_element((0,2,0), (5,2), t_13)
    hop_dict.set_element((0,-1,0), (5,2), t_13)
    
    ####################
    # NEXT-NEXT-NEXT-NEAREST NEIGHBOURS
    # ACROSS THE ZIGZAG
    ####################
    
    hop_dict.set_element((0,-2,0), (0,1), t_14)
    hop_dict.set_element((2,-2,0), (0,2), t_14)
    hop_dict.set_element((2,0,0), (1,2), t_14)
    hop_dict.set_element((0,2,0), (1,0), t_14)
    hop_dict.set_element((-2,2,0), (2,0), t_14)
    hop_dict.set_element((-2,0,0), (2,1), t_14)
    hop_dict.set_element((0,2,0), (3,4), t_14)
    hop_dict.set_element((-2,2,0), (3,5), t_14)
    hop_dict.set_element((-2,0,0), (4,5), t_14)
    hop_dict.set_element((0,-2,0), (4,3), t_14)
    hop_dict.set_element((2,-2,0), (5,3), t_14)
    hop_dict.set_element((2,0,0), (5,4), t_14)
    
    hop_dict.set_element((0,-2,0), (1,0), t_15)
    hop_dict.set_element((2,-2,0), (2,0), t_15)
    hop_dict.set_element((2,0,0), (2,1), t_15)
    hop_dict.set_element((0,2,0), (0,1), t_15)
    hop_dict.set_element((-2,2,0), (0,2), t_15)
    hop_dict.set_element((-2,0,0), (1,2), t_15)
    hop_dict.set_element((0,2,0), (4,3), t_15)
    hop_dict.set_element((-2,2,0), (5,3), t_15)
    hop_dict.set_element((-2,0,0), (5,4), t_15)
    hop_dict.set_element((0,-2,0), (3,4), t_15)
    hop_dict.set_element((2,-2,0), (3,5), t_15)
    hop_dict.set_element((2,0,0), (4,5), t_15)
    
    # deal with SOC
    if SOC:
        for rel_unit_cell, hop in hop_dict.dict.items():
            newhop = np.zeros((12, 12), dtype = "complex")
            hop00 = hop[0:3,0:3]
            hop01 = hop[0:3,3:6]
            hop10 = hop[3:6,0:3]
            hop11 = hop[3:6,3:6]
            newhop[0:3,0:3] = hop00
            newhop[3:6,3:6] = hop00
            newhop[0:3,6:9] = hop01
            newhop[3:6,9:12] = hop01
            newhop[6:9,0:3] = hop10
            newhop[9:12,3:6] = hop10
            newhop[6:9,6:9] = hop11
            newhop[9:12,9:12] = hop11
            hop_dict.set(rel_unit_cell, newhop)
        hop_dict.dict[(0,0,0)] += np.kron(np.eye(2), SOC_matrix(SOC_lambda))
    
    return hop_dict

def sheet(W, H, SOC = True):
    """Antimonene SiteSet for a rectangular sheet.
    
    Parameters
    ----------
    W : integer
        width of SiteSet in unit cells
    H : integer
        height of SiteSet in unit cells
    SOC : bool
        set to True to include spin orbit coupling
        
    Returns
    ----------
    site_set : tipsi.SiteSet object
        rectangular antimonene SiteSet
    """
    
    site_set = tipsi.SiteSet()
    if SOC:
        n_orbs = 12
    else:
        n_orbs = 6
    for x in range(int(W / 2)):
        for y in range(H):
            i, j = x - y, x + y
            for orb in range(n_orbs):
                unit_cell_coords = (i, j, 0)
                site_set.add_site(unit_cell_coords, orb)
                unit_cell_coords = (i, j + 1, 0)
                site_set.add_site(unit_cell_coords, orb)
    return site_set
    
def pbc(W, H, unit_cell_coords, orbital):
    """PBC for a rectangular antimonene sheet.
    
    Parameters
    ----------
    W : integer
        width of SiteSet in unit cells
    H : integer
        height of SiteSet in unit cells
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index
        
    Returns
    ----------
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index
    """
    
    # get input
    x, y, z = unit_cell_coords
    # transform to rectangular coords (xloc, yloc)
    xloc = (x + y) / 2.
    yloc = (y - x) / 2.
    # use standard pbc
    xloc = xloc % (W / 2)
    yloc = yloc % H
    # transform back
    x = int(xloc - yloc)
    y = int(xloc + yloc)
    # done
    return (x, y, z), orbital
    
def pbc_armchair(W, H, unit_cell_coords, orbital):
    """PBC for a rectangular antimonene sheet
    with an armchair edge.
    
    Parameters
    ----------
    W : integer
        width of SiteSet in unit cells
    H : integer
        height of SiteSet in unit cells
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index
        
    Returns
    ----------
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index
    """
    
    # get input
    x, y, z = unit_cell_coords
    # transform to rectangular coords (xloc, yloc)
    xloc = (x + y) / 2.
    yloc = (y - x) / 2.
    # use zigzag pbc
    xloc = xloc % (W / 2)
    yloc = yloc
    # transform back
    x = int(xloc - yloc)
    y = int(xloc + yloc)
    # done
    return (x, y, z), orbital
    
def pbc_zigzag(W, H, unit_cell_coords, orbital):
    """PBC for a rectangular antimonene sheet
    with a zigzag edge.
    
    Parameters
    ----------
    W : integer
        width of SiteSet in unit cells
    H : integer
        height of SiteSet in unit cells
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index
        
    Returns
    ----------
    unit_cell_coords : 3-tuple of integers
        unit cell coordinates
    orbital : integer
        orbital index
    """
    
    # get input
    x, y, z = unit_cell_coords
    # transform to rectangular coords (xloc, yloc)
    xloc = (x + y) / 2.
    yloc = (y - x) / 2.
    # use zigzag pbc
    xloc = xloc
    yloc = yloc % H
    # transform back
    x = int(xloc - yloc)
    y = int(xloc + yloc)
    # done
    return (x, y, z), orbital

def sample(W = 500, H = 500, SOC = True, SOC_lambda = 0.34, \
           a = 0.411975806, z = 0.16455347, nr_processes = 1):
    """Rectangular antimonene sample.
    
    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
    SOC : bool
        set to True to include spin orbit coupling
    SOC_lambda : float
        strength of spin orbit coupling
    a : float
        lattice constant
    z : float
        vertical displacement
    nr_processes : integer
        number of processes for sample building, optional (default 1)
        
    Returns
    ----------
    sample : tipsi.Sample object
        Antimonene sample.
    """
    
    # create lattice, hop_dict and pbc_wrap
    lat = lattice(SOC, a, z)
    hops = hop_dict(SOC, SOC_lambda)
    def pbc_wrap(unit_cell_coords, orbital):
        return pbc(W, H, unit_cell_coords, orbital)
    
    # create SiteSet object
    site_set = sheet(W, H, SOC)
    
    # make sample
    sample = tipsi.Sample(lat, site_set, pbc_wrap, nr_processes)

    # apply HopDict
    sample.add_hop_dict(hops)
    
    # rescale Hamiltonian
    sample.rescale_H(4.5)
    
    # done
    return sample