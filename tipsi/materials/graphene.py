import sys
sys.path.append("../..")
import tipsi
import numpy as np
  
def lattice(a = 0.24):
    """Graphene lattice.
    
    Parameters
    ----------
    a : float
        lattice constant
        
    Returns
    ----------
    tipsi.Lattice object
        Graphene lattice.
    """
    
    b = a / np.sqrt(3.)
    vectors        = [[1.5 * b, -0.5 * a, 0.], 
                      [1.5 * b, 0.5 * a, 0.]]
    orbital_coords = [[-b / 2., 0., 0.], 
                      [b / 2., 0., 0.]]
    return tipsi.Lattice(vectors, orbital_coords)
             
def hop_dict_nn(t = 2.7, e = 0.):
    """Graphene nearest neighbor HopDict.
    
    Parameters
    ----------
    t : float
        hopping constant
    e : float
        on-site potential
        
    Returns
    ----------
    hops : tipsi.HopDict object
        Graphene HopDict.
    """
    
    A_0   = [[e, t],
             [t, e]]
    A_nn0 = [[0., 0.],
             [t, 0.]]
    A_nn1 = [[0., t],
             [0., 0.]]
    hops = tipsi.HopDict()
    hops.set((0, 0, 0),  A_0)
    hops.set((1, 0, 0),  A_nn0)
    hops.set((-1, 0, 0), A_nn1)
    hops.set((0, 1, 0),  A_nn0)
    hops.set((0, -1, 0), A_nn1)
    return hops
  
def sheet(W, H):
    """Graphene SiteSet, diamond shaped.
    
    Parameters
    ----------
    W : integer
        width of the SiteSet, in unit cells
    H : integer
        height of the SiteSet, in unit cells
        
    Returns
    ----------
    site_set : tipsi.SiteSet object
        Diamond shaped graphene SiteSet.
    """
    
    site_set = tipsi.SiteSet()
    for i in range(W):
        for j in range(H):
            unit_cell_coords = (i, j, 0)
            site_set.add_site(unit_cell_coords, 0)
            site_set.add_site(unit_cell_coords, 1)
    return site_set

def sheet_rectangle(W, H):
    """Graphene SiteSet, rectangular.
    
    Parameters
    ----------
    W : integer
        width of the SiteSet, in unit cells
    H : integer
        height of the SiteSet, in unit cells
        
    Returns
    ----------
    site_set : tipsi.SiteSet object
        Rectangular graphene SiteSet.
    """
    
    site_set = tipsi.SiteSet()
    for x in range(int(W / 2)):
        for y in range(H):
            i, j = x - y, x + y
            unit_cell_coords = (i, j, 0)
            site_set.add_site(unit_cell_coords, 0)
            site_set.add_site(unit_cell_coords, 1)
            unit_cell_coords = (i, j + 1, 0)
            site_set.add_site(unit_cell_coords, 0)
            site_set.add_site(unit_cell_coords, 1)
    return site_set

def pbc(W, H, unit_cell_coords, orbital):
    """PBC for a diamond shaped graphene sample.
    
    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
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
    
    x, y, z = unit_cell_coords
    return (x % W, y % H, z), orbital

def pbc_rectangle(W, H, unit_cell_coords, orbital):
    """PBC for a rectangular graphene sample.
    
    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
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
    
def pbc_rectangle_armchair(W, H, unit_cell_coords, orbital):
    """PBC for a rectangular graphene sample
    with an armchair boundary.
    
    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
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
    
def pbc_rectangle_zigzag(W, H, unit_cell_coords, orbital):
    """PBC for a rectangular graphene sample
    with a zigzag boundary.
    
    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
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

def sample(W = 500, H = 500, a = 0.24, t = 2.8, e = 0.0, \
           nr_processes = 1):
    """Diamond shaped graphene sample.
    
    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
    a : float
        lattice constant, optional (default 0.24)
    t : float
        hopping constant, optional (default 2.8)
    e : float
        hopping constant, optional (default 0.0)
    nr_processes : integer
        number of processes for sample building, optional (default 1)
        
    Returns
    ----------
    sample : tipsi.Sample object
        Diamond shaped graphene sample.
    """
    
    # create lattice, hop_dict and pbc_wrap
    lat = lattice(a)
    hop_dict = hop_dict_nn(t, e)
    def pbc_wrap(unit_cell_coords, orbital):
        return pbc(W, H, unit_cell_coords, orbital)
    
    # create SiteSet object
    site_set = sheet(W, H)
    
    # make sample
    sample = tipsi.Sample(lat, site_set, pbc_wrap, nr_processes)

    # apply HopDict
    sample.add_hop_dict(hop_dict)
    
    # rescale Hamiltonian
    sample.rescale_H(9.)
    
    # done
    return sample

def sample_rectangle(W = 500, H = 500, a = 0.24, t = 2.8, \
                       e = 0.0, nr_processes = 1):
    """Rectangular graphene sample.
    
    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
    a : float
        lattice constant, optional (default 0.24)
    t : float
        hopping constant, optional (default 2.8)
    e : float
        hopping constant, optional (default 0.0)
    nr_processes : integer
        number of processes for sample building, optional (default 1)
        
    Returns
    ----------
    sample : tipsi.Sample object
        Rectangular graphene sample.
    """
    
    # create lattice, hop_dict and pbc_wrap
    lat = lattice(a)
    hops = hop_dict_nn(t, e)
    def pbc_wrap(unit_cell_coords, orbital):
        return pbc_rectangle(W, H, unit_cell_coords, orbital)
    
    # create SiteSet object
    site_set = sheet_rectangle(W, H)
    
    # make sample
    sample = tipsi.Sample(lat, site_set, pbc_wrap, nr_processes)

    # apply HopDict
    sample.add_hop_dict(hops)
    
    # rescale Hamiltonian
    sample.rescale_H(9.)
    
    # done
    return sample