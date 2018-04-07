"""graphene.py contains hoppings and geometric data for a
graphene tipsi model.

Functions
----------
    lattice
        Returns honeycomb tipsi.Lattice
    hop_dict_nn
        Returns graphene nearest neighbor tipsi.HopDict
    sheet
        Returns rectangular graphene sheet tipsi.SiteSet
    pbc
        Periodic boundary conditions function
    pbc_armchair
        Periodic boundary conditions function with armchair edge
    pbc_zigzag
        Periodic boundary conditions function with zigzag edge
"""

import sys
sys.path.append("..")
import tipsi
import numpy as np
  
# return graphene tipsi.Lattice  
def lattice(a = 0.24):
    # a is lattice constant in nm
    b = a / np.sqrt(3.)
    vectors        = [[1.5 * b, -0.5 * a, 0.], 
                      [1.5 * b, 0.5 * a, 0.]]
    orbital_coords = [[-b / 2., 0., 0.], 
                      [b / 2., 0., 0.]]
    return tipsi.Lattice(vectors, orbital_coords)
             
# return graphene nearest neighbor tipsi.HopDict
def hop_dict_nn(t = 2.7, e = 0.):
    # t and e are hopping value and onsite potential in eV
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
  
# fill tipsi.SiteSet with diamond shaped graphene sheet sites  
def sheet(W, H):
    site_set = tipsi.SiteSet()
    for i in range(W):
        for j in range(H):
            unit_cell_coords = (i, j, 0)
            site_set.add_site(unit_cell_coords, 0)
            site_set.add_site(unit_cell_coords, 1)
    return site_set

# fill tipsi.SiteSet with rectangular graphene sheet sites  
def sheet_rectangle(W, H):
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

# pbc in all directions for a diamond shaped sample
def pbc(W, H, unit_cell_coords, orbital):
    x, y, z = unit_cell_coords
    return (x % W, y % H, z), orbital

# pbc in all directions for a rectangular sample
def pbc_rectangle(W, H, unit_cell_coords, orbital):
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
    
# pbc for armchair boundary
def pbc_armchair(W, H, unit_cell_coords, orbital):
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
    
# pbc for zigzag boundary
def pbc_zigzag(W, H, unit_cell_coords, orbital):
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