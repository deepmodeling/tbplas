import sys
sys.path.append("../..")
import tipsi
import numpy as np

def lattice(dist_nn = 0.22156, dist_nnb = 0.07159, theta = 48.395, \
            thetaz = 108.657, dist_nz = 0.22378, dist_interz = 0.54923):
    """Black phosphorus lattice.

    Parameters
    ----------
    all kinds of floats
        don't touch them

    Returns
    ----------
    tipsi.Lattice object
        Black phosphorus lattice.
    """

    # useful params
    a = 2 * dist_nn * np.sin(np.radians(theta))
    b = 2 * dist_nnb + 2 * dist_nn * np.cos(np.radians(theta))
    c = dist_interz
    p = dist_nnb
    q = dist_nz * np.cos(np.radians(thetaz - 90))

    # lattice arrays
    vectors = [[a, 0., 0.], [0., b, 0.], [a / 2., 0., c]]
    orbital_coords = [[] for i in range(4)]
    orbital_coords[0] = [0., 0., 0.]
    orbital_coords[1] = [0., p, q]
    orbital_coords[2] = [a / 2., b / 2., q]
    orbital_coords[3] = [a / 2., b / 2. + p, 0.]

    return tipsi.Lattice(vectors, orbital_coords)


def hop_dict():
    """Black phosphorus HopDict.

    Returns
    ----------
    hop_dict : tipsi.HopDict object
        Black phosphorus HopDict.
    """

    # hopping parameters
    t_1 = -1.486
    t_2 = 3.729
    t_3 = -0.252
    t_4 = -0.071
    t_5 = 0.019
    t_6 = 0.186
    t_7 = -0.063
    t_8 = 0.101
    t_9 = -0.042
    t_10 = 0.073
    t_p1 = 0.524
    t_p2 = 0.180
    t_p3 = -0.123
    t_p4 = -0.168
    t_p5 = 0.005

    # initialize
    hop_dict = tipsi.HopDict()
    rel_unit_cells = [(-1, 1, 0), (1, 1, 0), (-1, 1, 1), \
        (-1, -1, 1), (1, 1, 1), (0, 0, 1), (-1, 0, 1), \
        (1, -1, 0), (-1, -1, 0), (0, 1, 1), (1, 0, 0), \
        (-2, 0, 1), (0, 0, 0), (0, -1, 0), (0, -1, 1), \
        (-2, 0, 0), (-1, 0, 0), (0, 1, 0), (-2, -1, 0), \
        (1, 0, 1), (-2, -1, 1)]
    for uc in rel_unit_cells:
        hop_dict.empty(uc, (4, 4))

    # make dictionary, leaving out all the conjugates
    # firstly, in-plane
    hop_dict.set_element((0, 0, 0), (0, 1), t_2)
    hop_dict.set_element((0, -1, 0), (0, 3), t_1)
    hop_dict.set_element((-1, -1, 0), (0, 3), t_1)
    hop_dict.set_element((0, 0, 0), (0, 2), t_5)
    hop_dict.set_element((-1, 0, 0), (0, 2), t_5)
    hop_dict.set_element((0, -1, 0), (0, 2), t_5)
    hop_dict.set_element((-1, -1, 0), (0, 2), t_5)
    hop_dict.set_element((0, 0, 0), (0, 3), t_4)
    hop_dict.set_element((-1, 0, 0), (0, 3), t_4)
    hop_dict.set_element((0, -1, 0), (0, 1), t_6)
    hop_dict.set_element((1, 0, 0), (0, 0), t_3)
    hop_dict.set_element((-1, 0, 0), (0, 0), t_3)
    hop_dict.set_element((0, 1, 0), (0, 0), t_7)
    hop_dict.set_element((0, -1, 0), (0, 0), t_7)
    hop_dict.set_element((-2, -1, 0), (0, 3), t_8)
    hop_dict.set_element((1, -1, 0), (0, 3), t_8)
    hop_dict.set_element((-1, -1, 0), (0, 1), t_9)
    hop_dict.set_element((1, -1, 0), (0, 1), t_9)
    hop_dict.set_element((1, 1, 0), (0, 0), t_10)
    hop_dict.set_element((-1, 1, 0), (0, 0), t_10)
    hop_dict.set_element((1, -1, 0), (0, 0), t_10)
    hop_dict.set_element((-1, -1, 0), (0, 0), t_10)
    hop_dict.set_element((0, 0, 0), (1, 2), t_1)
    hop_dict.set_element((-1, 0, 0), (1, 2), t_1)
    hop_dict.set_element((-1, -1, 0), (1, 3), t_5)
    hop_dict.set_element((0, -1, 0), (1, 3), t_5)
    hop_dict.set_element((0, 0, 0), (1, 3), t_5)
    hop_dict.set_element((-1, 0, 0), (1, 3), t_5)
    hop_dict.set_element((-1, -1, 0), (1, 2), t_4)
    hop_dict.set_element((0, -1, 0), (1, 2), t_4)
    hop_dict.set_element((1, 0, 0), (1, 1), t_3)
    hop_dict.set_element((-1, 0, 0), (1, 1), t_3)
    hop_dict.set_element((0, 1, 0), (1, 1), t_7)
    hop_dict.set_element((0, -1, 0), (1, 1), t_7)
    hop_dict.set_element((-2, 0, 0), (1, 2), t_8)
    hop_dict.set_element((1, 0, 0), (1, 2), t_8)
    hop_dict.set_element((1, 1, 0), (1, 1), t_10)
    hop_dict.set_element((-1, 1, 0), (1, 1), t_10)
    hop_dict.set_element((1, -1, 0), (1, 1), t_10)
    hop_dict.set_element((-1, -1, 0), (1, 1), t_10)
    hop_dict.set_element((1, 0, 0), (2, 2), t_3)
    hop_dict.set_element((-1, 0, 0), (2, 2), t_3)
    hop_dict.set_element((0, 1, 0), (2, 2), t_7)
    hop_dict.set_element((0, -1, 0), (2, 2), t_7)
    hop_dict.set_element((1, 1, 0), (2, 2), t_10)
    hop_dict.set_element((-1, 1, 0), (2, 2), t_10)
    hop_dict.set_element((1, -1, 0), (2, 2), t_10)
    hop_dict.set_element((-1, -1, 0), (2, 2), t_10)
    hop_dict.set_element((0, 0, 0), (2, 3), t_2)
    hop_dict.set_element((0, -1, 0), (2, 3), t_6)
    hop_dict.set_element((-1, -1, 0), (2, 3), t_9)
    hop_dict.set_element((1, -1, 0), (2, 3), t_9)
    hop_dict.set_element((1, 0, 0), (3, 3), t_3)
    hop_dict.set_element((-1, 0, 0), (3, 3), t_3)
    hop_dict.set_element((0, 1, 0), (3, 3), t_7)
    hop_dict.set_element((0, -1, 0), (3, 3), t_7)
    hop_dict.set_element((1, 1, 0), (3, 3), t_10)
    hop_dict.set_element((-1, 1, 0), (3, 3), t_10)
    hop_dict.set_element((1, -1, 0), (3, 3), t_10)
    hop_dict.set_element((-1, -1, 0), (3, 3), t_10)

    # secondly, inter-plane
    hop_dict.set_element((-1, -1, 1), (0, 3), t_p5)
    hop_dict.set_element((0, 0, 1), (1, 0), t_p1)
    hop_dict.set_element((-1, 0, 1), (1, 0), t_p1)
    hop_dict.set_element((-1, 0, 1), (1, 3), t_p2)
    hop_dict.set_element((-1, -1, 1), (1, 3), t_p2)
    hop_dict.set_element((0, 0, 1), (1, 3), t_p3)
    hop_dict.set_element((-2, 0, 1), (1, 3), t_p3)
    hop_dict.set_element((-2, -1, 1), (1, 3), t_p3)
    hop_dict.set_element((0, -1, 1), (1, 3), t_p3)
    hop_dict.set_element((0, 1, 1), (1, 0), t_p4)
    hop_dict.set_element((-1, 1, 1), (1, 0), t_p4)
    hop_dict.set_element((-1, 0, 1), (1, 2), t_p5)
    hop_dict.set_element((0, 0, 1), (2, 3), t_p1)
    hop_dict.set_element((-1, 0, 1), (2, 3), t_p1)
    hop_dict.set_element((0, 0, 1), (2, 0), t_p2)
    hop_dict.set_element((0, 1, 1), (2, 0), t_p2)
    hop_dict.set_element((-1, 0, 1), (2, 0), t_p3)
    hop_dict.set_element((1, 0, 1), (2, 0), t_p3)
    hop_dict.set_element((-1, 1, 1), (2, 0), t_p3)
    hop_dict.set_element((1, 1, 1), (2, 0), t_p3)
    hop_dict.set_element((0, -1, 1), (2, 3), t_p4)
    hop_dict.set_element((-1, -1, 1), (2, 3), t_p4)
    hop_dict.set_element((0, 0, 1), (2, 1), t_p5)
    hop_dict.set_element((0, 0, 1), (3, 0), t_p5)

    return hop_dict


def sheet(W, H, n_layers=1):
    """Multilayer black phosphorus sheet.

    Parameters
    ----------
    W : integer
        width of SiteSet in unit cells
    H : integer
        height of SiteSet in unit cells
    n_layers : integer
        number of layers

    Returns
    ----------
    site_set : tipsi.SiteSet object
        rectangular black phosphorus sheet SiteSet
    """

    site_set = tipsi.SiteSet()
    for z in range(n_layers):
        for x in range(W):
            for y in range(H):
                for i in range(4):
                    site_set.add_site((x, y, z), i)
    return site_set


def pbc(W, H, unit_cell_coords, orbital):
    """PBC for a rectangular black phosphorus sheet.

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

    x, y, z = unit_cell_coords
    return (x % W, y % H, z), orbital


def sample(W=500, H=500, n_layers=1, nr_processes=1):
    """Multilayer rectangular black phosphorus sample.

    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
    n_layers : integer
        number of layers, optional (default 1)
    nr_processes : integer
        number of processes for sample building, optional (default 1)

    Returns
    ----------
    sample : tipsi.Sample object
        Rectangular black phosphorus sample.
    """

    # create lattice, hop_dict and pbc_wrap
    lat = lattice()
    hops = hop_dict()

    def pbc_wrap(unit_cell_coords, orbital):
        return pbc(W, H, unit_cell_coords, orbital)

    # create SiteSet object
    site_set = sheet(W, H)

    # make sample
    sample = tipsi.Sample(lat, site_set, pbc_wrap, nr_processes)

    # apply HopDict
    sample.add_hop_dict(hops)

    # rescale Hamiltonian
    sample.rescale_H(8.5)

    # done
    return sample
