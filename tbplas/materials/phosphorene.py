"""
Utilities for constructing phosphorene samples.

Functions
---------
    make_black_phosphorus: user function
        make black phosphorus primitive cell
"""

import numpy as np

import tbplas
from ..builder import PrimitiveCell, cart2frac


def make_black_phosphorus(c=10.0):
    """
    Make black phosphorus primitive cell.

    Reference:
    https://journals.aps.org/prb/pdf/10.1103/PhysRevB.92.085419

    :param c: float
        length of c-axis in ANGSTROM
    :return: cell: instance of 'PrimitiveCell' class
        black phosphorus primitive cell
    """
    # Geometric constants
    # Lengths are in nm and angles are in degree.
    dist_nn = 0.22156
    dist_nnb = 0.07159
    theta = 48.395
    theta_z = 108.657
    dist_nz = 0.22378

    # Hopping terms in eV
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

    # Calculate lattice vectors
    a = 2 * dist_nn * np.sin(np.radians(theta))
    b = 2 * dist_nnb + 2 * dist_nn * np.cos(np.radians(theta))
    p = dist_nnb
    q = dist_nz * np.cos(np.radians(theta_z - 90))
    vectors = np.array([[a, 0., 0.], [0., b, 0.], [0., 0., c*tbplas.ANG]])

    # Calculate orbital coordinates
    orbital_coords = np.zeros((4, 3))
    orbital_coords[0] = [0., 0., 0.]
    orbital_coords[1] = [0., p, q]
    orbital_coords[2] = [a / 2., b / 2., q]
    orbital_coords[3] = [a / 2., b / 2. + p, 0.]
    orbital_coords = cart2frac(vectors, orbital_coords)

    # Create primitive cell and add orbital
    # Since lattice vectors are already in nm, unit is set to 1.0.
    cell = PrimitiveCell(lat_vec=vectors, unit=1.0)
    for coord in orbital_coords:
        cell.add_orbital(coord)

    # Add hopping terms
    cell.add_hopping((0, 0), 0, 1, t_2)
    cell.add_hopping((0, -1), 0, 3, t_1)
    cell.add_hopping((-1, -1), 0, 3, t_1)
    cell.add_hopping((0, 0), 0, 2, t_5)
    cell.add_hopping((-1, 0), 0, 2, t_5)
    cell.add_hopping((0, -1), 0, 2, t_5)
    cell.add_hopping((-1, -1), 0, 2, t_5)
    cell.add_hopping((0, 0), 0, 3, t_4)
    cell.add_hopping((-1, 0), 0, 3, t_4)
    cell.add_hopping((0, -1), 0, 1, t_6)
    cell.add_hopping((1, 0), 0, 0, t_3)
    cell.add_hopping((0, 1), 0, 0, t_7)
    cell.add_hopping((-2, -1), 0, 3, t_8)
    cell.add_hopping((1, -1), 0, 3, t_8)
    cell.add_hopping((-1, -1), 0, 1, t_9)
    cell.add_hopping((1, -1), 0, 1, t_9)
    cell.add_hopping((1, 1), 0, 0, t_10)
    cell.add_hopping((-1, 1), 0, 0, t_10)
    cell.add_hopping((0, 0), 1, 2, t_1)
    cell.add_hopping((-1, 0), 1, 2, t_1)
    cell.add_hopping((-1, -1), 1, 3, t_5)
    cell.add_hopping((0, -1), 1, 3, t_5)
    cell.add_hopping((0, 0), 1, 3, t_5)
    cell.add_hopping((-1, 0), 1, 3, t_5)
    cell.add_hopping((-1, -1), 1, 2, t_4)
    cell.add_hopping((0, -1), 1, 2, t_4)
    cell.add_hopping((1, 0), 1, 1, t_3)
    cell.add_hopping((0, 1), 1, 1, t_7)
    cell.add_hopping((-2, 0), 1, 2, t_8)
    cell.add_hopping((1, 0), 1, 2, t_8)
    cell.add_hopping((1, 1), 1, 1, t_10)
    cell.add_hopping((-1, 1), 1, 1, t_10)
    cell.add_hopping((1, 0), 2, 2, t_3)
    cell.add_hopping((0, 1), 2, 2, t_7)
    cell.add_hopping((1, 1), 2, 2, t_10)
    cell.add_hopping((-1, 1), 2, 2, t_10)
    cell.add_hopping((0, 0), 2, 3, t_2)
    cell.add_hopping((0, -1), 2, 3, t_6)
    cell.add_hopping((-1, -1), 2, 3, t_9)
    cell.add_hopping((1, -1), 2, 3, t_9)
    cell.add_hopping((1, 0), 3, 3, t_3)
    cell.add_hopping((0, 1), 3, 3, t_7)
    cell.add_hopping((1, 1), 3, 3, t_10)
    cell.add_hopping((-1, 1), 3, 3, t_10)
    return cell
