"""
Utilities for constructing graphene and silicene samples.

Functions
---------
    make_graphene_diamond: user API
        make graphene primitive cell in diamond shape
    make_graphene_rect: user API
        make graphene primitive cell in rectangular shape
"""

import numpy as np

from ..builder import PrimitiveCell, cart2frac, HopDict


def make_black_phosphorus():
    """
    Make black phosphorus primitive cell.

    Reference:
    https://journals.aps.org/prb/pdf/10.1103/PhysRevB.92.085419

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
    vectors = np.array([[a, 0., 0.], [0., b, 0.], [0., 0., 1.]])

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


def make_antimonene(with_soc=True, soc_lambda=0.34):
    """
    Make antimonene primitive cell.

    Reference:
    https://journals.aps.org/prb/pdf/10.1103/PhysRevB.95.081407

    :param with_soc: boolean
        whether to include spin-orbital coupling in constructing the model
    :param soc_lambda: float
        strength of spin-orbital coupling
    :return: cell: instance of 'PrimitiveCell' class
        antimonene primitive cell
    """
    # Geometric constants
    a = 0.411975806
    z = 0.16455347

    # Hopping terms
    t_01 = -2.09
    t_02 = 0.47
    t_03 = 0.18
    t_04 = -0.50
    t_05 = -0.11
    t_06 = 0.21
    t_07 = 0.08
    t_08 = -0.07
    t_09 = 0.07
    t_10 = 0.07
    t_11 = -0.06
    t_12 = -0.06
    t_13 = -0.03
    t_14 = -0.03
    t_15 = -0.04

    # SOC coupling matrix
    soc_mat = 0.5 * soc_lambda * np.array([
        [0.00000 + 0.0000j, 0.0000 + 0.5854j, 0.0000 - 0.5854j,
         -0.0000 + 0.0000j, 0.7020 - 0.4053j, -0.0000 - 0.8107j],
        [0.00000 - 0.5854j, 0.0000 + 0.0000j, 0.0000 + 0.5854j,
         -0.7020 + 0.4053j, 0.0000 + 0.0000j, -0.7020 - 0.4053j],
        [0.00000 + 0.5854j, 0.0000 - 0.5854j, 0.0000 + 0.0000j,
         -0.0000 + 0.8107j, 0.7020 + 0.4053j, -0.0000 + 0.0000j],
        [0.00000 + 0.0000j, -0.7020 - 0.4053j, 0.0000 - 0.8107j,
         0.00000 + 0.0000j, 0.0000 - 0.5854j, 0.0000 + 0.5854j],
        [0.70200 + 0.4053j, -0.0000 + 0.0000j, 0.7020 - 0.4053j,
         0.00000 + 0.5854j, 0.0000 + 0.0000j, 0.0000 - 0.5854j],
        [0.00000 + 0.8107j, -0.7020 + 0.4053j, 0.0000 + 0.0000j,
         0.00000 - 0.5854j, 0.0000 + 0.5854j, 0.0000 + 0.0000j]
    ])

    # Calculate lattice vectors
    b = a / np.sqrt(3.)
    vectors = np.array([[1.5 * b, -0.5 * a, 0.],
                        [1.5 * b, 0.5 * a, 0.],
                        [0.0, 0.0, 1.0]])

    # Calculate orbital coordinates
    num_orb_per_site = 3
    if with_soc:
        num_orb_per_site *= 2
    coord_site_1 = [-b / 2., 0., -z / 2.]
    coord_site_2 = [b / 2., 0., z / 2.]
    orbital_coords = [coord_site_1 for _ in range(num_orb_per_site)]
    orbital_coords.extend([coord_site_2 for _ in range(num_orb_per_site)])
    orbital_coords = np.array(orbital_coords)
    orbital_coords = cart2frac(vectors, orbital_coords)

    # Create primitive cell and add orbital
    # Since lattice vectors are already in nm, unit is set to 1.0.
    cell = PrimitiveCell(lat_vec=vectors, unit=1.0)
    for coord in orbital_coords:
        cell.add_orbital(coord)

    # Build hop_dict
    if with_soc:
        hop_dict = HopDict(cell.num_orb // 2)
    else:
        hop_dict = HopDict(cell.num_orb)

    # 1st NEAREST NEIGHBOURS
    hop_dict.set_element((0, -1, 0), (0, 3), t_01)
    hop_dict.set_element((0, 0, 0), (1, 4), t_01)
    hop_dict.set_element((-1, 0, 0), (2, 5), t_01)
    hop_dict.set_element((0, 1, 0), (3, 0), t_01)
    hop_dict.set_element((0, 0, 0), (4, 1), t_01)
    hop_dict.set_element((1, 0, 0), (5, 2), t_01)

    hop_dict.set_element((0, 0, 0), (0, 3), t_02)
    hop_dict.set_element((-1, 0, 0), (0, 3), t_02)
    hop_dict.set_element((-1, 0, 0), (1, 4), t_02)
    hop_dict.set_element((0, -1, 0), (1, 4), t_02)
    hop_dict.set_element((0, -1, 0), (2, 5), t_02)
    hop_dict.set_element((0, 0, 0), (2, 5), t_02)
    hop_dict.set_element((0, 0, 0), (3, 0), t_02)
    hop_dict.set_element((1, 0, 0), (3, 0), t_02)
    hop_dict.set_element((1, 0, 0), (4, 1), t_02)
    hop_dict.set_element((0, 1, 0), (4, 1), t_02)
    hop_dict.set_element((0, 1, 0), (5, 2), t_02)
    hop_dict.set_element((0, 0, 0), (5, 2), t_02)

    hop_dict.set_element((0, 0, 0), (0, 4), t_07)
    hop_dict.set_element((-1, 0, 0), (0, 5), t_07)
    hop_dict.set_element((-1, 0, 0), (1, 5), t_07)
    hop_dict.set_element((0, -1, 0), (1, 3), t_07)
    hop_dict.set_element((0, -1, 0), (2, 3), t_07)
    hop_dict.set_element((0, 0, 0), (2, 4), t_07)
    hop_dict.set_element((0, 0, 0), (3, 1), t_07)
    hop_dict.set_element((1, 0, 0), (3, 2), t_07)
    hop_dict.set_element((1, 0, 0), (4, 2), t_07)
    hop_dict.set_element((0, 1, 0), (4, 0), t_07)
    hop_dict.set_element((0, 1, 0), (5, 0), t_07)
    hop_dict.set_element((0, 0, 0), (5, 1), t_07)

    # 2nd NEAREST NEIGHBOURS
    hop_dict.set_element((-1, 1, 0), (0, 0), t_03)
    hop_dict.set_element((0, 1, 0), (0, 0), t_03)
    hop_dict.set_element((1, -1, 0), (0, 0), t_03)
    hop_dict.set_element((0, -1, 0), (0, 0), t_03)
    hop_dict.set_element((0, -1, 0), (1, 1), t_03)
    hop_dict.set_element((-1, 0, 0), (1, 1), t_03)
    hop_dict.set_element((0, 1, 0), (1, 1), t_03)
    hop_dict.set_element((1, 0, 0), (1, 1), t_03)
    hop_dict.set_element((-1, 0, 0), (2, 2), t_03)
    hop_dict.set_element((-1, 1, 0), (2, 2), t_03)
    hop_dict.set_element((1, 0, 0), (2, 2), t_03)
    hop_dict.set_element((1, -1, 0), (2, 2), t_03)
    hop_dict.set_element((-1, 1, 0), (3, 3), t_03)
    hop_dict.set_element((0, 1, 0), (3, 3), t_03)
    hop_dict.set_element((1, -1, 0), (3, 3), t_03)
    hop_dict.set_element((0, -1, 0), (3, 3), t_03)
    hop_dict.set_element((0, -1, 0), (4, 4), t_03)
    hop_dict.set_element((-1, 0, 0), (4, 4), t_03)
    hop_dict.set_element((0, 1, 0), (4, 4), t_03)
    hop_dict.set_element((1, 0, 0), (4, 4), t_03)
    hop_dict.set_element((-1, 0, 0), (5, 5), t_03)
    hop_dict.set_element((-1, 1, 0), (5, 5), t_03)
    hop_dict.set_element((1, 0, 0), (5, 5), t_03)
    hop_dict.set_element((1, -1, 0), (5, 5), t_03)

    hop_dict.set_element((0, 1, 0), (0, 1), t_04)
    hop_dict.set_element((-1, 1, 0), (0, 2), t_04)
    hop_dict.set_element((-1, 0, 0), (1, 2), t_04)
    hop_dict.set_element((0, -1, 0), (1, 0), t_04)
    hop_dict.set_element((1, -1, 0), (2, 0), t_04)
    hop_dict.set_element((1, 0, 0), (2, 1), t_04)
    hop_dict.set_element((0, -1, 0), (3, 4), t_04)
    hop_dict.set_element((1, -1, 0), (3, 5), t_04)
    hop_dict.set_element((1, 0, 0), (4, 5), t_04)
    hop_dict.set_element((0, 1, 0), (4, 3), t_04)
    hop_dict.set_element((-1, 1, 0), (5, 3), t_04)
    hop_dict.set_element((-1, 0, 0), (5, 4), t_04)

    hop_dict.set_element((0, -1, 0), (0, 1), t_06)
    hop_dict.set_element((1, -1, 0), (0, 2), t_06)
    hop_dict.set_element((1, 0, 0), (1, 2), t_06)
    hop_dict.set_element((0, 1, 0), (1, 0), t_06)
    hop_dict.set_element((-1, 1, 0), (2, 0), t_06)
    hop_dict.set_element((-1, 0, 0), (2, 1), t_06)
    hop_dict.set_element((0, 1, 0), (3, 4), t_06)
    hop_dict.set_element((-1, 1, 0), (3, 5), t_06)
    hop_dict.set_element((-1, 0, 0), (4, 5), t_06)
    hop_dict.set_element((0, -1, 0), (4, 3), t_06)
    hop_dict.set_element((1, -1, 0), (5, 3), t_06)
    hop_dict.set_element((1, 0, 0), (5, 4), t_06)

    hop_dict.set_element((-1, 0, 0), (0, 0), t_11)
    hop_dict.set_element((1, 0, 0), (0, 0), t_11)
    hop_dict.set_element((-1, 1, 0), (1, 1), t_11)
    hop_dict.set_element((1, -1, 0), (1, 1), t_11)
    hop_dict.set_element((0, -1, 0), (2, 2), t_11)
    hop_dict.set_element((0, 1, 0), (2, 2), t_11)
    hop_dict.set_element((-1, 0, 0), (3, 3), t_11)
    hop_dict.set_element((1, 0, 0), (3, 3), t_11)
    hop_dict.set_element((-1, 1, 0), (4, 4), t_11)
    hop_dict.set_element((1, -1, 0), (4, 4), t_11)
    hop_dict.set_element((0, -1, 0), (5, 5), t_11)
    hop_dict.set_element((0, 1, 0), (5, 5), t_11)

    # 3rd NEAREST NEIGHBOURS
    # ACROSS THE HEXAGON
    hop_dict.set_element((-1, 1, 0), (0, 4), t_08)
    hop_dict.set_element((-1, 1, 0), (0, 5), t_08)
    hop_dict.set_element((-1, -1, 0), (1, 5), t_08)
    hop_dict.set_element((-1, -1, 0), (1, 3), t_08)
    hop_dict.set_element((1, -1, 0), (2, 3), t_08)
    hop_dict.set_element((1, -1, 0), (2, 4), t_08)
    hop_dict.set_element((1, -1, 0), (3, 1), t_08)
    hop_dict.set_element((1, -1, 0), (3, 2), t_08)
    hop_dict.set_element((1, 1, 0), (4, 2), t_08)
    hop_dict.set_element((1, 1, 0), (4, 0), t_08)
    hop_dict.set_element((-1, 1, 0), (5, 0), t_08)
    hop_dict.set_element((-1, 1, 0), (5, 1), t_08)

    hop_dict.set_element((1, -1, 0), (0, 4), t_12)
    hop_dict.set_element((-1, -1, 0), (0, 5), t_12)
    hop_dict.set_element((-1, 1, 0), (1, 5), t_12)
    hop_dict.set_element((1, -1, 0), (1, 3), t_12)
    hop_dict.set_element((-1, -1, 0), (2, 3), t_12)
    hop_dict.set_element((-1, 1, 0), (2, 4), t_12)
    hop_dict.set_element((-1, 1, 0), (3, 1), t_12)
    hop_dict.set_element((1, 1, 0), (3, 2), t_12)
    hop_dict.set_element((1, -1, 0), (4, 2), t_12)
    hop_dict.set_element((-1, 1, 0), (4, 0), t_12)
    hop_dict.set_element((1, 1, 0), (5, 0), t_12)
    hop_dict.set_element((1, -1, 0), (5, 1), t_12)

    # 3rd NEAREST NEIGHBOURS
    # ACROSS THE ZIGZAG
    hop_dict.set_element((1, -2, 0), (0, 3), t_05)
    hop_dict.set_element((0, -2, 0), (0, 3), t_05)
    hop_dict.set_element((0, 1, 0), (1, 4), t_05)
    hop_dict.set_element((1, 0, 0), (1, 4), t_05)
    hop_dict.set_element((-2, 1, 0), (2, 5), t_05)
    hop_dict.set_element((-2, 0, 0), (2, 5), t_05)
    hop_dict.set_element((-1, 2, 0), (3, 0), t_05)
    hop_dict.set_element((0, 2, 0), (3, 0), t_05)
    hop_dict.set_element((-1, 0, 0), (4, 1), t_05)
    hop_dict.set_element((0, -1, 0), (4, 1), t_05)
    hop_dict.set_element((2, -1, 0), (5, 2), t_05)
    hop_dict.set_element((2, 0, 0), (5, 2), t_05)

    hop_dict.set_element((-2, 1, 0), (0, 3), t_09)
    hop_dict.set_element((0, 1, 0), (0, 3), t_09)
    hop_dict.set_element((0, -2, 0), (1, 4), t_09)
    hop_dict.set_element((-2, 0, 0), (1, 4), t_09)
    hop_dict.set_element((1, 0, 0), (2, 5), t_09)
    hop_dict.set_element((1, -2, 0), (2, 5), t_09)
    hop_dict.set_element((2, -1, 0), (3, 0), t_09)
    hop_dict.set_element((0, -1, 0), (3, 0), t_09)
    hop_dict.set_element((0, 2, 0), (4, 1), t_09)
    hop_dict.set_element((2, 0, 0), (4, 1), t_09)
    hop_dict.set_element((-1, 2, 0), (5, 2), t_09)
    hop_dict.set_element((-1, 0, 0), (5, 2), t_09)

    hop_dict.set_element((0, 1, 0), (0, 4), t_10)
    hop_dict.set_element((-2, 1, 0), (0, 5), t_10)
    hop_dict.set_element((-2, 0, 0), (1, 5), t_10)
    hop_dict.set_element((0, -2, 0), (1, 3), t_10)
    hop_dict.set_element((1, -2, 0), (2, 3), t_10)
    hop_dict.set_element((1, 0, 0), (2, 4), t_10)
    hop_dict.set_element((0, -1, 0), (3, 1), t_10)
    hop_dict.set_element((2, -1, 0), (3, 2), t_10)
    hop_dict.set_element((2, 0, 0), (4, 2), t_10)
    hop_dict.set_element((0, 2, 0), (4, 0), t_10)
    hop_dict.set_element((-1, 2, 0), (5, 0), t_10)
    hop_dict.set_element((-1, 0, 0), (5, 1), t_10)

    hop_dict.set_element((1, 0, 0), (0, 3), t_13)
    hop_dict.set_element((-2, 0, 0), (0, 3), t_13)
    hop_dict.set_element((-2, 1, 0), (1, 4), t_13)
    hop_dict.set_element((1, -2, 0), (1, 4), t_13)
    hop_dict.set_element((0, -2, 0), (2, 5), t_13)
    hop_dict.set_element((0, 1, 0), (2, 5), t_13)
    hop_dict.set_element((-1, 0, 0), (3, 0), t_13)
    hop_dict.set_element((2, 0, 0), (3, 0), t_13)
    hop_dict.set_element((-1, 2, 0), (4, 1), t_13)
    hop_dict.set_element((2, -1, 0), (4, 1), t_13)
    hop_dict.set_element((0, 2, 0), (5, 2), t_13)
    hop_dict.set_element((0, -1, 0), (5, 2), t_13)

    # 4th NEAREST NEIGHBOURS
    # ACROSS THE ZIGZAG
    hop_dict.set_element((0, -2, 0), (0, 1), t_14)
    hop_dict.set_element((2, -2, 0), (0, 2), t_14)
    hop_dict.set_element((2, 0, 0), (1, 2), t_14)
    hop_dict.set_element((0, 2, 0), (1, 0), t_14)
    hop_dict.set_element((-2, 2, 0), (2, 0), t_14)
    hop_dict.set_element((-2, 0, 0), (2, 1), t_14)
    hop_dict.set_element((0, 2, 0), (3, 4), t_14)
    hop_dict.set_element((-2, 2, 0), (3, 5), t_14)
    hop_dict.set_element((-2, 0, 0), (4, 5), t_14)
    hop_dict.set_element((0, -2, 0), (4, 3), t_14)
    hop_dict.set_element((2, -2, 0), (5, 3), t_14)
    hop_dict.set_element((2, 0, 0), (5, 4), t_14)

    hop_dict.set_element((0, -2, 0), (1, 0), t_15)
    hop_dict.set_element((2, -2, 0), (2, 0), t_15)
    hop_dict.set_element((2, 0, 0), (2, 1), t_15)
    hop_dict.set_element((0, 2, 0), (0, 1), t_15)
    hop_dict.set_element((-2, 2, 0), (0, 2), t_15)
    hop_dict.set_element((-2, 0, 0), (1, 2), t_15)
    hop_dict.set_element((0, 2, 0), (4, 3), t_15)
    hop_dict.set_element((-2, 2, 0), (5, 3), t_15)
    hop_dict.set_element((-2, 0, 0), (5, 4), t_15)
    hop_dict.set_element((0, -2, 0), (3, 4), t_15)
    hop_dict.set_element((2, -2, 0), (3, 5), t_15)
    hop_dict.set_element((2, 0, 0), (4, 5), t_15)

    # Deal with SOC
    if with_soc:
        hop_dict.set_num_orb(cell.num_orb)
        for rel_unit_cell, hop in hop_dict.dict.items():
            new_hop = np.zeros((12, 12), dtype="complex")
            hop00 = hop[0:3, 0:3]
            hop01 = hop[0:3, 3:6]
            hop10 = hop[3:6, 0:3]
            hop11 = hop[3:6, 3:6]
            new_hop[0:3, 0:3] = hop00
            new_hop[3:6, 3:6] = hop00
            new_hop[0:3, 6:9] = hop01
            new_hop[3:6, 9:12] = hop01
            new_hop[6:9, 0:3] = hop10
            new_hop[9:12, 3:6] = hop10
            new_hop[6:9, 6:9] = hop11
            new_hop[9:12, 9:12] = hop11
            hop_dict.set_mat(rel_unit_cell, new_hop)
        hop_dict.dict[(0, 0, 0)] += np.kron(np.eye(2), soc_mat)

    # Apply hop_dict
    cell.add_hopping_dict(hop_dict)
    return cell
