"""Utilities for constructing antimonene primitive cells."""

import numpy as np

from ..base import cart2frac, NM
from ..builder import PrimitiveCell, HopDict


__all__ = ["make_antimonene"]


def make_antimonene(with_soc: bool = True,
                    soc_lambda: float = 0.34,
                    c: float = 1.0) -> PrimitiveCell:
    """
    Make antimonene primitive cell.

    Reference:
    https://journals.aps.org/prb/pdf/10.1103/PhysRevB.95.081407

    :param with_soc: whether to include spin-orbital coupling in
        constructing the model
    :param soc_lambda: strength of spin-orbital coupling
    :param c: length of lattice vector along c direction in NANOMETER
    :return: antimonene primitive cell
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
                        [0.0, 0.0, c]])

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
    if with_soc:
        orbital_labels = ["p11+", "p12+", "p13+", "p11-", "p12-", "p13-",
                          "p21+", "p22+", "p23+", "p21-", "p22-", "p23-"]
    else:
        orbital_labels = ["p11", "p12", "p13", "p21", "p22", "p23"]

    # Create primitive cell and add orbital
    # Since lattice vectors are already in nm, unit is set to 1.0.
    cell = PrimitiveCell(lat_vec=vectors, unit=NM)
    for i_o, coord in enumerate(orbital_coords):
        cell.add_orbital(coord, label=orbital_labels[i_o])

    # Build hop_dict
    if with_soc:
        hop_dict = HopDict(cell.num_orb // 2)
    else:
        hop_dict = HopDict(cell.num_orb)

    # 1st NEAREST NEIGHBOURS
    hop_dict[(0, -1, 0)][0, 3] = t_01
    hop_dict[(0, 0, 0)][1, 4] = t_01
    hop_dict[(-1, 0, 0)][2, 5] = t_01
    hop_dict[(0, 1, 0)][3, 0] = t_01
    hop_dict[(0, 0, 0)][4, 1] = t_01
    hop_dict[(1, 0, 0)][5, 2] = t_01

    hop_dict[(0, 0, 0)][0, 3] = t_02
    hop_dict[(-1, 0, 0)][0, 3] = t_02
    hop_dict[(-1, 0, 0)][1, 4] = t_02
    hop_dict[(0, -1, 0)][1, 4] = t_02
    hop_dict[(0, -1, 0)][2, 5] = t_02
    hop_dict[(0, 0, 0)][2, 5] = t_02
    hop_dict[(0, 0, 0)][3, 0] = t_02
    hop_dict[(1, 0, 0)][3, 0] = t_02
    hop_dict[(1, 0, 0)][4, 1] = t_02
    hop_dict[(0, 1, 0)][4, 1] = t_02
    hop_dict[(0, 1, 0)][5, 2] = t_02
    hop_dict[(0, 0, 0)][5, 2] = t_02

    hop_dict[(0, 0, 0)][0, 4] = t_07
    hop_dict[(-1, 0, 0)][0, 5] = t_07
    hop_dict[(-1, 0, 0)][1, 5] = t_07
    hop_dict[(0, -1, 0)][1, 3] = t_07
    hop_dict[(0, -1, 0)][2, 3] = t_07
    hop_dict[(0, 0, 0)][2, 4] = t_07
    hop_dict[(0, 0, 0)][3, 1] = t_07
    hop_dict[(1, 0, 0)][3, 2] = t_07
    hop_dict[(1, 0, 0)][4, 2] = t_07
    hop_dict[(0, 1, 0)][4, 0] = t_07
    hop_dict[(0, 1, 0)][5, 0] = t_07
    hop_dict[(0, 0, 0)][5, 1] = t_07

    # 2nd NEAREST NEIGHBOURS
    hop_dict[(-1, 1, 0)][0, 0] = t_03
    hop_dict[(0, 1, 0)][0, 0] = t_03
    hop_dict[(1, -1, 0)][0, 0] = t_03
    hop_dict[(0, -1, 0)][0, 0] = t_03
    hop_dict[(0, -1, 0)][1, 1] = t_03
    hop_dict[(-1, 0, 0)][1, 1] = t_03
    hop_dict[(0, 1, 0)][1, 1] = t_03
    hop_dict[(1, 0, 0)][1, 1] = t_03
    hop_dict[(-1, 0, 0)][2, 2] = t_03
    hop_dict[(-1, 1, 0)][2, 2] = t_03
    hop_dict[(1, 0, 0)][2, 2] = t_03
    hop_dict[(1, -1, 0)][2, 2] = t_03
    hop_dict[(-1, 1, 0)][3, 3] = t_03
    hop_dict[(0, 1, 0)][3, 3] = t_03
    hop_dict[(1, -1, 0)][3, 3] = t_03
    hop_dict[(0, -1, 0)][3, 3] = t_03
    hop_dict[(0, -1, 0)][4, 4] = t_03
    hop_dict[(-1, 0, 0)][4, 4] = t_03
    hop_dict[(0, 1, 0)][4, 4] = t_03
    hop_dict[(1, 0, 0)][4, 4] = t_03
    hop_dict[(-1, 0, 0)][5, 5] = t_03
    hop_dict[(-1, 1, 0)][5, 5] = t_03
    hop_dict[(1, 0, 0)][5, 5] = t_03
    hop_dict[(1, -1, 0)][5, 5] = t_03

    hop_dict[(0, 1, 0)][0, 1] = t_04
    hop_dict[(-1, 1, 0)][0, 2] = t_04
    hop_dict[(-1, 0, 0)][1, 2] = t_04
    hop_dict[(0, -1, 0)][1, 0] = t_04
    hop_dict[(1, -1, 0)][2, 0] = t_04
    hop_dict[(1, 0, 0)][2, 1] = t_04
    hop_dict[(0, -1, 0)][3, 4] = t_04
    hop_dict[(1, -1, 0)][3, 5] = t_04
    hop_dict[(1, 0, 0)][4, 5] = t_04
    hop_dict[(0, 1, 0)][4, 3] = t_04
    hop_dict[(-1, 1, 0)][5, 3] = t_04
    hop_dict[(-1, 0, 0)][5, 4] = t_04

    hop_dict[(0, -1, 0)][0, 1] = t_06
    hop_dict[(1, -1, 0)][0, 2] = t_06
    hop_dict[(1, 0, 0)][1, 2] = t_06
    hop_dict[(0, 1, 0)][1, 0] = t_06
    hop_dict[(-1, 1, 0)][2, 0] = t_06
    hop_dict[(-1, 0, 0)][2, 1] = t_06
    hop_dict[(0, 1, 0)][3, 4] = t_06
    hop_dict[(-1, 1, 0)][3, 5] = t_06
    hop_dict[(-1, 0, 0)][4, 5] = t_06
    hop_dict[(0, -1, 0)][4, 3] = t_06
    hop_dict[(1, -1, 0)][5, 3] = t_06
    hop_dict[(1, 0, 0)][5, 4] = t_06

    hop_dict[(-1, 0, 0)][0, 0] = t_11
    hop_dict[(1, 0, 0)][0, 0] = t_11
    hop_dict[(-1, 1, 0)][1, 1] = t_11
    hop_dict[(1, -1, 0)][1, 1] = t_11
    hop_dict[(0, -1, 0)][2, 2] = t_11
    hop_dict[(0, 1, 0)][2, 2] = t_11
    hop_dict[(-1, 0, 0)][3, 3] = t_11
    hop_dict[(1, 0, 0)][3, 3] = t_11
    hop_dict[(-1, 1, 0)][4, 4] = t_11
    hop_dict[(1, -1, 0)][4, 4] = t_11
    hop_dict[(0, -1, 0)][5, 5] = t_11
    hop_dict[(0, 1, 0)][5, 5] = t_11

    # 3rd NEAREST NEIGHBOURS
    # ACROSS THE HEXAGON
    hop_dict[(-1, 1, 0)][0, 4] = t_08
    hop_dict[(-1, 1, 0)][0, 5] = t_08
    hop_dict[(-1, -1, 0)][1, 5] = t_08
    hop_dict[(-1, -1, 0)][1, 3] = t_08
    hop_dict[(1, -1, 0)][2, 3] = t_08
    hop_dict[(1, -1, 0)][2, 4] = t_08
    hop_dict[(1, -1, 0)][3, 1] = t_08
    hop_dict[(1, -1, 0)][3, 2] = t_08
    hop_dict[(1, 1, 0)][4, 2] = t_08
    hop_dict[(1, 1, 0)][4, 0] = t_08
    hop_dict[(-1, 1, 0)][5, 0] = t_08
    hop_dict[(-1, 1, 0)][5, 1] = t_08

    hop_dict[(1, -1, 0)][0, 4] = t_12
    hop_dict[(-1, -1, 0)][0, 5] = t_12
    hop_dict[(-1, 1, 0)][1, 5] = t_12
    hop_dict[(1, -1, 0)][1, 3] = t_12
    hop_dict[(-1, -1, 0)][2, 3] = t_12
    hop_dict[(-1, 1, 0)][2, 4] = t_12
    hop_dict[(-1, 1, 0)][3, 1] = t_12
    hop_dict[(1, 1, 0)][3, 2] = t_12
    hop_dict[(1, -1, 0)][4, 2] = t_12
    hop_dict[(-1, 1, 0)][4, 0] = t_12
    hop_dict[(1, 1, 0)][5, 0] = t_12
    hop_dict[(1, -1, 0)][5, 1] = t_12

    # 3rd NEAREST NEIGHBOURS
    # ACROSS THE ZIGZAG
    hop_dict[(1, -2, 0)][0, 3] = t_05
    hop_dict[(0, -2, 0)][0, 3] = t_05
    hop_dict[(0, 1, 0)][1, 4] = t_05
    hop_dict[(1, 0, 0)][1, 4] = t_05
    hop_dict[(-2, 1, 0)][2, 5] = t_05
    hop_dict[(-2, 0, 0)][2, 5] = t_05
    hop_dict[(-1, 2, 0)][3, 0] = t_05
    hop_dict[(0, 2, 0)][3, 0] = t_05
    hop_dict[(-1, 0, 0)][4, 1] = t_05
    hop_dict[(0, -1, 0)][4, 1] = t_05
    hop_dict[(2, -1, 0)][5, 2] = t_05
    hop_dict[(2, 0, 0)][5, 2] = t_05

    hop_dict[(-2, 1, 0)][0, 3] = t_09
    hop_dict[(0, 1, 0)][0, 3] = t_09
    hop_dict[(0, -2, 0)][1, 4] = t_09
    hop_dict[(-2, 0, 0)][1, 4] = t_09
    hop_dict[(1, 0, 0)][2, 5] = t_09
    hop_dict[(1, -2, 0)][2, 5] = t_09
    hop_dict[(2, -1, 0)][3, 0] = t_09
    hop_dict[(0, -1, 0)][3, 0] = t_09
    hop_dict[(0, 2, 0)][4, 1] = t_09
    hop_dict[(2, 0, 0)][4, 1] = t_09
    hop_dict[(-1, 2, 0)][5, 2] = t_09
    hop_dict[(-1, 0, 0)][5, 2] = t_09

    hop_dict[(0, 1, 0)][0, 4] = t_10
    hop_dict[(-2, 1, 0)][0, 5] = t_10
    hop_dict[(-2, 0, 0)][1, 5] = t_10
    hop_dict[(0, -2, 0)][1, 3] = t_10
    hop_dict[(1, -2, 0)][2, 3] = t_10
    hop_dict[(1, 0, 0)][2, 4] = t_10
    hop_dict[(0, -1, 0)][3, 1] = t_10
    hop_dict[(2, -1, 0)][3, 2] = t_10
    hop_dict[(2, 0, 0)][4, 2] = t_10
    hop_dict[(0, 2, 0)][4, 0] = t_10
    hop_dict[(-1, 2, 0)][5, 0] = t_10
    hop_dict[(-1, 0, 0)][5, 1] = t_10

    hop_dict[(1, 0, 0)][0, 3] = t_13
    hop_dict[(-2, 0, 0)][0, 3] = t_13
    hop_dict[(-2, 1, 0)][1, 4] = t_13
    hop_dict[(1, -2, 0)][1, 4] = t_13
    hop_dict[(0, -2, 0)][2, 5] = t_13
    hop_dict[(0, 1, 0)][2, 5] = t_13
    hop_dict[(-1, 0, 0)][3, 0] = t_13
    hop_dict[(2, 0, 0)][3, 0] = t_13
    hop_dict[(-1, 2, 0)][4, 1] = t_13
    hop_dict[(2, -1, 0)][4, 1] = t_13
    hop_dict[(0, 2, 0)][5, 2] = t_13
    hop_dict[(0, -1, 0)][5, 2] = t_13

    # 4th NEAREST NEIGHBOURS
    # ACROSS THE ZIGZAG
    hop_dict[(0, -2, 0)][0, 1] = t_14
    hop_dict[(2, -2, 0)][0, 2] = t_14
    hop_dict[(2, 0, 0)][1, 2] = t_14
    hop_dict[(0, 2, 0)][1, 0] = t_14
    hop_dict[(-2, 2, 0)][2, 0] = t_14
    hop_dict[(-2, 0, 0)][2, 1] = t_14
    hop_dict[(0, 2, 0)][3, 4] = t_14
    hop_dict[(-2, 2, 0)][3, 5] = t_14
    hop_dict[(-2, 0, 0)][4, 5] = t_14
    hop_dict[(0, -2, 0)][4, 3] = t_14
    hop_dict[(2, -2, 0)][5, 3] = t_14
    hop_dict[(2, 0, 0)][5, 4] = t_14

    hop_dict[(0, -2, 0)][1, 0] = t_15
    hop_dict[(2, -2, 0)][2, 0] = t_15
    hop_dict[(2, 0, 0)][2, 1] = t_15
    hop_dict[(0, 2, 0)][0, 1] = t_15
    hop_dict[(-2, 2, 0)][0, 2] = t_15
    hop_dict[(-2, 0, 0)][1, 2] = t_15
    hop_dict[(0, 2, 0)][4, 3] = t_15
    hop_dict[(-2, 2, 0)][5, 3] = t_15
    hop_dict[(-2, 0, 0)][5, 4] = t_15
    hop_dict[(0, -2, 0)][3, 4] = t_15
    hop_dict[(2, -2, 0)][3, 5] = t_15
    hop_dict[(2, 0, 0)][4, 5] = t_15

    # Deal with SOC
    if with_soc:
        hop_dict.num_orb = cell.num_orb
        for rel_unit_cell, hop in hop_dict.hoppings.items():
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
            hop_dict[rel_unit_cell] = new_hop
        hop_dict[(0, 0, 0)] += np.kron(np.eye(2), soc_mat)

    # Apply hop_dict
    cell.add_hopping_dict(hop_dict)
    return cell
