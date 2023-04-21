"""
Utilities for constructing TMDC primitive cells.

Reference:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.205108

NOTE: We use unsymmetrical orbitals in constructing the model.
"""

import math
from typing import Tuple, List

import numpy as np

from ..builder import PrimitiveCell, HopDict, cart2frac, NM


__all__ = ["make_tmdc"]


_STRUCT_CONSTS = {
    'MoS2': (0.316, 0.317, 1.229 / 2),
    'MoSe2': (0.329, 0.333, 1.290 / 2),
    'WS2': (0.315, 0.314, 1.232 / 2),
    'WSe2': (0.328, 0.334, 1.296 / 2)
}

_HOP_CONSTS = {
    'MoS2': (1.0688, 1.0688, -0.7755, -1.2902, -1.2902,
             -0.1380, 0.0874, 0.0874, -2.8949, -1.9065,
             -1.9065, -0.2069, 0.0323, -0.1739, 0.8651,
             -0.1872, -0.2979, 0.2747, -0.5581, -0.1916,
             0.9122, 0.0059, -0.0679, 0.4096, 0.0075,
             -0.2562, -0.0995, -0.0705, -0.1145, -0.2487,
             0.1063, -0.0385, -0.7883, -1.3790, 2.1584,
             -0.8836, -0.9402, 1.4114, -0.9535, 0.6517,
             -0.0686, -0.1498, -0.2205, -0.2451),
    'MoSe2': (0.7819, 0.7819, -0.6567, -1.1726, -1.1726,
              -0.2297, 0.0149, 0.0149, -2.9015, -1.7806,
              -1.7806, -0.1460, 0.0177, -0.2112, 0.9638,
              -0.1724, -0.2636, 0.2505, -0.4734, -0.2166,
              0.9911, -0.0036, -0.0735, 0.3520, 0.0047,
              -0.1912, -0.0755, -0.0680, -0.0960, -0.2012,
              0.1216, -0.0394, -0.6946, -1.3258, 1.9415,
              -0.7720, -0.8738, 1.2677, -0.8578, 0.5545,
              -0.0691, -0.1553, -0.2227, -0.2154),
    'WS2': (1.3754, 1.3754, -1.1278, -1.5534, -1.5534,
            -0.0393, 0.1984, 0.1984, -3.3706, -2.3461,
            -2.3461, -0.2011, 0.0263, -0.1749, 0.8726,
            -0.2187, -0.3716, 0.3537, -0.6892, -0.2112,
            0.9673, 0.0143, -0.0818, 0.4896, -0.0315,
            -0.3106, -0.1105, -0.0989, -0.1467, -0.3030,
            0.1645, -0.1018, -0.8855, -1.4376, 2.3121,
            -1.0130, -0.9878, 1.5629, -0.9491, 0.6718,
            -0.0659, -0.1533, -0.2618, -0.2736),
    'WSe2': (1.0349, 1.0349, -0.9573, -1.3937, -1.3937,
             -0.1667, 0.0984, 0.0984, -3.3642, -2.1820,
             -2.1820, -0.1395, 0.0129, -0.2171, 0.9763,
             -0.1985, -0.3330, 0.3190, -0.5837, -0.2399,
             1.0470, 0.0029, -0.0912, 0.4233, -0.0377,
             -0.2321, -0.0797, -0.0920, -0.1250, -0.2456,
             0.1857, -0.1027, -0.7744, -1.4014, 2.0858,
             -0.8998, -0.9044, 1.4030, -0.8548, 0.5711,
             -0.0676, -0.1608, -0.2618, -0.2424)}


def _gen_lattice(material: str = "MoS2",
                 c: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate coordinates of lattice vectors and orbitals.

    :param material: chemical label of material
        Should be in ("MoS2", "MoSe2", "WS2", "WSe2")
    :param c: length of c-axis in NANOMETER
    :return: (vectors, orbital_coordinates)
        vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors in NM
        orbital_coordinates: (11, 3) float64 array
        Cartesian coordinates of orbitals in NM
    :raises NotImplementedError: if material has not been implemented
    """
    # a: Mo-Mo distance in nm
    # d_xx: closest distance between S and S in the Z direction
    # z: interlayer distance in nm
    try:
        a, d_xx, z = _STRUCT_CONSTS[material]
    except KeyError as err:
        raise NotImplementedError(f"{material} not implemented yet") from err

    sqrt_3 = math.sqrt(3)
    vectors = np.array([[a / 2, a * sqrt_3 / 2, 0],
                        [-a / 2, a * sqrt_3 / 2, 0],
                        [0, 0, c]])
    coord_mo = [0, a / sqrt_3, d_xx / 2]
    coord_s1 = [0, 0, d_xx]
    coord_s2 = [0, 0, 0]
    orbital_coords = [coord_mo for _ in range(5)]
    orbital_coords.extend([coord_s1 for _ in range(3)])
    orbital_coords.extend([coord_s2 for _ in range(3)])
    orbital_coords = np.array(orbital_coords)
    return vectors, orbital_coords


def _gen_hop_dict(material: str = "MoS2") -> Tuple[HopDict, List[float]]:
    """
    Generate hopping dictionary and on-site energies.

    :param material: chemical label of material
        Should be in ("MoS2", "MoSe2", "WS2", "WSe2")
    :return: (hop_dict, on_site)
        hop_dict: TMDC hopping dictionary
        on_site: on-site energies of orbitals
    :raises NotImplementedError: if material has not been implemented
    """
    # Mathematical constants
    sqrt_3 = math.sqrt(3)
    inv_sqrt_2 = 1 / math.sqrt(2)

    # Initialize hopping matrices 
    hop_mat1 = np.zeros((11, 11))
    hop_mat2 = np.zeros((11, 11))
    hop_mat3 = np.zeros((11, 11))
    hop_mat4 = np.zeros((11, 11))
    hop_mat5 = np.zeros((11, 11))
    hop_mat6 = np.zeros((11, 11))
    try:
        hop_data = _HOP_CONSTS[material]
    except KeyError as err:
        raise NotImplementedError(f"{material} not implemented yet") from err
    else:
        epsilon = hop_data[:11]
        hop_mat1[0, 0], hop_mat1[1, 1], hop_mat1[2, 2], hop_mat1[3, 3], \
            hop_mat1[4, 4], hop_mat1[5, 5], hop_mat1[6, 6], hop_mat1[7, 7], \
            hop_mat1[8, 8], hop_mat1[9, 9], hop_mat1[10, 10], hop_mat1[2, 4], \
            hop_mat1[5, 7], hop_mat1[8, 10], hop_mat1[0, 1], hop_mat1[2, 3], \
            hop_mat1[3, 4], hop_mat1[5, 6], hop_mat1[6, 7], hop_mat1[8, 9], \
            hop_mat1[9, 10], hop_mat5[3, 0], hop_mat5[2, 1], hop_mat5[4, 1], \
            hop_mat5[8, 5], hop_mat5[10, 5], hop_mat5[9, 6], hop_mat5[8, 7], \
            hop_mat5[10, 7], hop_mat6[8, 5], hop_mat6[10, 5], hop_mat6[8, 7], \
            hop_mat6[10, 7] = hop_data[11:]

    # Indices used in deriving the tight binding model in the literature
    list0 = [[3, 4, 2], [6, 7, 5], [9, 10, 8]]
    list00 = [[0, 1, 3, 4, 2], [6, 7, 9, 10, 8]]

    for m1 in list0:
        hop_mat2[m1[0], m1[0]] = \
            hop_mat1[m1[0], m1[0]] / 4 + 3 * hop_mat1[m1[1], m1[1]] / 4
        hop_mat2[m1[1], m1[1]] = \
            3 * hop_mat1[m1[0], m1[0]] / 4 + hop_mat1[m1[1], m1[1]] / 4
        hop_mat2[m1[2], m1[2]] = hop_mat1[m1[2], m1[2]]
        hop_mat2[m1[2], m1[1]] = \
            sqrt_3 * hop_mat1[m1[2], m1[0]] / 2 - hop_mat1[m1[2], m1[1]] / 2
        hop_mat3[m1[2], m1[1]] = \
            -sqrt_3 * hop_mat1[m1[2], m1[0]] / 2 - \
            hop_mat1[m1[2], m1[1]] / 2
        hop_mat2[m1[0], m1[1]] = \
            sqrt_3 * (hop_mat1[m1[0], m1[0]] - hop_mat1[m1[1], m1[1]]) / 4 \
            - hop_mat1[m1[0], m1[1]]
        hop_mat3[m1[0], m1[1]] = \
            -sqrt_3 * (hop_mat1[m1[0], m1[0]] - hop_mat1[m1[1], m1[1]]) / 4\
            - hop_mat1[m1[0], m1[1]]
        hop_mat2[m1[2], m1[0]] = \
            hop_mat1[m1[2], m1[0]] / 2 + sqrt_3 * hop_mat1[m1[2], m1[1]] / 2
        hop_mat3[m1[2], m1[0]] = \
            hop_mat1[m1[2], m1[0]] / 2 - sqrt_3 * hop_mat1[m1[2], m1[1]] / 2
    hop_mat2[0, 0] = hop_mat1[0, 0] / 4 + 3 * hop_mat1[1, 1] / 4
    hop_mat2[1, 1] = 3 * hop_mat1[0, 0] / 4 + hop_mat1[1, 1] / 4
    hop_mat2[0, 1] = \
        sqrt_3 * (hop_mat1[0, 0] - hop_mat1[1, 1]) / 4 - hop_mat1[0, 1]
    hop_mat3[0, 1] = \
        -sqrt_3 * (hop_mat1[0, 0] - hop_mat1[1, 1]) / 4 - hop_mat1[0, 1]

    for m2 in list00:
        hop_mat4[m2[2], m2[0]] = \
            hop_mat5[m2[2], m2[0]] / 4 + 3 * hop_mat5[m2[3], m2[1]] / 4
        hop_mat4[m2[3], m2[1]] = \
            3 * hop_mat5[m2[2], m2[0]] / 4 + hop_mat5[m2[3], m2[1]] / 4
        hop_mat4[m2[3], m2[0]] = \
            -sqrt_3 * hop_mat5[m2[2], m2[0]] / 4 + sqrt_3 * \
            hop_mat5[m2[3], m2[1]] / 4
        hop_mat4[m2[2], m2[1]] = hop_mat4[m2[3], m2[0]]
        hop_mat4[m2[4], m2[0]] = -sqrt_3 * hop_mat5[m2[4], m2[1]] / 2
        hop_mat4[m2[4], m2[1]] = -hop_mat5[m2[4], m2[1]] / 2
    hop_mat4[8, 5] = hop_mat5[8, 5]
    hop_mat4[9, 5] = -sqrt_3 * hop_mat5[10, 5] / 2
    hop_mat4[10, 5] = -hop_mat5[10, 5] / 2

    # Symmetric orbitals are used in the literature, and asymmetric orbitals
    # are used here. So we need to make a unitary transformation of the hopping
    # matrix.
    trans_mat = \
        np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, inv_sqrt_2, 0, 0, inv_sqrt_2],
                  [0, 0, 0, 0, 0, inv_sqrt_2, 0, 0, -inv_sqrt_2, 0, 0],
                  [0, 0, 0, 0, 0, 0, inv_sqrt_2, 0, 0, -inv_sqrt_2, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, inv_sqrt_2, 0, 0, -inv_sqrt_2],
                  [0, 0, 0, 0, 0, inv_sqrt_2, 0, 0, inv_sqrt_2, 0, 0],
                  [0, 0, 0, 0, 0, 0, inv_sqrt_2, 0, 0, inv_sqrt_2, 0]]).transpose()
    inverse_trans_mat = trans_mat.transpose()

    # Indices used in deriving the tight binding model in the literature
    list1 = [1, 2, 6, 7, 8]
    list2 = [3, 4, 5, 9, 10, 11]
    list3 = [[3, 1], [5, 1], [4, 2], [10, 6], [9, 7], [11, 7], [10, 8]]
    list4 = [[4, 1], [3, 2], [5, 2], [9, 6], [11, 6], [10, 7], [9, 8], [11, 8]]
    hop_dict = HopDict(11)

    # Set hopping matrix within (0, 0, 0) cell
    a_000 = np.array(np.zeros((11, 11)))
    for i in list1:
        a_000[i - 1, i - 1] = epsilon[i - 1]
    for i in list1:
        for j in list2:
            if [j, i] in list4:
                a_000[i - 1, j - 1] = hop_mat5[j - 1, i - 1]
    for i in list2:
        a_000[i - 1, i - 1] = epsilon[i - 1]
    for i in list2:
        for j in list1:
            if [i, j] in list4:
                a_000[i - 1, j - 1] = hop_mat5[i - 1, j - 1]
    a_000 = np.dot(np.dot(trans_mat, a_000), inverse_trans_mat)
    on_site = []
    for i in range(11):
        on_site.append(a_000[i, i])
        a_000[i, i] = 0.
    hop_dict.set_mat((0, 0, 0), a_000)  # on-site energies will be set separately

    # Set hopping matrix to (1, 0, 0) cell
    a_100 = np.array(np.zeros((11, 11)))
    for i in list1:
        for j in list1:
            if i == j:
                a_100[i - 1, i - 1] = hop_mat2[i - 1, i - 1]
            elif (i, j) == (6, 8):
                a_100[i - 1, j - 1] = hop_mat3[i - 1, j - 1]
                a_100[j - 1, i - 1] = hop_mat2[i - 1, j - 1]
            elif (i, j) in ((1, 2), (6, 7), (7, 8)):
                a_100[i - 1, j - 1] = -hop_mat3[i - 1, j - 1]
                a_100[j - 1, i - 1] = hop_mat2[i - 1, j - 1]
    for i in list1:
        for j in list2:
            if [j, i] in list3:
                a_100[i - 1, j - 1] = hop_mat4[j - 1, i - 1]
            elif [j, i] in list4:
                a_100[i - 1, j - 1] = hop_mat4[j - 1, i - 1]
    for i in list2:
        for j in list2:
            if i == j:
                a_100[i - 1, j - 1] = hop_mat2[i - 1, j - 1]
            elif (i, j) in ((3, 5), (9, 11)):
                a_100[i - 1, j - 1] = hop_mat3[i - 1, j - 1]
                a_100[j - 1, i - 1] = hop_mat2[i - 1, j - 1]
            elif (i, j) in ((3, 4), (4, 5), (9, 10), (10, 11)):
                a_100[i - 1, j - 1] = -hop_mat3[i - 1, j - 1]
                a_100[j - 1, i - 1] = hop_mat2[i - 1, j - 1]
    a_100 = np.dot(np.dot(trans_mat, a_100), inverse_trans_mat)
    hop_dict.set_mat((1, 0, 0), a_100)

    # (0, 0, 0) -> (1, 1, 0)
    a_110 = np.array(np.zeros((11, 11)))
    for i in list1:
        for j in list2:
            if (j, i) == (9, 6):
                a_110[i - 1, j - 1] = hop_mat6[j - 1, i - 1]
            elif (j, i) == (11, 6):
                a_110[i - 1, j - 1] = hop_mat6[j - 1, i - 1]
            elif (j, i) == (9, 8):
                a_110[i - 1, j - 1] = hop_mat6[j - 1, i - 1]
            elif (j, i) == (11, 8):
                a_110[i - 1, j - 1] = hop_mat6[j - 1, i - 1]
    a_110 = np.dot(np.dot(trans_mat, a_110), inverse_trans_mat)
    hop_dict.set_mat((1, 1, 0), a_110)

    # (0, 0, 0) -> (0, 1, 0)
    a_010 = np.array(np.zeros((11, 11)))
    for i in list1:
        for j in list1:
            if i == j:
                a_010[i - 1, j - 1] = hop_mat2[i - 1, j - 1]
            elif (i, j) == (6, 8):
                a_010[i - 1, j - 1] = hop_mat3[i - 1, j - 1]
                a_010[j - 1, i - 1] = hop_mat2[i - 1, j - 1]
            elif (i, j) in ((1, 2), (6, 7), (7, 8)):
                a_010[i - 1, j - 1] = hop_mat3[i - 1, j - 1]
                a_010[j - 1, i - 1] = -hop_mat2[i - 1, j - 1]
    for i in list1:
        for j in list2:
            if [j, i] in list3:
                a_010[i - 1, j - 1] = -hop_mat4[j - 1, i - 1]
            elif [j, i] in list4:
                a_010[i - 1, j - 1] = hop_mat4[j - 1, i - 1]
    for i in list2:
        for j in list2:
            if i == j:
                a_010[i - 1, j - 1] = hop_mat2[i - 1, j - 1]
            elif (i, j) in ((3, 5), (9, 11)):
                a_010[i - 1, j - 1] = hop_mat3[i - 1, j - 1]
                a_010[j - 1, i - 1] = hop_mat2[i - 1, j - 1]
            elif (i, j) in ((3, 4), (4, 5), (9, 10), (10, 11)):
                a_010[i - 1, j - 1] = hop_mat3[i - 1, j - 1]
                a_010[j - 1, i - 1] = -hop_mat2[i - 1, j - 1]
    a_010 = np.dot(np.dot(trans_mat, a_010), inverse_trans_mat)
    hop_dict.set_mat((0, 1, 0), a_010)

    # (0, 0, 0) -> (-1, 1, 0)
    a__110 = np.array(np.zeros((11, 11)))
    for i in list1:
        for j in list1:
            if i == j:
                a__110[i - 1, i - 1] = hop_mat1[i - 1, i - 1]
            elif (i, j) == (6, 8):
                a__110[i - 1, j - 1] = hop_mat1[i - 1, j - 1]
                a__110[j - 1, i - 1] = hop_mat1[i - 1, j - 1]
            elif (i, j) == (1, 2) or (i, j) == (6, 7) or (i, j) == (7, 8):
                a__110[i - 1, j - 1] = hop_mat1[i - 1, j - 1]
                a__110[j - 1, i - 1] = -hop_mat1[i - 1, j - 1]
    for i in list1:
        for j in list2:
            if (j, i) == (9, 6):
                a__110[i - 1, j - 1] = hop_mat6[j - 1, i - 1]
            elif (j, i) == (11, 6):
                a__110[i - 1, j - 1] = -0.5 * hop_mat6[j - 1, i - 1]
                a__110[5, 9] = -sqrt_3 * hop_mat6[j - 1, i - 1] / 2
            elif (j, i) == (9, 8):
                a__110[i - 1, j - 1] = -0.5 * hop_mat6[j - 1, i - 1]
                a__110[6, 8] = -sqrt_3 * hop_mat6[j - 1, i - 1] / 2
            elif (j, i) == (11, 8):
                a__110[6, 9] = 3 * hop_mat6[j - 1, i - 1] / 4
                a__110[6, 10] = sqrt_3 * hop_mat6[j - 1, i - 1] / 4
                a__110[7, 9] = sqrt_3 * hop_mat6[j - 1, i - 1] / 4
                a__110[7, 10] = hop_mat6[j - 1, i - 1] / 4
    for i in list2:
        for j in list1:
            if (i, j) == (9, 6):
                a__110[i - 1, j - 1] = hop_mat6[i - 1, j - 1]
            elif (i, j) == (11, 6):
                a__110[i - 1, j - 1] = -0.5 * hop_mat6[i - 1, j - 1]
                a__110[9, 5] = sqrt_3 * hop_mat6[i - 1, j - 1] / 2
            elif (i, j) == (9, 8):
                a__110[i - 1, j - 1] = -0.5 * hop_mat6[i - 1, j - 1]
                a__110[8, 6] = sqrt_3 * hop_mat6[i - 1, j - 1] / 2
            elif (i, j) == (11, 8):
                a__110[9, 6] = 3 * hop_mat6[i - 1, j - 1] / 4
                a__110[10, 6] = -sqrt_3 * hop_mat6[i - 1, j - 1] / 4
                a__110[9, 7] = -sqrt_3 * hop_mat6[i - 1, j - 1] / 4
                a__110[10, 7] = hop_mat6[i - 1, j - 1] / 4
    for i in list2:
        for j in list2:
            if i == j:
                a__110[i - 1, i - 1] = hop_mat1[i - 1, i - 1]
            elif (i, j) in ((3, 5), (9, 11)):
                a__110[i - 1, j - 1] = hop_mat1[i - 1, j - 1]
                a__110[j - 1, i - 1] = hop_mat1[i - 1, j - 1]
            elif (i, j) in ((3, 4), (4, 5), (9, 10), (10, 11)):
                a__110[i - 1, j - 1] = hop_mat1[i - 1, j - 1]
                a__110[j - 1, i - 1] = -hop_mat1[i - 1, j - 1]
    a__110 = np.dot(np.dot(trans_mat, a__110), inverse_trans_mat)
    hop_dict.set_mat((-1, 1, 0), a__110)
    return hop_dict, on_site


def _gen_orb_labels(material: str = "MoS2") -> List[str]:
    """
    Generate orbital labels.

    :param material: chemical label of material
        Should be in ("MoS2", "MoSe2", "WS2", "WSe2")
    :return: orbital labels
    :raises NotImplementedError: if material has not been implemented
    """
    d_orbitals = ("dz2", "dxy", "dx2-y2", "dxz", "dyz")
    p_orbitals = ("px", "py", "pz")
    element_dict = {"MoS2": ("Mo", "SA", "SB"),
                    "MoSe2": ("Mo", "SeA", "SeB"),
                    "WS2": ("W", "SA", "SB"),
                    "WSe2": ("W", "SeA", "SeB")}
    try:
        elements = element_dict[material]
    except KeyError as err:
        raise NotImplementedError(f"{material} not implemented yet") from err
    else:
        orb_labels = []
        for orb in d_orbitals:
            orb_labels.append(f"{elements[0]}_{orb}")
        for orb in p_orbitals:
            orb_labels.append(f"{elements[1]}_{orb}")
        for orb in p_orbitals:
            orb_labels.append(f"{elements[2]}_{orb}")
    return orb_labels


def make_tmdc(material: str = "MoS2") -> PrimitiveCell:
    """
    Make TMDC primitive cell.

    :param material: chemical label of material
        Should be in ("MoS2", "MoSe2", "WS2", "WSe2")
    :return: TMDC primitive cell
    :raises NotImplementedError: if material has not been implemented
    """
    vectors, orbital_coords = _gen_lattice(material)
    hop_dict, on_site = _gen_hop_dict(material)
    orbital_coords = cart2frac(vectors, orbital_coords)
    orbital_labels = _gen_orb_labels(material)
    cell = PrimitiveCell(vectors, unit=NM)
    for i_o, coord in enumerate(orbital_coords):
        cell.add_orbital(coord, on_site[i_o], orbital_labels[i_o])
    cell.add_hopping_dict(hop_dict)
    return cell
