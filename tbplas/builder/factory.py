"""
Functions and classes for constructing samples.

Functions
---------
    extend_prim_cell: user function
        extend primitive cell along a, b, and c directions
        reserved for compatibility with old version of TBPlaS
    reshape_prim_cell: user function
        reshape primitive cell to given lattice vectors and origin
    spiral_prim_cell: user function
        rotate and shift primitive cell with respect to z-axis
    make_hetero_layer: user function
        make one layer in the hetero-structure by reshaping primitive cell to
        given lattice vectors
    merge_prim_cell: user function
        merge primitive cells and inter-hopping dictionaries to build a large
        primitive cell


Classes
-------
    PCInterHopping: user class
        container for holding hopping terms between different primitive cells in
        hetero-structure
"""

import math
from typing import Union

import numpy as np
from scipy.spatial import KDTree

from . import constants as consts
from . import exceptions as exc
from .lattice import frac2cart, cart2frac, rotate_coord
from .base import correct_coord, InterHopping
from .primitive import PrimitiveCell


def extend_prim_cell(prim_cell: PrimitiveCell, dim=(1, 1, 1)):
    """
    Extend primitive cell along a, b and c directions.

    :param prim_cell: instance of 'PrimitiveCell'
        primitive cell from which the extended cell is constructed
    :param dim: (na, nb, nc)
        dimension of the extended cell along 3 directions
    :return: extend_cell: instance of 'PrimitiveCell'
        extended cell created from primitive cell
    :raises CoordLenError: if len(dim) != 2 or 3
    :raises ValueError: if dimension along any direction is smaller than 1
    """
    # Check the dimension of extended cell
    dim = correct_coord(dim, complete_item=1)
    for i_dim in range(len(dim)):
        if dim[i_dim] < 1:
            raise ValueError(f"Dimension along direction {i_dim} should not"
                             f" be smaller than 1")

    # Extend lattice vectors
    lat_vec_ext = prim_cell.lat_vec.copy()
    for i_dim in range(3):
        lat_vec_ext[i_dim] *= dim[i_dim]

    # Create extended cell and add orbitals
    extend_cell = PrimitiveCell(lat_vec_ext, unit=consts.NM)
    extend_cell.extended = prim_cell.extended * np.prod(dim)
    orb_id_pc, orb_id_sc = [], {}
    id_sc = 0
    for i_a in range(dim[0]):
        for i_b in range(dim[1]):
            for i_c in range(dim[2]):
                for i_o, orbital in enumerate(prim_cell.orbital_list):
                    id_pc = (i_a, i_b, i_c, i_o)
                    orb_id_pc.append(id_pc)
                    orb_id_sc[id_pc] = id_sc
                    id_sc += 1
                    pos_ext = [(orbital.position[0] + i_a) / dim[0],
                               (orbital.position[1] + i_b) / dim[1],
                               (orbital.position[2] + i_c) / dim[2]]
                    extend_cell.add_orbital(pos_ext, orbital.energy,
                                            orbital.label)

    # Define periodic boundary condition.
    def _wrap_pbc(ji, ni):
        return ji % ni, ji // ni

    # Add hopping terms
    pc_hop_dict = prim_cell.hopping_dict.dict
    for id_sc_i in range(extend_cell.num_orb):
        id_pc_i = orb_id_pc[id_sc_i]
        for rn, hop_rn in pc_hop_dict.items():
            for pair, energy in hop_rn.items():
                if id_pc_i[3] == pair[0]:
                    ja, na = _wrap_pbc(id_pc_i[0] + rn[0], dim[0])
                    jb, nb = _wrap_pbc(id_pc_i[1] + rn[1], dim[1])
                    jc, nc = _wrap_pbc(id_pc_i[2] + rn[2], dim[2])
                    id_pc_j = (ja, jb, jc, pair[1])
                    id_sc_j = orb_id_sc[id_pc_j]
                    rn = (na, nb, nc)
                    extend_cell.add_hopping(rn, id_sc_i, id_sc_j, energy)

    # NOTE: if you modify orbital_list and hopping_dict of extend_cell manually,
    # then sync_array should be called with force_sync=True. Or alternatively,
    # update the timestamps of orb_list and hop_dict.
    extend_cell.sync_array()
    return extend_cell


def reshape_prim_cell(prim_cell: PrimitiveCell, lat_frac: np.ndarray,
                      origin: np.ndarray = np.zeros(3),
                      delta=0.01, pos_tol=1e-5):
    """
    Reshape primitive cell to given lattice vectors and origin.

    :param prim_cell: instance of 'PrimitiveCell' class
        primitive cell from which the reshaped cell is constructed
    :param lat_frac: (3, 3) float64 array
        FRACTIONAL coordinates of lattice vectors of reshaped cell in basis
        vectors of primitive cell
    :param origin: (3,) float64 array
        FRACTIONAL coordinates of origin of reshaped cell in basis vectors
        of reshaped cell
    :param delta: float64
        small parameter to add to origin such that orbitals fall on cell
        borders will not be clipped
        This parameter will be subtracted from orbital positions of reshaped
        cell. So the origin is still correct.
    :param pos_tol: float64
        tolerance on positions for identifying equivalent orbitals
    :return: res_cell: instance of 'PrimitiveCell' class
        reshaped cell
    :raises LatVecError: if shape of lat_frac.shape is not (3, 3)
    :raises ValueError: if length of origin is not 3
    """
    # Check lattice vectors and origin
    if lat_frac.shape != (3, 3):
        raise exc.LatVecError()
    if origin.shape != (3,):
        raise ValueError("Length of origin is not 3")

    # Conversion matrix of fractional coordinates from primitive cell to
    # reshaped cell: x_res = x_prim * conv_mat, with x_new and x_prim
    # being ROW vectors
    conv_mat = np.linalg.inv(lat_frac)

    # Function for getting cell index from fractional coordinates
    def _get_cell_index(x):
        return math.floor(x.item(0)), math.floor(x.item(1)), \
               math.floor(x.item(2))

    # Create reshaped cell
    lat_cart = np.zeros((3, 3), dtype=np.float64)
    for i_dim in range(3):
        lat_cart[i_dim] = np.matmul(lat_frac[i_dim], prim_cell.lat_vec)
    res_cell = PrimitiveCell(lat_cart, unit=1.0)
    vol_res = res_cell.get_lattice_volume()
    vol_prim = prim_cell.get_lattice_volume()
    res_cell.extended = prim_cell.extended * (vol_res / vol_prim)

    # Add orbitals
    prim_cell.sync_array()
    rn_range = np.zeros((3, 2), dtype=np.int32)
    for i_dim in range(3):
        sum_vec = lat_frac.sum(axis=0) - lat_frac[i_dim]
        for j_dim in range(3):
            rn_range[j_dim, 0] = min(rn_range.item(j_dim, 0),
                                     math.floor(sum_vec[j_dim]))
            rn_range[j_dim, 1] = max(rn_range.item(j_dim, 1),
                                     math.ceil(sum_vec[j_dim]))
    rn_range[:, 0] -= 1
    rn_range[:, 1] += 1

    orb_id_pc, orb_id_sc = [], {}
    id_sc = 0
    for i_a in range(rn_range.item(0, 0), rn_range.item(0, 1)+1):
        for i_b in range(rn_range.item(1, 0), rn_range.item(1, 1)+1):
            for i_c in range(rn_range.item(2, 0), rn_range.item(2, 1)+1):
                rn = (i_a, i_b, i_c)
                for i_o, pos in enumerate(prim_cell.orb_pos):
                    res_pos = np.matmul(rn + pos, conv_mat) - origin + delta
                    res_rn = _get_cell_index(res_pos)
                    if res_rn == (0, 0, 0):
                        id_pc = (i_a, i_b, i_c, i_o)
                        orb_id_pc.append(id_pc)
                        orb_id_sc[id_pc] = id_sc
                        id_sc += 1
                        res_cell.add_orbital(res_pos,
                                             prim_cell.orb_eng.item(i_o),
                                             prim_cell.orbital_list[i_o].label)

    # Add hopping terms
    res_cell.sync_array(force_sync=True)
    kd_tree = KDTree(res_cell.orb_pos)
    for id_sc_i in range(res_cell.num_orb):
        id_pc_i = orb_id_pc[id_sc_i]
        for i_h, hop in enumerate(prim_cell.hop_ind):
            if id_pc_i[3] == hop.item(3):
                # Get fractional coordinate of id_sc_j in reshaped cell
                rn = id_pc_i[:3] + hop[:3]
                pos = prim_cell.orb_pos[hop.item(4)]
                res_pos = np.matmul(rn + pos, conv_mat) - origin + delta

                # Wrap back into (0, 0, 0)-th reshaped cell
                res_rn = _get_cell_index(res_pos)
                res_pos -= res_rn

                # Determine corresponding id_sc_j
                neighbours = kd_tree.query_ball_point(res_pos, r=pos_tol)
                for id_sc_j in neighbours:
                    id_pc_j = orb_id_pc[id_sc_j]
                    if id_pc_j[3] == hop.item(4):
                        res_cell.add_hopping(res_rn, id_sc_i, id_sc_j,
                                             prim_cell.hop_eng.item(i_h))

    # Subtract delta from orbital positions
    res_cell.orb_pos -= delta
    for i_o, pos in enumerate(res_cell.orb_pos):
        res_cell.set_orbital(i_o, position=tuple(pos))

    # NOTE: if you modify orbital_list and hopping_dict of res_cell manually,
    # then sync_array should be called with force_sync=True. Or alternatively,
    # update the timestamps of orb_list and hop_dict.
    res_cell.sync_array()
    return res_cell


def spiral_prim_cell(prim_cell: PrimitiveCell, angle=0.0, shift=0.0):
    """
    Rotate and shift primitive cell with respect to z-axis.

    :param prim_cell: instance of 'PrimitiveCell' class
        primitive cell to twist
    :param angle: float
        twisting angle in RADIANs, NOT degrees
    :param shift: float
        distance of shift in NANOMETER
    :return: None
        Incoming primitive cell is modified.
    """
    prim_cell.sync_array()

    # Since prim_cell uses fractional coordinates internally, we
    # just need to rotate its lattice vectors.
    prim_cell.lat_vec = rotate_coord(prim_cell.lat_vec, angle)

    # Shift coordinates
    orb_pos = frac2cart(prim_cell.lat_vec, prim_cell.orb_pos)
    orb_pos[:, 2] += shift
    orb_pos = cart2frac(prim_cell.lat_vec, orb_pos)
    for i, pos in enumerate(orb_pos):
        prim_cell.set_orbital(i, position=tuple(pos))

    # NOTE: if you modify orbital_list and hopping_dict of prim_cell manually,
    # then sync_array should be called with force_sync=True. Or alternatively,
    # update the timestamps of orb_list and hop_dict.
    prim_cell.sync_array()


def make_hetero_layer(prim_cell: PrimitiveCell, hetero_lattice: np.ndarray,
                      **kwargs):
    """
    Make one layer in the hetero-structure by reshaping primitive cell to
    given lattice vectors.

    :param prim_cell: instance of 'PrimitiveCell' class
        primitive cell of the layer
    :param hetero_lattice: (3, 3) float64 array
        Cartesian coordinates of hetero-structure lattice vectors in NANOMETER
    :param kwargs: dictionary
        keyword arguments for 'reshape_prim_cell'
    :return: hetero_layer: instance of 'PrimitiveCell' class
        layer in the hetero-structure
    """
    hetero_lattice_frac = cart2frac(prim_cell.lat_vec, hetero_lattice)
    hetero_layer = reshape_prim_cell(prim_cell, hetero_lattice_frac, **kwargs)
    return hetero_layer


class PCInterHopping(InterHopping):
    """
    Class for holding hopping terms between different primitive cells
    in hetero-structure.

    Attributes
    ----------
    pc_bra: instance of 'PrimitiveCell' class
        the 'bra' primitive cell from which the hopping terms exist
    pc_ket: instance of 'PrimitiveCell' class
        the 'ket' primitive cell from which the hopping terms exist

    NOTES
    -----
    We assume hopping terms to be from (0, 0, 0) cell of pc_bra to any cell of
    pc_ket. The counterparts can be restored via the conjugate relation:
        <pc_bra, R0, i|H|pc_ket, Rn, j> = <pc_ket, R0, j|H|pc_bra, -Rn, i>*
    """
    def __init__(self, pc_bra: PrimitiveCell, pc_ket: PrimitiveCell):
        """
        :param pc_bra: instance of 'PrimitiveCell' class
            the 'bra' primitive cell from which the hopping terms exist
        :param pc_ket: instance of 'PrimitiveCell' class
            the 'ket' primitive cell from which the hopping terms exist
        """
        super().__init__()
        self.pc_bra = pc_bra
        self.pc_ket = pc_ket


def merge_prim_cell(*args: Union[PrimitiveCell, PCInterHopping]):
    """
    Merge primitive cells and inter-hopping dictionaries to build a large
    primitive cell.

    :param args: list of 'PrimitiveCell' or 'PCInterHopping' instances
        primitive cells and inter-hopping terms within the large primitive cell
    :return: merged_cell: instance of 'PrimitiveCell'
        merged primitive cell
    :raises ValueError: if no arg is given, or any arg is not instance of
        PrimitiveCell or PCInterHopping, or any inter_hop_dict involves primitive
        cells not included in args
    :raises PCOrbIndexError: if any orbital index in any inter_hop_dict is out
        of range
    """
    # Check arguments
    if len(args) == 0:
        raise ValueError("No components assigned to the primitive cell")

    # Parse primitive cells and inter-hopping terms
    pc_list = []
    hop_list = []
    for i_arg, arg in enumerate(args):
        if isinstance(arg, PrimitiveCell):
            pc_list.append(arg)
        elif isinstance(arg, PCInterHopping):
            hop_list.append(arg)
        else:
            raise ValueError(f"Component #{i_arg} should be instance of "
                             f"PrimitiveCell or PCInterHopping")

    # Check closure of inter-hopping instances
    for i_h, hop in enumerate(hop_list):
        if hop.pc_bra not in pc_list:
            raise ValueError(f"pc_bra of inter_hop #{i_h} not included in args")
        if hop.pc_ket not in pc_list:
            raise ValueError(f"pc_ket of inter_hop #{i_h} not included in args")

    # Get numbers of orbitals of each component
    num_orb = [pc.num_orb for pc in pc_list]

    # Get starting indices of orbitals of each component
    ind_start = [sum(num_orb[:_]) for _ in range(len(num_orb))]

    # Create merged primitive cell
    merged_cell = PrimitiveCell(pc_list[0].lat_vec, unit=consts.NM)

    # Determine the 'extended' attribute
    extended = pc_list[0].extended
    for pc in pc_list:
        if (pc.extended - extended) >= 1.e-3:
            print(f"INFO: resetting extend to 1.0")
            extended = 1.0
            break
    merged_cell.extended = extended

    # Add orbitals
    for pc in pc_list:
        for orb in pc.orbital_list:
            merged_cell.add_orbital(position=orb.position, energy=orb.energy,
                                    label=orb.label)

    # Add intra-hopping terms
    for i_pc, pc in enumerate(pc_list):
        offset = ind_start[i_pc]
        hop_dict = pc.hopping_dict.dict
        for rn, hop_rn in hop_dict.items():
            for pair, energy in hop_rn.items():
                orb_i = pair[0] + offset
                orb_j = pair[1] + offset
                merged_cell.add_hopping(rn=rn, orb_i=orb_i, orb_j=orb_j,
                                        energy=energy)

    # Add inter-hopping terms
    for hop in hop_list:
        offset_bra = ind_start[pc_list.index(hop.pc_bra)]
        offset_ket = ind_start[pc_list.index(hop.pc_ket)]
        for rn, hop_terms in hop.dict.items():
            for orb_pair, energy in hop_terms.items():
                orb_i = orb_pair[0] + offset_bra
                orb_j = orb_pair[1] + offset_ket
                merged_cell.add_hopping(rn, orb_i=orb_i, orb_j=orb_j,
                                        energy=energy)

    # NOTE: if you modify orbital_list and hopping_dict of merged_cell manually,
    # then sync_array should be called with force_sync=True. Or alternatively,
    # update the timestamps of orb_list and hop_dict.
    merged_cell.sync_array()
    return merged_cell
