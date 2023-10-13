"""Functions and classes for advanced modeling."""

from math import floor, ceil, sqrt
from typing import Union, List, Tuple, Dict
from collections import namedtuple
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import leastsq

from ..base import cart2frac, rotate_coord
from ..base import constants as consts
from . import exceptions as exc
from .base import check_rn, rn_type, id_pc_type
from .primitive import PrimitiveCell, PCInterHopping
from .super import SuperCell


__all__ = ["extend_prim_cell", "reshape_prim_cell", "spiral_prim_cell",
           "make_hetero_layer", "merge_prim_cell", "find_neighbors",
           "SK", "SOC", "SOCTable", "ParamFit"]


class OrbitalMap:
    """
    Helper class for converting orbital indices between pc and sc
    representations, inspired by the 'OrbitalSet' class in super.py.

    NOTE: this class is intended to be utilized by 'extend_prim_cell' and
    'reshape_prim_cell' functions only. The orbital indices passed to 'add'
    method must be non-redundant. Otherwise, bugs will be definitely raised.

    Attributes
    ----------
    _id_pc: List[Tuple[int, int, int, int]]
        orbital indices in pc representation
    _id_sc: Dict[Tuple[int, int, int, int], int]
        map of orbital indices from pc to sc representation
    _counter: int
        counter of the orbitals
    """
    def __init__(self) -> None:
        self._id_pc = []
        self._id_sc = dict()
        self._counter = 0

    def add(self, id_pc: id_pc_type) -> None:
        """
        Add an orbital from index in pc representation.

        :param id_pc: index of the orbital
        :return: None
        """
        self._id_pc.append(id_pc)
        self._id_sc[id_pc] = self._counter
        self._counter += 1

    def pc2sc(self, id_pc: id_pc_type) -> int:
        """
        Convert orbital index from pc to sc representation.

        :param id_pc: index of the orbital
        :return: corresponding index in sc representation
        """
        return self._id_sc[id_pc]

    def sc2pc(self, id_sc: int) -> id_pc_type:
        """
        Convert orbital index from sc to pc representation.

        :param id_sc: index of the orbital
        :return: corresponding index in pc representation
        """
        return self._id_pc[id_sc]


def extend_prim_cell(prim_cell: PrimitiveCell,
                     dim: rn_type = (1, 1, 1)) -> PrimitiveCell:
    """
    Extend primitive cell along a, b and c directions.

    :param prim_cell: primitive cell from which the extended cell is constructed
    :param dim: dimension of the extended cell along 3 directions
    :return: extended cell created from primitive cell
    :raises CoordLenError: if len(dim) != 2 or 3
    :raises ValueError: if dimension along any direction is smaller than 1
    """
    # Check the dimension of extended cell
    dim, legal = check_rn(dim, complete_item=1)
    if not legal:
        raise exc.CoordLenError(dim)
    for i_dim in range(len(dim)):
        if dim[i_dim] < 1:
            raise ValueError(f"Dimension along direction {i_dim} should not"
                             f" be smaller than 1")

    # Extend lattice vectors
    lat_vec_ext = prim_cell.lat_vec.copy()
    for i_dim in range(3):
        lat_vec_ext[i_dim] *= dim[i_dim]

    # Extended cell shares the same origin
    origin_ext = prim_cell.origin.copy()

    # Create extended cell and add orbitals
    extend_cell = PrimitiveCell(lat_vec_ext, origin_ext, unit=consts.NM)
    extend_cell.extended = prim_cell.extended * np.prod(dim)
    orb_map = OrbitalMap()
    for i_a in range(dim[0]):
        for i_b in range(dim[1]):
            for i_c in range(dim[2]):
                for i_o, orbital in enumerate(prim_cell.orbitals):
                    orb_map.add((i_a, i_b, i_c, i_o))
                    pos_ext = ((orbital.position[0] + i_a) / dim[0],
                               (orbital.position[1] + i_b) / dim[1],
                               (orbital.position[2] + i_c) / dim[2])
                    extend_cell.add_orbital(pos_ext, orbital.energy,
                                            orbital.label)

    # Define periodic boundary condition.
    def _wrap_pbc(ji, ni):
        return ji % ni, ji // ni

    # Add hopping terms
    for id_sc_i in range(extend_cell.num_orb):
        id_pc_i = orb_map.sc2pc(id_sc_i)
        for rn, hop_rn in prim_cell.hoppings.items():
            for pair, energy in hop_rn.items():
                if id_pc_i[3] == pair[0]:
                    ja, na = _wrap_pbc(id_pc_i[0] + rn[0], dim[0])
                    jb, nb = _wrap_pbc(id_pc_i[1] + rn[1], dim[1])
                    jc, nc = _wrap_pbc(id_pc_i[2] + rn[2], dim[2])
                    id_pc_j = (ja, jb, jc, pair[1])
                    id_sc_j = orb_map.pc2sc(id_pc_j)
                    ext_rn = (na, nb, nc)
                    extend_cell.add_hopping(ext_rn, id_sc_i, id_sc_j, energy)

    return extend_cell


def reshape_prim_cell(prim_cell: PrimitiveCell,
                      lat_frac: np.ndarray,
                      delta: float = 1e-2,
                      pos_tol: float = 1e-5) -> PrimitiveCell:
    """
    Reshape primitive cell to given lattice vectors.

    :param prim_cell: primitive cell from which the reshaped cell is constructed
    :param lat_frac: (3, 3) float64 array
        FRACTIONAL coordinates of lattice vectors of reshaped cell in basis
        vectors of primitive cell
    :param delta: small displacement added to orbital positions such that orbitals
        fall on cell borders will not be clipped
    :param pos_tol: tolerance on positions for identifying equivalent orbitals
    :return: reshaped cell
    :raises LatVecError: if shape of lat_frac.shape is not (3, 3)
    """
    # Check lattice vectors
    if lat_frac.shape != (3, 3):
        raise exc.LatVecError()

    # Conversion matrix of fractional coordinates from primitive cell to
    # reshaped cell: x_res = x_prim * conv_mat, with x_new and x_prim
    # being ROW vectors
    conv_mat = np.linalg.inv(lat_frac)

    # Reshaped cell lattice vectors
    lat_cart = np.zeros((3, 3), dtype=np.float64)
    for i_dim in range(3):
        lat_cart[i_dim] = np.matmul(lat_frac[i_dim], prim_cell.lat_vec)

    # Reshaped cell shares the same origin
    origin = prim_cell.origin.copy()

    # Create reshaped cell
    res_cell = PrimitiveCell(lat_cart, origin, unit=consts.NM)
    vol_res = res_cell.get_lattice_volume()
    vol_prim = prim_cell.get_lattice_volume()
    res_cell.extended = prim_cell.extended * (vol_res / vol_prim)

    # Determine searching range
    # sum_vec is actually a0+a1, a1+a2 or a2+a0 depending on j, i.e., the
    # diagonal vector. If it is not taken into consideration, orbitals and
    # hopping terms in the top right corner of reshaped cell may be missing.
    rn_range = np.zeros((3, 2), dtype=np.int32)
    for i in range(3):
        sum_vec = lat_frac.sum(axis=0) - lat_frac[i]
        for j in range(3):
            rn_range[j, 0] = min(rn_range[j, 0], floor(sum_vec[j]))
            rn_range[j, 1] = max(rn_range[j, 1], ceil(sum_vec[j]))
    rn_range[:, 0] -= 1
    rn_range[:, 1] += 1

    # Function for getting cell index from fractional coordinates
    def _get_cell_index(x):
        return floor(x.item(0)), floor(x.item(1)), floor(x.item(2))

    # Add orbitals
    orb_pos = prim_cell.orb_pos
    orb_map = OrbitalMap()
    for i_a in range(rn_range.item(0, 0), rn_range.item(0, 1)+1):
        for i_b in range(rn_range.item(1, 0), rn_range.item(1, 1)+1):
            for i_c in range(rn_range.item(2, 0), rn_range.item(2, 1)+1):
                rn = (i_a, i_b, i_c)
                for i_o, pos in enumerate(orb_pos):
                    res_pos = np.matmul(rn + pos, conv_mat)
                    res_rn = _get_cell_index(res_pos + delta)
                    if res_rn == (0, 0, 0):
                        orb_map.add((i_a, i_b, i_c, i_o))
                        res_cell.add_orbital(tuple(res_pos),
                                             prim_cell.orbitals[i_o].energy,
                                             prim_cell.orbitals[i_o].label)

    # Add hopping terms
    if prim_cell.num_hop > 0:
        hop_ind = prim_cell.hop_ind
        hop_eng = prim_cell.hop_eng
        kd_tree = KDTree(res_cell.orb_pos)
        for id_sc_i in range(res_cell.num_orb):
            id_pc_i = orb_map.sc2pc(id_sc_i)
            for i_h, hop in enumerate(hop_ind):
                if id_pc_i[3] == hop.item(3):
                    # Get cell index of id_sc_j in reshaped cell
                    rn = id_pc_i[:3] + hop[:3]
                    pos = orb_pos[hop.item(4)]
                    res_pos = np.matmul(rn + pos, conv_mat)
                    res_rn = _get_cell_index(res_pos + delta)

                    # Wrap fractional coordinate of id_sc_j back into the
                    # (0, 0, 0) reshaped cell
                    res_pos -= res_rn

                    # Determine corresponding id_sc_j
                    candidates = kd_tree.query_ball_point(res_pos, r=pos_tol)
                    for id_sc_j in candidates:
                        id_pc_j = orb_map.sc2pc(id_sc_j)
                        if id_pc_j[3] == hop.item(4):
                            res_cell.add_hopping(res_rn, id_sc_i, id_sc_j,
                                                 hop_eng.item(i_h))

    return res_cell


def spiral_prim_cell(prim_cell: PrimitiveCell,
                     angle: float = 0.0,
                     center: np.ndarray = np.zeros(3),
                     shift: float = 0.0) -> None:
    """
    Rotate and shift primitive cell with respect to z-axis.

    NOTE: this function returns nothing. But the incoming primitive cell
    will be modified in-place.

    :param prim_cell: primitive cell to twist
    :param angle: twisting angle in RADIANs, NOT degrees
    :param center: (3,) float64 array
        Cartesian coordinates of the rotation center in NANOMETER
    :param shift: distance of shift in NANOMETER
    :return: None
    """
    # Get rotated lattice vectors and origin
    end_points = np.vstack((np.zeros(3), prim_cell.lat_vec)) + prim_cell.origin
    end_points = rotate_coord(end_points, angle=angle, center=center)
    lat_vec = end_points[1:] - end_points[0]
    origin = end_points[0]

    # Reset lattice vectors and origin without fixing the orbitals
    prim_cell.reset_lattice(lat_vec, origin, unit=consts.NM, fix_orb=False)

    # Shift orbitals along z-axis
    orb_pos = prim_cell.orb_pos_nm + np.array([0, 0, shift])

    # Update orbital positions in fractional coordinates
    orb_pos = cart2frac(lat_vec, orb_pos, origin)
    for i, pos in enumerate(orb_pos):
        prim_cell.set_orbital(i, position=tuple(pos))

    # # Or alternatively, working with Cartesian coordinates
    # for i, pos in enumerate(orb_pos):
    #     prim_cell.set_orbital_cart(i, position=tuple(pos), unit=consts.NM)

    # # Or alternatively, shift the origin instead of orbitals
    # origin = end_points[0] + np.array([0, 0, shift])
    # prim_cell.reset_lattice(lat_vec, origin, unit=consts.NM, fix_orb=False)


def make_hetero_layer(prim_cell: PrimitiveCell,
                      hetero_lattice: np.ndarray,
                      **kwargs) -> PrimitiveCell:
    """
    Make one layer in the hetero-structure by reshaping primitive cell to
    given lattice vectors.

    :param prim_cell: primitive cell of the layer
    :param hetero_lattice: (3, 3) float64 array
        Cartesian coordinates of hetero-structure lattice vectors in NANOMETER
    :param kwargs: arguments for 'reshape_prim_cell'
    :return: layer in the hetero-structure
    """
    hetero_lattice_frac = cart2frac(prim_cell.lat_vec, hetero_lattice)
    hetero_layer = reshape_prim_cell(prim_cell, hetero_lattice_frac, **kwargs)
    return hetero_layer


def merge_prim_cell(*args: Union[PrimitiveCell, PCInterHopping]) -> PrimitiveCell:
    """
    Merge primitive cells and inter-hopping dictionaries to build a large
    primitive cell.

    :param args: primitive cells and inter-hopping terms within the large
        primitive cell
    :return: merged primitive cell
    :raises ValueError: if no arg is given, or any arg is not instance of
        PrimitiveCell or PCInterHopping, or any inter_hop_dict involves primitive
        cells not included in args, or lattice vectors of primitive cells do not
        match
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

    # Check lattice mismatch
    lat_ref = pc_list[0].lat_vec
    for i, pc in enumerate(pc_list):
        if np.sum(np.abs(pc.lat_vec - lat_ref)) > 1e-5:
            raise ValueError(f"Lattice vectors of pc #0 and #{i} mismatch!")

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
    # The merge cell shares the same lattice vector and origin as the 1st
    # primitive cell.
    origin_ref = pc_list[0].origin
    merged_cell = PrimitiveCell(lat_ref, origin_ref, unit=consts.NM)

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
        pc.reset_lattice(lat_ref, origin_ref, unit=consts.NM, fix_orb=True)
        for orb in pc.orbitals:
            merged_cell.add_orbital(position=orb.position, energy=orb.energy,
                                    label=orb.label)

    # Add intra-hopping terms
    for i_pc, pc in enumerate(pc_list):
        offset = ind_start[i_pc]
        for rn, hop_rn in pc.hoppings.items():
            for pair, energy in hop_rn.items():
                orb_i = pair[0] + offset
                orb_j = pair[1] + offset
                merged_cell.add_hopping(rn=rn, orb_i=orb_i, orb_j=orb_j,
                                        energy=energy)

    # Add inter-hopping terms
    for hop in hop_list:
        offset_bra = ind_start[pc_list.index(hop.pc_bra)]
        offset_ket = ind_start[pc_list.index(hop.pc_ket)]
        for rn, hop_terms in hop.hoppings.items():
            for orb_pair, energy in hop_terms.items():
                orb_i = orb_pair[0] + offset_bra
                orb_j = orb_pair[1] + offset_ket
                merged_cell.add_hopping(rn, orb_i=orb_i, orb_j=orb_j,
                                        energy=energy)
    return merged_cell


def find_neighbors(cell_bra: Union[PrimitiveCell, SuperCell],
                   cell_ket: Union[PrimitiveCell, SuperCell] = None,
                   a_max: int = 0,
                   b_max: int = 0,
                   c_max: int = 0,
                   max_distance: float = 1.0) -> List[namedtuple]:
    """
    Find neighbours between the (0, 0, 0) cell of model_bra and nearby cells of
    cell_ket up to given cutoff distance.

    NOTE: only neighbours with distance > 0 will be returned.

    The searching range of nearby cells is:
    [-a_max, a_max] * [-b_max, b_max] * [-c_max, c_max].

    The named tuples have four attributes: rn for cell index, pair for orbital
    indices, rij for Cartesian coordinates of displacement vector in nm and
    distance for the norm of rij.

    :param cell_bra: the 'bra' primitive cell or supercell
    :param cell_ket: the 'ket' primitive cell or supercell
        default to pc_ket if not set
    :param a_max: upper bound of range on a-axis
    :param b_max: upper bound of range on b-axis
    :param c_max: upper bound of range on c-axis
    :param max_distance: cutoff distance in NM
    :return: list of neighbors as named tuples
    :raise PCOrbEmptyError: if cell_bra or cell_ket does not contain orbitals
    """
    if cell_ket is None:
        cell_ket = cell_bra

    # Check for number of orbitals
    if isinstance(cell_bra, PrimitiveCell):
        cell_bra.verify_orbitals()
        cell_ket.verify_orbitals()

    # Get orbital positions
    if isinstance(cell_bra, PrimitiveCell):
        pos_bra = cell_bra.orb_pos_nm
        pos_ket = cell_ket.orb_pos_nm
    else:
        pos_bra = cell_bra.get_orb_pos()
        pos_ket = cell_ket.get_orb_pos()

    # Get lattice vectors of cell_ket
    if isinstance(cell_ket, PrimitiveCell):
        lat_ket = cell_ket.lat_vec
    else:
        lat_ket = cell_ket.sc_lat_vec

    # Prepare for the loop
    tree_bra = KDTree(pos_bra)
    neighbor_rn = [(ia, ib, ic)
                   for ia in range(-a_max, a_max+1)
                   for ib in range(-b_max, b_max+1)
                   for ic in range(-c_max, c_max+1)]
    Term = namedtuple("Term", ["rn", "pair", "rij", "distance"])

    # Loop over neighboring cells to search for orbital pairs
    neighbors = []
    for rn in neighbor_rn:
        pos_ket_rn = pos_ket + np.matmul(rn, lat_ket)
        tree_ket_rn = KDTree(pos_ket_rn)
        dist_matrix = tree_bra.sparse_distance_matrix(tree_ket_rn,
                                                      max_distance=max_distance)
        for pair, distance in dist_matrix.items():
            if distance > 0.0:
                i, j = pair
                rij = pos_ket_rn[j] - pos_bra[i]
                neighbors.append(Term(rn, pair, rij, distance))
    neighbors = sorted(neighbors, key=lambda x: x.distance)
    return neighbors


class SK:
    """
    Class for evaluating hopping integrals using Slater-Koster formula.

    The maximum supported angular momentum is l=2 (d orbitals).
    Reference: https://journals.aps.org/pr/abstract/10.1103/PhysRev.94.1498

    The reason why we make orbital labels as attributes is to avoid misspelling,
    which is likely to happen as we have to repeat them many times.
    """
    def __init__(self) -> None:
        self._s = "s"
        self._px = "px"
        self._py = "py"
        self._pz = "pz"
        self._dxy = "dxy"
        self._dyz = "dyz"
        self._dzx = "dzx"
        self._dx2_y2 = "dx2-y2"
        self._dz2 = "dz2"
        self._p_labels = {self._px, self._py, self._pz}
        self._d_labels = {self._dxy, self._dyz, self._dzx, self._dx2_y2,
                          self._dz2}
        self._sqrt3 = sqrt(3)
        self._half_sqrt3 = self._sqrt3 * 0.5

    def _check_p_labels(self, *labels: str) -> None:
        """
        Check the sanity of labels of p orbitals.

        :param labels: labels to check
        :return: None
        :raises ValueError: if any label is not in self.p_labels
        """
        for label in labels:
            if label not in self._p_labels:
                raise ValueError(f"Illegal label: {label}")

    def _check_d_labels(self, *labels: str) -> None:
        """
        Check the sanity of labels of d orbitals.

        :param labels: labels to check
        :return: None
        :raises ValueError: if any label is not in self.d_labels
        """
        for label in labels:
            if label not in self._d_labels:
                raise ValueError(f"Illegal label: {label}")

    @staticmethod
    def _perm_vector(vector: np.ndarray, x_new: str) -> np.ndarray:
        """
        Permute a given vector according to the new x_axis.

        :param vector: vector to permute
        :param x_new: label of the new x_axis
        :return: permuted vector
        :raises ValueError: if x_new is not in 'x', 'y', 'z'
        """
        if x_new == "x":
            return vector
        elif x_new == "y":
            return vector[[1, 2, 0]]
        elif x_new == "z":
            return vector[[2, 0, 1]]
        else:
            raise ValueError(f"Illegal x_new {x_new}")

    @staticmethod
    def _remap_label(label: str, x_new: str) -> str:
        """
        Remap orbital label after permutation.

        :param label: orbital label to remap
        :param x_new: index of the new 'x' direction
        :return: remapped orbital label
        :raises ValueError: if x_new is not in 'x', 'y', 'z'
        """
        if x_new == "x":
            return label
        else:
            if x_new == "y":
                # keys: x, y, z, values: x_new, y_new, z_new
                map_table = {"x": "z", "y": "x", "z": "y"}
            elif x_new == "z":
                # keys: x, y, z, values: x_new, y_new, z_new
                map_table = {"x": "y", "y": "z", "z": "x"}
            else:
                raise ValueError(f"Illegal new_x {x_new}")
            new_label = label[0]
            for c in label[1:]:
                new_label += map_table[c]
            return new_label

    @staticmethod
    def _eval_dir_cos(r: np.ndarray) -> Tuple[float, float, float]:
        """
        Evaluate direction cosines for given displacement vector.

        :param r: Cartesian coordinates of the displacement vector
        :return: the direction cosines along x, y, and z directions
        :raises ValueError: if the norm of r is too small
        """
        norm = np.linalg.norm(r)
        if norm <= 1.0e-15:
            raise ValueError("Norm of displacement vector too small")
        dir_cos = r / norm
        l, m, n = dir_cos.item(0), dir_cos.item(1), dir_cos.item(2)
        return l, m, n

    @staticmethod
    def ss(v_sss: complex = 0.0) -> complex:
        """
        Evaluate the hopping integral <s,0|H|s,r>.

        :param v_sss: V_ss_sigma
        :return: hopping integral
        """
        return v_sss

    def sp(self, r: np.ndarray,
           label_p: str = "px",
           v_sps: complex = 0.0) -> complex:
        """
        Evaluate the hopping integral <s,0|H|p,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_p: label of p orbital
        :param v_sps: V_sp_sigma
        :return: hopping integral
        :raises ValueError: if label_p is not in self.p_labels
        """
        self._check_p_labels(label_p)
        l, m, n = self._eval_dir_cos(r)
        if label_p == self._px:
            t = l * v_sps
        elif label_p == self._py:
            t = m * v_sps
        else:
            t = n * v_sps
        return t

    def sd(self, r: np.ndarray,
           label_d: str = "dxy",
           v_sds: complex = 0.0) -> complex:
        """
        Evaluate the hopping integral <s,0|H|d,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_d: label of d orbital
        :param v_sds: V_sd_sigma
        :return: hopping integral
        :raises ValueError: if label_d is not in self.d_labels
        """
        self._check_d_labels(label_d)

        # Permute the coordinates
        if label_d == self._dyz:
            x_new = "y"
        elif label_d == self._dzx:
            x_new = "z"
        else:
            x_new = "x"
        r = self._perm_vector(r, x_new)
        label_d = self._remap_label(label_d, x_new)

        # Evaluate the hopping integral
        l, m, n = self._eval_dir_cos(r)
        if label_d == self._dxy:
            t = self._sqrt3 * l * m * v_sds
        elif label_d == self._dx2_y2:
            t = self._half_sqrt3 * (l ** 2 - m ** 2) * v_sds
        elif label_d == self._dz2:
            t = (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * v_sds
        else:
            raise ValueError(f"Undefined label pair s {label_d}")
        return t

    def pp(self, r: np.ndarray,
           label_i: str = "px",
           label_j: str = "px",
           v_pps: complex = 0.0,
           v_ppp: complex = 0.0) -> complex:
        """
        Evaluate the hopping integral <p_i,0|H|p_j,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_i: label of p_i orbital
        :param label_j: label of p_j orbital
        :param v_pps: V_pp_sigma
        :param v_ppp: V_pp_pi
        :return: hopping integral
        :raises ValueError: if label_i or label_j is not in self.p_labels
        """
        self._check_p_labels(label_i, label_j)

        # Permute the coordinates
        x_new = label_i[1]
        r = self._perm_vector(r, x_new)
        label_i = self._remap_label(label_i, x_new)
        label_j = self._remap_label(label_j, x_new)

        # After permutation, label_i will always be px.
        # Otherwise, something must be wrong.
        if label_i != self._px:
            raise ValueError(f"Undefined label pair {label_i} {label_j}")

        # The minimal hopping table in the reference.
        l, m, n = self._eval_dir_cos(r)
        if label_j == self._px:
            t = l ** 2 * v_pps + (1 - l ** 2) * v_ppp
        elif label_j == self._py:
            t = l * m * (v_pps - v_ppp)
        else:
            t = l * n * (v_pps - v_ppp)
        return t

    def pd(self, r: np.ndarray,
           label_p: str = "px",
           label_d: str = "dxy",
           v_pds: complex = 0.0,
           v_pdp: complex = 0.0) -> complex:
        """
        Evaluate the hopping integral <p,0|H|d,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_p: label of p orbital
        :param label_d: label of d orbital
        :param v_pds: V_pd_sigma
        :param v_pdp: V_pd_pi
        :return: hopping integral
        :raises ValueError: if label_p is not in self.p_labels or
            label_d is not in self.d_labels
        """
        self._check_p_labels(label_p)
        self._check_d_labels(label_d)

        # Permute coordinates
        perm_labels = (self._dxy, self._dyz, self._dzx)
        if label_d in perm_labels:
            x_new = label_p[1]
            r = self._perm_vector(r, x_new)
            label_p = self._remap_label(label_p, x_new)
            label_d = self._remap_label(label_d, x_new)

        # Evaluate hopping integral
        l, m, n = self._eval_dir_cos(r)
        l2, m2, n2 = l ** 2, m ** 2, n ** 2
        l2_p_m2 = l2 + m2
        l2_m_m2 = l2 - m2
        sqrt3 = self._sqrt3
        sqrt3_2 = self._half_sqrt3

        if label_p == self._px:
            if label_d == self._dxy:
                t = sqrt3 * l2 * m * v_pds + m * (1 - 2 * l2) * v_pdp
            elif label_d == self._dyz:
                t = l * m * n * (sqrt3 * v_pds - 2 * v_pdp)
            elif label_d == self._dzx:
                t = sqrt3 * l2 * n * v_pds + n * (1 - 2 * l2) * v_pdp
            elif label_d == self._dx2_y2:
                t = sqrt3_2 * l * l2_m_m2 * v_pds + l * (1 - l2_m_m2) * v_pdp
            else:
                t = l * (n2 - 0.5 * l2_p_m2) * v_pds - sqrt3 * l * n2 * v_pdp
        elif label_p == self._py:
            if label_d == self._dx2_y2:
                t = sqrt3_2 * m * l2_m_m2 * v_pds - m * (1 + l2_m_m2) * v_pdp
            elif label_d == self._dz2:
                t = m * (n2 - 0.5 * l2_p_m2) * v_pds - sqrt3 * m * n2 * v_pdp
            else:
                raise ValueError(f"Undefined label pair {label_p} {label_d}")
        else:
            if label_d == self._dx2_y2:
                t = sqrt3_2 * n * l2_m_m2 * v_pds - n * l2_m_m2 * v_pdp
            elif label_d == self._dz2:
                t = n * (n2 - 0.5 * l2_p_m2) * v_pds + sqrt3 * n * l2_p_m2 * v_pdp
            else:
                raise ValueError(f"Undefined label pair {label_p} {label_d}")
        return t

    def dd(self, r: np.ndarray,
           label_i: str = "dxy",
           label_j: str = "dxy",
           v_dds: complex = 0.0,
           v_ddp: complex = 0.0,
           v_ddd: complex = 0) -> complex:
        """
        Evaluate the hopping integral <d_i,0|H|d_j,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_i: label of d_i orbital
        :param label_j: label of d_j orbital
        :param v_dds: V_dd_sigma
        :param v_ddp: V_dd_pi
        :param v_ddd: V_dd_delta
        :return: hopping integral
        :raises ValueError: if label_i or label_j is not in self.d_labels
        """
        self._check_d_labels(label_i, label_j)

        # Number the orbitals such that we can filter the diagonal terms
        # The order of the orbitals strictly follows the reference.
        # DO NOT CHANGE IT UNLESS YOU KNOW WHAT YOU ARE DOING!
        d_labels = (self._dxy, self._dyz, self._dzx, self._dx2_y2, self._dz2)
        id_i = d_labels.index(label_i)
        id_j = d_labels.index(label_j)

        if id_i > id_j:
            t = self.dd(r=-r, label_i=label_j, label_j=label_i, v_dds=v_dds,
                        v_ddp=v_ddp, v_ddd=v_ddd).conjugate()
        else:
            # Permute the coordinates if essential
            if label_i == self._dyz and label_j in (self._dyz, self._dzx):
                x_new = "y"
            elif label_i == self._dzx and label_j == self._dzx:
                x_new = "z"
            else:
                x_new = "x"
            r = self._perm_vector(r, x_new)
            label_i = self._remap_label(label_i, x_new)
            label_j = self._remap_label(label_j, x_new)

            # Evaluate hopping integral
            l, m, n = self._eval_dir_cos(r)
            l2, m2, n2 = l ** 2, m ** 2, n ** 2
            l2_p_m2 = l2 + m2
            l2_m_m2 = l2 - m2
            lm, mn, nl = l * m, m * n, n * l
            l2m2 = l2 * m2
            sqrt3 = self._sqrt3

            if label_i == self._dxy:
                if label_j == self._dxy:
                    factor = 1.0
                    t1 = 3 * l2m2
                    t2 = l2_p_m2 - 4 * l2m2
                    t3 = n2 + l2m2
                elif label_j == self._dyz:
                    factor = nl
                    t1 = 3 * m2
                    t2 = 1 - 4 * m2
                    t3 = m2 - 1
                elif label_j == self._dzx:
                    factor = mn
                    t1 = 3 * l2
                    t2 = 1 - 4 * l2
                    t3 = l2 - 1
                elif label_j == self._dx2_y2:
                    factor = lm * l2_m_m2
                    t1 = 1.5
                    t2 = -2
                    t3 = 0.5
                else:
                    factor = sqrt3 * lm
                    t1 = n2 - 0.5 * l2_p_m2
                    t2 = -2 * n2
                    t3 = 0.5 * (1 + n2)
            elif label_i == self._dyz:
                if label_j == self._dx2_y2:
                    factor = mn
                    t1 = 1.5 * l2_m_m2
                    t2 = -(1 + 2 * l2_m_m2)
                    t3 = 1 + 0.5 * l2_m_m2
                elif label_j == self._dz2:
                    factor = sqrt3 * mn
                    t1 = n2 - 0.5 * l2_p_m2
                    t2 = l2_p_m2 - n2
                    t3 = -0.5 * l2_p_m2
                else:
                    raise ValueError(f"Undefined label pair {label_i} {label_j}")
            elif label_i == self._dzx:
                if label_j == self._dx2_y2:
                    factor = nl
                    t1 = 1.5 * l2_m_m2
                    t2 = 1 - 2 * l2_m_m2
                    t3 = -1 * (1 - 0.5 * l2_m_m2)
                elif label_j == self._dz2:
                    factor = sqrt3 * nl
                    t1 = n2 - 0.5 * l2_p_m2
                    t2 = l2_p_m2 - n2
                    t3 = -0.5 * l2_p_m2
                else:
                    raise ValueError(f"Undefined label pair {label_i} {label_j}")
            elif label_i == self._dx2_y2:
                if label_j == self._dx2_y2:
                    factor = 1
                    t1 = 0.75 * l2_m_m2 ** 2
                    t2 = l2_p_m2 - l2_m_m2 ** 2
                    t3 = n2 + 0.25 * l2_m_m2 ** 2
                elif label_j == self._dz2:
                    factor = sqrt3 * l2_m_m2
                    t1 = 0.5 * (n2 - 0.5 * l2_p_m2)
                    t2 = -n2
                    t3 = 0.25 * (1 + n2)
                else:
                    raise ValueError(f"Undefined label pair {label_i} {label_j}")
            else:
                if label_j == self._dz2:
                    factor = 1
                    t1 = (n2 - 0.5 * l2_p_m2) ** 2
                    t2 = 3 * n2 * l2_p_m2
                    t3 = 0.75 * l2_p_m2 ** 2
                else:
                    raise ValueError(f"Undefined label pair {label_i} {label_j}")
            t = factor * (t1 * v_dds + t2 * v_ddp + t3 * v_ddd)
        return t

    def ps(self, r: np.ndarray,
           label_p: str = "px",
           v_sps: complex = 0.0) -> complex:
        """
        Evaluate the hopping integral <p,0|H|s,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_p: label of p orbital
        :param v_sps: V_sp_sigma
        :return: hopping integral
        :raises ValueError: if label_p is not in self.p_labels
        """
        return self.sp(r=-r, label_p=label_p, v_sps=v_sps).conjugate()

    def ds(self, r: np.ndarray,
           label_d: str = "dxy",
           v_sds: complex = 0.0) -> complex:
        """
        Evaluate the hopping integral <d,0|H|s,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_d: label of d orbital
        :param v_sds: V_sd_sigma
        :return: hopping integral
        :raises ValueError: if label_d is not in self.d_labels
        """
        return self.sd(r=-r, label_d=label_d, v_sds=v_sds).conjugate()

    def dp(self, r: np.ndarray,
           label_p: str = "px",
           label_d: str = "dxy",
           v_pds: complex = 0.0,
           v_pdp: complex = 0.0) -> complex:
        """
        Evaluate the hopping integral <d,0|H|p,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_p: label of p orbital
        :param label_d: label of d orbital
        :param v_pds: V_pd_sigma
        :param v_pdp: V_pd_pi
        :return: hopping integral
        :raises ValueError: if label_p is not in self.p_labels or
            label_d is not in self.d_labels
        """
        return self.pd(r=-r, label_p=label_p, label_d=label_d, v_pds=v_pds,
                       v_pdp=v_pdp).conjugate()

    def eval(self, r: np.ndarray,
             label_i: str = "s",
             label_j: str = "s",
             v_sss: complex = 0.0, v_sps: complex = 0.0, v_sds: complex = 0.0,
             v_pps: complex = 0.0, v_ppp: complex = 0.0,
             v_pds: complex = 0.0, v_pdp: complex = 0.0,
             v_dds: complex = 0.0, v_ddp: complex = 0.0, v_ddd: complex = 0.0) -> complex:
        """
        Common interface for evaluating hopping integral <i,0|H|j,r>.

        :param r: Cartesian coordinates of the displacement vector
        :param label_i: label for orbital i
        :param label_j: label for orbital j
        :param v_sss: V_ss_sigma
        :param v_sps: V_sp_sigma
        :param v_sds: V_sd_sigma
        :param v_pps: V_pp_sigma
        :param v_ppp: V_pp_pi
        :param v_pds: V_pd_sigma
        :param v_pdp: V_pd_pi
        :param v_dds: V_dd_sigma
        :param v_ddp: V_dd_pi
        :param v_ddd: V_dd_delta
        :return: hopping integral
        :raises ValueError: if label_i or label_j is not in predefined labels
        """
        if label_i == self._s:
            if label_j == self._s:
                t = self.ss(v_sss=v_sss)
            elif label_j in self._p_labels:
                t = self.sp(r=r, label_p=label_j, v_sps=v_sps)
            else:
                t = self.sd(r=r, label_d=label_j, v_sds=v_sds)
        elif label_i in self._p_labels:
            if label_j == self._s:
                t = self.ps(r=r, label_p=label_i, v_sps=v_sps)
            elif label_j in self._p_labels:
                t = self.pp(r=r, label_i=label_i, label_j=label_j,
                            v_pps=v_pps, v_ppp=v_ppp)
            else:
                t = self.pd(r=r, label_p=label_i, label_d=label_j,
                            v_pds=v_pds, v_pdp=v_pdp)
        else:
            if label_j == self._s:
                t = self.ds(r=r, label_d=label_i, v_sds=v_sds)
            elif label_j in self._p_labels:
                t = self.dp(r=r, label_d=label_i, label_p=label_j,
                            v_pds=v_pds, v_pdp=v_pdp)
            else:
                t = self.dd(r=r, label_i=label_i, label_j=label_j,
                            v_dds=v_dds, v_ddp=v_ddp, v_ddd=v_ddd)
        return t


class AtomicOrbital:
    """
    Class representing an atomic orbital composed of multiple |l,m> states.

    Attributes
    ----------
    _l_max: int
        maximum angular quantum number l
    _coeff: Dict[Tuple[int, int], complex]
        keys: (l, m), values: coefficients on |l,m>
    _allowed_qn: Set[Tuple[int, int]]
        set of allowed (l, m) pairs
    """
    def __init__(self, l_max: int = 2) -> None:
        """
        :param l_max: maximum angular quantum number l
        """
        self._l_max = l_max
        self._coeff = self._init_coeff()
        self._allowed_qn = set(self._coeff.keys())

    def __setitem__(self, key: Tuple[int, int], value: complex = 1.0) -> None:
        """
        Set the coefficient on state |l,m>.

        :param key: (l, m) of the state
        :param value: the new coefficient
        :return: None
        """
        if key not in self._allowed_qn:
            raise KeyError(f"Undefined key {key}")
        self._coeff[key] = value

    def __getitem__(self, key: Tuple[int, int]) -> complex:
        """
        Get the coefficient on state |l,m>.

        :param key: (l, m) of the state
        :return: the coefficient
        """
        if key not in self._allowed_qn:
            raise KeyError(f"Undefined key {key}")
        return self._coeff[key]

    def _init_coeff(self) -> Dict[Tuple[int, int], complex]:
        """
        Build initial coefficients.

        :return: dictionary with keys being (l, m) and values being zero
        """
        coefficients = dict()
        for l_i in range(self._l_max + 1):
            for m in range(-l_i, l_i+1):
                coefficients[(l_i, m)] = 0.0
        return coefficients

    def l_plus(self) -> None:
        """
        Apply the l+ operator on the atomic orbital.

        As the formula is
            l+|l,m> = sqrt((l - m) * (l + m + 1)) * h_bar * |l,m+1>
        the actual coefficient should be multiplied by a factor of h_bar.

        :return: None
        """
        new_coefficients = self._init_coeff()
        for key, value in self._coeff.items():
            l, m = key
            key_new = (l, m+1)
            if key_new in self._allowed_qn:
                factor = sqrt((l - m) * (l + m + 1))
                new_coefficients[key_new] = value * factor
        self._coeff = new_coefficients

    def l_minus(self) -> None:
        """
        Apply the l- operator on the atomic orbital.

        As the formula is
            l-|l,m> = sqrt((l + m) * (l - m + 1)) * h_bar * |l,m-1>.
        the actual coefficient should be multiplied by a factor of h_bar.

        :return: None
        """
        new_coefficients = self._init_coeff()
        for key, value in self._coeff.items():
            l, m = key
            key_new = (l, m-1)
            if key_new in self._allowed_qn:
                factor = sqrt((l + m) * (l - m + 1))
                new_coefficients[key_new] = value * factor
        self._coeff = new_coefficients

    def l_z(self) -> None:
        """
        Apply the lz operator on the atomic orbital.

        As the formula is
            lz|l,m> = m * h_bar * |l,m>
        the actual coefficient should be multiplied by a factor of h_bar.

        :return: None
        """
        for key in self._coeff.keys():
            m = key[1]
            self._coeff[key] *= m

    def product(self, ket) -> complex:
        """
        Evaluate the inner product <self|ket>.

        :param ket: the ket vector
        :return: the inner product
        """
        product = 0.0
        for key, value in self._coeff.items():
            product += value.conjugate() * ket[key]
        return product

    def mtxel(self, ket, operator: str) -> complex:
        """
        Evaluate the matrix element <self|operator|ket>.

        :param ket: the ket vector
        :param operator: the operator
        :return: the matrix element
        :raises ValueError: if the operator is not in l+, l-, lz
        """
        ket_copy = deepcopy(ket)
        if operator == "l+":
            ket_copy.l_plus()
        elif operator == "l-":
            ket_copy.l_minus()
        elif operator == "lz":
            ket_copy.l_z()
        else:
            raise ValueError(f"Illegal operator {operator}")
        return self.product(ket_copy)


class SOC:
    """
    Class for evaluating spin-orbital coupling terms.

    Attributes
    ----------
    _orbital_basis: Dict[str, AtomicOrbital]
        collection of atomic orbitals s, px, py, pz. etc
    _orbital_labels: Set[str]
        labels of atomic orbitals
    _spin_labels: Set[str]
        directions of spins
    """
    def __init__(self) -> None:
        self._orbital_basis = dict()
        orbital_labels = ("s", "px", "py", "pz", "dxy", "dx2-y2", "dyz", "dzx",
                          "dz2")
        for label in orbital_labels:
            self._orbital_basis[label] = AtomicOrbital(l_max=2)
        self._orbital_labels = set(self._orbital_basis.keys())
        self._spin_labels = {"up", "down"}

        # Reference:
        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
        c = sqrt(0.5)
        ci = c * 1j

        # s state
        self._orbital_basis["s"][(0, 0)] = 1.0

        # p states
        self._orbital_basis["py"][(1, -1)] = ci
        self._orbital_basis["py"][(1, 1)] = ci
        self._orbital_basis["pz"][(1, 0)] = 1.0
        self._orbital_basis["px"][(1, -1)] = c
        self._orbital_basis["px"][(1, 1)] = -c

        # d states
        self._orbital_basis["dxy"][(2, -2)] = ci
        self._orbital_basis["dxy"][(2, 2)] = -ci
        self._orbital_basis["dyz"][(2, -1)] = ci
        self._orbital_basis["dyz"][(2, 1)] = ci
        self._orbital_basis["dz2"][(2, 0)] = 1.0
        self._orbital_basis["dzx"][(2, -1)] = c
        self._orbital_basis["dzx"][(2, 1)] = -c
        self._orbital_basis["dx2-y2"][(2, -2)] = c
        self._orbital_basis["dx2-y2"][(2, 2)] = c

    @staticmethod
    def _eval_soc(bra: AtomicOrbital,
                  ket: AtomicOrbital,
                  spin_bra: str,
                  spin_ket: str) -> complex:
        """
        Evaluate the matrix element <bra|l*s|ket>.

        The SOC term is evaluated as 0.5 * h_bar * <bra|A|ket>, where A is
        either lz, l-, l+ and -lz depending on the spins of atomic orbitals.

        The result has a factor of 0.5 * h_bar from the spin part, and a factor
        of h_bar from the orbital part. Since we further multiply the result by
        0.5, the final factor becomes h_bar**2

        :param bra: left operand atomic orbital
        :param ket: right operand atomic orbital
        :param spin_bra: spin of bra
        :param spin_ket: spin of ket
        :return: soc term in h_bar**2
        """
        spin_idx = (spin_bra, spin_ket)
        if spin_idx == ("up", "up"):
            soc = 0.5 * bra.mtxel(ket, "lz")
        elif spin_idx == ("up", "down"):
            soc = 0.5 * bra.mtxel(ket, "l-")
        elif spin_idx == ("down", "up"):
            soc = 0.5 * bra.mtxel(ket, "l+")
        else:
            soc = -0.5 * bra.mtxel(ket, "lz")
        return soc

    def print_soc_table(self, spin_i: str = "up", spin_j: str = "up") -> None:
        """
        Print SOC terms between orbital basis functions, for generating the
        coefficients in 'SOCTable' class.

        :param spin_i: spin direction of bra
        :param spin_j: spin direction kf ket
        :return: None
        :raises ValueError: if spin directions are illegal
        """
        for spin in (spin_i, spin_j):
            if spin not in self._spin_labels:
                raise ValueError(f"Illegal spin direction {spin}")
        soc_table = dict()
        for label_bra, bra in self._orbital_basis.items():
            for label_ket, ket in self._orbital_basis.items():
                soc = self._eval_soc(bra, ket, spin_i, spin_j)
                if abs(soc) > 1.0e-5:
                    soc_table[(label_bra, label_ket)] = soc
        print("Factor: h_bar**2")
        print(soc_table)

    def eval(self, label_i: str = "s",
             spin_i: str = "up",
             label_j: str = "s",
             spin_j: str = "down") -> complex:
        """
        Evaluate the matrix element <i,s_i|l*s|j,s_j>.

        :param label_i: orbital label of bra
        :param spin_i: spin direction of bra
        :param label_j: orbital label of ket
        :param spin_j: spin direction of ket
        :return: matrix element in h_bar**2
        :raises ValueError: if orbital labels or spin directions are illegal
        """
        for label in (label_i, label_j):
            if label not in self._orbital_labels:
                raise ValueError(f"Illegal orbital label {label}")
        for spin in (spin_i, spin_j):
            if spin not in self._spin_labels:
                raise ValueError(f"Illegal spin direction {spin}")
        bra = self._orbital_basis[label_i]
        ket = self._orbital_basis[label_j]
        soc = self._eval_soc(bra, ket, spin_i, spin_j)
        return soc


class SOCTable:
    """
    Hard-coded spin-orbital coupling term table, generated using the
    'print_soc_table' method of 'SOC' class.

    Attributes
    ----------
    _up_up: Dict[[str, str], complex]
        soc terms for up-up spin chanel
    _up_down: Dict[[str, str], complex]
        soc terms for up-down spin chanel
    _down_up: Dict[[str, str], complex]
        soc terms for down-up spin chanel
    _orbital_labels: Set[str]
        labels of atomic orbitals
    _spin_labels: Set[str]
        directions of spins
    """
    def __init__(self):
        self._up_up = {('px', 'py'): -0.5000000000000001j,
                       ('py', 'px'): 0.5000000000000001j,
                       ('dxy', 'dx2-y2'): 1.0000000000000002j,
                       ('dx2-y2', 'dxy'): -1.0000000000000002j,
                       ('dyz', 'dzx'): 0.5000000000000001j,
                       ('dzx', 'dyz'): -0.5000000000000001j}
        self._up_down = {('px', 'pz'): 0.5000000000000001,
                         ('py', 'pz'): -0.5000000000000001j,
                         ('pz', 'px'): -0.5000000000000001,
                         ('pz', 'py'): 0.5000000000000001j,
                         ('dxy', 'dyz'): (0.5000000000000001+0j),
                         ('dxy', 'dzx'): -0.5000000000000001j,
                         ('dx2-y2', 'dyz'): 0.5000000000000001j,
                         ('dx2-y2', 'dzx'): 0.5000000000000001,
                         ('dyz', 'dxy'): (-0.5000000000000001+0j),
                         ('dyz', 'dx2-y2'): -0.5000000000000001j,
                         ('dyz', 'dz2'): -0.8660254037844386j,
                         ('dzx', 'dxy'): 0.5000000000000001j,
                         ('dzx', 'dx2-y2'): -0.5000000000000001,
                         ('dzx', 'dz2'): 0.8660254037844386,
                         ('dz2', 'dyz'): 0.8660254037844386j,
                         ('dz2', 'dzx'): -0.8660254037844386}
        self._down_up = {('px', 'pz'): -0.5000000000000001,
                         ('py', 'pz'): -0.5000000000000001j,
                         ('pz', 'px'): 0.5000000000000001,
                         ('pz', 'py'): 0.5000000000000001j,
                         ('dxy', 'dyz'): (-0.5000000000000001+0j),
                         ('dxy', 'dzx'): -0.5000000000000001j,
                         ('dx2-y2', 'dyz'): 0.5000000000000001j,
                         ('dx2-y2', 'dzx'): -0.5000000000000001,
                         ('dyz', 'dxy'): (0.5000000000000001+0j),
                         ('dyz', 'dx2-y2'): -0.5000000000000001j,
                         ('dyz', 'dz2'): -0.8660254037844386j,
                         ('dzx', 'dxy'): 0.5000000000000001j,
                         ('dzx', 'dx2-y2'): 0.5000000000000001,
                         ('dzx', 'dz2'): -0.8660254037844386,
                         ('dz2', 'dyz'): 0.8660254037844386j,
                         ('dz2', 'dzx'): 0.8660254037844386}
        self._orbital_labels = {"s", "px", "py", "pz", "dxy", "dx2-y2", "dyz",
                                "dzx", "dz2"}
        self._spin_labels = {"up", "down"}

    def eval(self, label_i: str = "s",
             spin_i: str = "up",
             label_j: str = "s",
             spin_j: str = "down") -> complex:
        """
        Evaluate the matrix element <i,s_i|l*s|j,s_j>.

        :param label_i: orbital label of bra
        :param spin_i: spin direction of bra
        :param label_j: orbital label of ket
        :param spin_j: spin direction of ket
        :return: matrix element in h_bar**2
        :raises ValueError: if orbital labels or spin directions are illegal
        """
        label_idx = (label_i, label_j)
        spin_idx = (spin_i, spin_j)
        for label in label_idx:
            if label not in self._orbital_labels:
                raise ValueError(f"Illegal orbital label {label}")
        for spin in spin_idx:
            if spin not in self._spin_labels:
                raise ValueError(f"Illegal spin direction {spin}")
        try:
            if spin_idx == ("up", "up"):
                product = self._up_up[label_idx]
            elif spin_idx == ("up", "down"):
                product = self._up_down[label_idx]
            elif spin_idx == ("down", "up"):
                product = self._down_up[label_idx]
            else:
                product = -1 * self._up_up[label_idx]
        except KeyError:
            product = 0.0
        return product


class ParamFit(ABC):
    """
    Class for fitting on-site energies and hopping parameters to reference
    band data.

    Attributes
    ----------
    _k_points: (num_kpt, 3) float64 array
        FRACTIONAL coordinates of the k-points corresponding to the band data
    _bands_ref: (num_kpt, num_orb) float64 array
        copy of reference band data
    _weights: (num_orb,) float64 array
        weights of each band during fitting process
    """
    def __init__(self, k_points: np.ndarray,
                 weights: np.ndarray = None) -> None:
        """
        :param k_points: (num_kpt, 3) float64 array
            FRACTIONAL coordinates of the k-points
        :param weights: (num_orb,) float64 array
            weights of each band for fitting
        :raises ValueError: if length of weights does not match band data
        """
        self._k_points = k_points
        self._bands_ref = self.calc_bands_ref()
        num_bands = self._bands_ref.shape[1]
        if weights is None:
            self._weights = np.ones(num_bands)
        else:
            if weights.shape[0] != num_bands:
                raise ValueError(f"Length of weights should be {num_bands}")
            weights = np.abs(weights)
            weights /= weights.sum()
            self._weights = weights

    @abstractmethod
    def calc_bands_ref(self) -> np.ndarray:
        """
        Method for calculating reference band data.

        :return: (num_kpt, num_orb) float64 array
            reference band data
        """
        pass

    @abstractmethod
    def calc_bands_fit(self, params: np.ndarray) -> np.ndarray:
        """
        Method for calculating fitted band data from given parameters.

        :param params: parameters to fit
        :return: (num_kpt, num_orb) float64 array
            band data of the model built from parameters
        """
        pass

    def estimate_error(self, params: np.ndarray) -> np.ndarray:
        """
        Object function for minimizing the error between reference and fitted
        band data.

        :param params: parameters to fit
        :return: flattened difference between band data
        """
        bands_fit = self.calc_bands_fit(params)
        bands_diff = self._bands_ref - bands_fit
        for i, w in enumerate(self._weights):
            bands_diff[:, i] *= w
        return bands_diff.flatten()

    def fit(self, params0: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit the parameters to reference band data.

        :param params0: initial value of parameters to fit
        :param kwargs: keyword parameters for the 'leastsq' function of scipy
        :return: the optimal parameters after fitting
        :raises RuntimeError: if the fitting fails
        """
        result = leastsq(self.estimate_error, params0, **kwargs)
        params, status = result[0], result[1]
        if status not in (1, 2, 3, 4):
            raise RuntimeError(f"Fitting failed with status {status}")
        return params
