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
    find_neighbors: user function
        searching for neighbours between the (0, 0, 0) cell of pc_bra and nearby
        cells of pc_ket up to given cutoff distance.

Classes
-------
    PCInterHopping: user class
        container for holding hopping terms between different primitive cells in
        hetero-structure
    SK: user class
        for evaluating hopping integrals using Slater-Koster formula
"""

import math
from typing import Union, List, Tuple
from collections import namedtuple

import numpy as np
from scipy.spatial import KDTree

from . import constants as consts
from . import exceptions as exc
from .lattice import cart2frac, rotate_coord
from .base import check_coord, InterHopping
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
    dim, legal = check_coord(dim, complete_item=1)
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


def spiral_prim_cell(prim_cell: PrimitiveCell, angle=0.0,
                     center=np.zeros(3), shift=0.0):
    """
    Rotate and shift primitive cell with respect to z-axis.

    :param prim_cell: instance of 'PrimitiveCell' class
        primitive cell to twist
    :param angle: float
        twisting angle in RADIANs, NOT degrees
    :param center: (3,) float64 array
        Cartesian coordinates of the rotation center in NANOMETER
    :param shift: float
        distance of shift in NANOMETER
    :return: None
        Incoming primitive cell is modified.
    """
    prim_cell.sync_array()

    # Get rotated lattice vectors
    end_points = np.vstack((np.zeros(3), prim_cell.lat_vec))
    end_points = rotate_coord(end_points, angle=angle, center=center)
    lat_vec = end_points[1:] - end_points[0]

    # Get rotated orbital positions
    # CAUTION: DO NOT normalize the positions after rotation.
    # They should be kept as they are.
    orb_pos = prim_cell.orb_pos_nm
    orb_pos = rotate_coord(orb_pos, angle=angle, center=center)

    # Shift orbital positions
    orb_pos[:, 2] += shift

    # Update lattice vectors and orbital positions
    prim_cell.lat_vec = lat_vec
    orb_pos = cart2frac(lat_vec, orb_pos)
    for i, pos in enumerate(orb_pos):
        prim_cell.set_orbital(i, position=tuple(pos))
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


def find_neighbors(pc_bra: PrimitiveCell, pc_ket: PrimitiveCell = None,
                   a_max: int = 0, b_max: int = 0, c_max: int = 0,
                   max_distance: float = 1.0) -> List[namedtuple]:
    """
    Find neighbours between the (0, 0, 0) cell of pc_bra and nearby cells of
    pc_ket up to given cutoff distance.

    The searching range of nearby cells is:
    [-a_max, a_max] * [-b_max, b_max] * [-c_max, c_max].

    :param pc_bra: the 'bra' primitive cell
    :param pc_ket: the 'ket' primitive cell, default to pc_ket if not set
    :param a_max: upper bound of range on a-axis
    :param b_max: upper bound of range on b-axis
    :param c_max: upper bound of range on c-axis
    :param max_distance: cutoff distance in NM
    :return: list of neighbors as named tuples
    """
    if pc_ket is None:
        pc_ket = pc_bra

    # Get orbital positions
    pc_bra.sync_array()
    pc_ket.sync_array()
    pos_bra = pc_bra.orb_pos_nm
    pos_ket = pc_ket.orb_pos_nm

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
        pos_ket_rn = pos_ket + np.matmul(rn, pc_ket.lat_vec)
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
        self.s = "s"
        self.px = "px"
        self.py = "py"
        self.pz = "pz"
        self.dxy = "dxy"
        self.dyz = "dyz"
        self.dzx = "dzx"
        self.dx2_y2 = "dx2-y2"
        self.dz2 = "dz2"
        self.p_labels = {self.px, self.py, self.pz}
        self.d_labels = {self.dxy, self.dyz, self.dzx, self.dx2_y2, self.dz2}
        self.sqrt3 = math.sqrt(3)
        self.half_sqrt3 = self.sqrt3 * 0.5

    def _check_p_labels(self, *labels: str) -> None:
        """
        Check the sanity of labels of p orbitals.

        :param labels: labels to check
        :raises ValueError: if any label is not in self.p_labels
        """
        for label in labels:
            if label not in self.p_labels:
                raise ValueError(f"Illegal label: {label}")

    def _check_d_labels(self, *labels: str) -> None:
        """
        Check the sanity of labels of d orbitals.

        :param labels: labels to check
        :raises ValueError: if any label is not in self.d_labels
        """
        for label in labels:
            if label not in self.d_labels:
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

    def sp(self, r: np.ndarray, label_p: str = "px",
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
        if label_p == self.px:
            t = l * v_sps
        elif label_p == self.py:
            t = m * v_sps
        else:
            t = n * v_sps
        return t

    def sd(self, r: np.ndarray, label_d: str = "dxy",
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
        if label_d == self.dyz:
            x_new = "y"
        elif label_d == self.dzx:
            x_new = "z"
        else:
            x_new = "x"
        r = self._perm_vector(r, x_new)
        label_d = self._remap_label(label_d, x_new)

        # Evaluate the hopping integral
        l, m, n = self._eval_dir_cos(r)
        if label_d == self.dxy:
            t = self.sqrt3 * l * m * v_sds
        elif label_d == self.dx2_y2:
            t = self.half_sqrt3 * (l ** 2 - m ** 2) * v_sds
        elif label_d == self.dz2:
            t = (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * v_sds
        else:
            raise ValueError(f"Undefined label pair s {label_d}")
        return t

    def pp(self, r: np.ndarray, label_i: str = "px", label_j: str = "px",
           v_pps: complex = 0.0, v_ppp: complex = 0.0) -> complex:
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
        if label_i != self.px:
            raise ValueError(f"Undefined label pair {label_i} {label_j}")

        # The minimal hopping table in the reference.
        l, m, n = self._eval_dir_cos(r)
        if label_j == self.px:
            t = l ** 2 * v_pps + (1 - l ** 2) * v_ppp
        elif label_j == self.py:
            t = l * m * (v_pps - v_ppp)
        else:
            t = l * n * (v_pps - v_ppp)
        return t

    def pd(self, r: np.ndarray, label_p: str = "px", label_d: str = "dxy",
           v_pds: complex = 0.0, v_pdp: complex = 0.0) -> complex:
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
        perm_labels = (self.dxy, self.dyz, self.dzx)
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
        sqrt3 = self.sqrt3
        sqrt3_2 = self.half_sqrt3

        if label_p == self.px:
            if label_d == self.dxy:
                t = sqrt3 * l2 * m * v_pds + m * (1 - 2 * l2) * v_pdp
            elif label_d == self.dyz:
                t = l * m * n * (sqrt3 * v_pds - 2 * v_pdp)
            elif label_d == self.dzx:
                t = sqrt3 * l2 * n * v_pds + n * (1 - 2 * l2) * v_pdp
            elif label_d == self.dx2_y2:
                t = sqrt3_2 * l * l2_m_m2 * v_pds + l * (1 - l2_m_m2) * v_pdp
            else:
                t = l * (n2 - 0.5 * l2_p_m2) * v_pds - sqrt3 * l * n2 * v_pdp
        elif label_p == self.py:
            if label_d == self.dx2_y2:
                t = sqrt3_2 * m * l2_m_m2 * v_pds - m * (1 + l2_m_m2) * v_pdp
            elif label_d == self.dz2:
                t = m * (n2 - 0.5 * l2_p_m2) * v_pds - sqrt3 * m * n2 * v_pdp
            else:
                raise ValueError(f"Undefined label pair {label_p} {label_d}")
        else:
            if label_d == self.dx2_y2:
                t = sqrt3_2 * n * l2_m_m2 * v_pds - n * l2_m_m2 * v_pdp
            elif label_d == self.dz2:
                t = n * (n2 - 0.5 * l2_p_m2) * v_pds + sqrt3 * n * l2_p_m2 * v_pdp
            else:
                raise ValueError(f"Undefined label pair {label_p} {label_d}")
        return t

    def dd(self, r: np.ndarray, label_i: str = "dxy", label_j: str = "dxy",
           v_dds: complex = 0.0, v_ddp: complex = 0.0, v_ddd: complex = 0) -> complex:
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
        d_labels = (self.dxy, self.dyz, self.dzx, self.dx2_y2, self.dz2)
        id_i = d_labels.index(label_i)
        id_j = d_labels.index(label_j)

        if id_i > id_j:
            t = self.dd(r=-r, label_i=label_j, label_j=label_i, v_dds=v_dds,
                        v_ddp=v_ddp, v_ddd=v_ddd).conjugate()
        else:
            # Permute the coordinates if essential
            if label_i == self.dyz and label_j in (self.dyz, self.dzx):
                x_new = "y"
            elif label_i == self.dzx and label_j == self.dzx:
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
            sqrt3 = self.sqrt3

            if label_i == self.dxy:
                if label_j == self.dxy:
                    factor = 1.0
                    t1 = 3 * l2m2
                    t2 = l2_p_m2 - 4 * l2m2
                    t3 = n2 + l2m2
                elif label_j == self.dyz:
                    factor = nl
                    t1 = 3 * m2
                    t2 = 1 - 4 * m2
                    t3 = m2 - 1
                elif label_j == self.dzx:
                    factor = mn
                    t1 = 3 * l2
                    t2 = 1 - 4 * l2
                    t3 = l2 - 1
                elif label_j == self.dx2_y2:
                    factor = lm * l2_m_m2
                    t1 = 1.5
                    t2 = -2
                    t3 = 0.5
                else:
                    factor = sqrt3 * lm
                    t1 = n2 - 0.5 * l2_p_m2
                    t2 = -2 * n2
                    t3 = 0.5 * (1 + n2)
            elif label_i == self.dyz:
                if label_j == self.dx2_y2:
                    factor = mn
                    t1 = 1.5 * l2_m_m2
                    t2 = -(1 + 2 * l2_m_m2)
                    t3 = 1 + 0.5 * l2_m_m2
                elif label_j == self.dz2:
                    factor = sqrt3 * mn
                    t1 = n2 - 0.5 * l2_p_m2
                    t2 = l2_p_m2 - n2
                    t3 = -0.5 * l2_p_m2
                else:
                    raise ValueError(f"Undefined label pair {label_i} {label_j}")
            elif label_i == self.dzx:
                if label_j == self.dx2_y2:
                    factor = nl
                    t1 = 1.5 * l2_m_m2
                    t2 = 1 - 2 * l2_m_m2
                    t3 = -1 * (1 - 0.5 * l2_m_m2)
                elif label_j == self.dz2:
                    factor = sqrt3 * nl
                    t1 = n2 - 0.5 * l2_p_m2
                    t2 = l2_p_m2 - n2
                    t3 = -0.5 * l2_p_m2
                else:
                    raise ValueError(f"Undefined label pair {label_i} {label_j}")
            elif label_i == self.dx2_y2:
                if label_j == self.dx2_y2:
                    factor = 1
                    t1 = 0.75 * l2_m_m2 ** 2
                    t2 = l2_p_m2 - l2_m_m2 ** 2
                    t3 = n2 + 0.25 * l2_m_m2 ** 2
                elif label_j == self.dz2:
                    factor = sqrt3 * l2_m_m2
                    t1 = 0.5 * (n2 - 0.5 * l2_p_m2)
                    t2 = -n2
                    t3 = 0.25 * (1 + n2)
                else:
                    raise ValueError(f"Undefined label pair {label_i} {label_j}")
            else:
                if label_j == self.dz2:
                    factor = 1
                    t1 = (n2 - 0.5 * l2_p_m2) ** 2
                    t2 = 3 * n2 * l2_p_m2
                    t3 = 0.75 * l2_p_m2 ** 2
                else:
                    raise ValueError(f"Undefined label pair {label_i} {label_j}")
            t = factor * (t1 * v_dds + t2 * v_ddp + t3 * v_ddd)
        return t

    def ps(self, r: np.ndarray, label_p: str = "px",
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

    def ds(self, r: np.ndarray, label_d: str = "dxy",
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

    def dp(self, r: np.ndarray, label_p: str = "px", label_d: str = "dxy",
           v_pds: complex = 0.0, v_pdp: complex = 0.0) -> complex:
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

    def eval(self, r: np.ndarray, label_i: str = "s", label_j: str = "s",
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
        if label_i == self.s:
            if label_j == self.s:
                t = self.ss(v_sss=v_sss)
            elif label_j in self.p_labels:
                t = self.sp(r=r, label_p=label_j, v_sps=v_sps)
            else:
                t = self.sd(r=r, label_d=label_j, v_sds=v_sds)
        elif label_i in self.p_labels:
            if label_j == self.s:
                t = self.ps(r=r, label_p=label_i, v_sps=v_sps)
            elif label_j in self.p_labels:
                t = self.pp(r=r, label_i=label_i, label_j=label_j,
                            v_pps=v_pps, v_ppp=v_ppp)
            else:
                t = self.pd(r=r, label_p=label_i, label_d=label_j,
                            v_pds=v_pds, v_pdp=v_pdp)
        else:
            if label_j == self.s:
                t = self.ds(r=r, label_d=label_i, v_sds=v_sds)
            elif label_j in self.p_labels:
                t = self.dp(r=r, label_d=label_i, label_p=label_j,
                            v_pds=v_pds, v_pdp=v_pdp)
            else:
                t = self.dd(r=r, label_i=label_i, label_j=label_j,
                            v_dds=v_dds, v_ddp=v_ddp, v_ddd=v_ddd)
        return t
