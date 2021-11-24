"""
Functions and classes for manipulating samples.

Functions
---------
    extend_prim_cell: user API
        extend primitive cell along a, b, and c directions
        reserved for compatibility with old version of TBPlaS
    reshape_prim_cell: user API
        reshape primitive cell to given lattice vectors and origin

Classes
-------
    HopDict: user class
        class for holding hopping terms
        reserved for compatibility with old version of TBPlaS
"""

import math

import numpy as np

from . import constants as consts
from . import exceptions as exc
from .primitive import correct_coord, PrimitiveCell


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
    extend_cell.extended *= np.prod(dim)
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
                    extend_cell.add_orbital(pos_ext, orbital.energy)

    # Define periodic boundary condition.
    def _wrap_pbc(ji, ni):
        return ji % ni, ji // ni

    # Add hopping terms
    for id_sc_i in range(extend_cell.num_orb):
        id_pc_i = orb_id_pc[id_sc_i]
        for hopping in prim_cell.hopping_list:
            hop_ind = hopping.index
            if id_pc_i[3] == hop_ind[3]:
                ja, na = _wrap_pbc(id_pc_i[0] + hop_ind[0], dim[0])
                jb, nb = _wrap_pbc(id_pc_i[1] + hop_ind[1], dim[1])
                jc, nc = _wrap_pbc(id_pc_i[2] + hop_ind[2], dim[2])
                id_pc_j = (ja, jb, jc, hop_ind[4])
                id_sc_j = orb_id_sc[id_pc_j]
                rn = (na, nb, nc)
                extend_cell.add_hopping(rn, id_sc_i, id_sc_j, hopping.energy)

    extend_cell.sync_array()
    return extend_cell


def reshape_prim_cell(prim_cell: PrimitiveCell, lat_frac: np.ndarray,
                      origin: np.ndarray = np.zeros(3),
                      delta=0.01, pos_tol=1e-5):
    """
    Reshape primitive cell to given lattice vectors and origin.

    :param prim_cell: instance of 'PrimitiveCell' class
        primitive cell from which the reshaped cell is constructed
    :param lat_frac: (3, 3) int32 array
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
    :raises LatVecError: if lat_frac.shape != (3, 3)
    :raises ValueError: if origin.shape != (3,)
    """
    # Check lattice vectors and origin
    lat_frac = np.array(lat_frac, dtype=np.int32)
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

    # Add orbitals
    prim_cell.sync_array()
    rn_range = np.zeros((3, 3), dtype=np.int32)
    for i_dim in range(3):
        rn_range[i_dim, 0] = lat_frac[:, i_dim].min() - 1
        rn_range[i_dim, 1] = lat_frac[:, i_dim].max() + 1

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
                                             prim_cell.orb_eng.item(i_o))

    # Add hopping terms
    res_cell.sync_array()
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
                for id_pc_j in orb_id_pc:
                    if id_pc_j[3] == hop.item(4):
                        id_sc_j = orb_id_sc[id_pc_j]
                        chk_pos = res_cell.orb_pos[id_sc_j]
                        if np.linalg.norm(chk_pos - res_pos) <= pos_tol:
                            res_cell.add_hopping(res_rn, id_sc_i, id_sc_j,
                                                 prim_cell.hop_eng[i_h])

    # Subtract delta from orbital positions
    res_cell.orb_pos -= delta
    for i_o, pos in enumerate(res_cell.orb_pos):
        res_cell.orbital_list[i_o].position = tuple(pos)
    res_cell.sync_array()
    return res_cell


class HopDict:
    """
    Class for holding hopping terms.

    Reserved for compatibility with old version of TBPlaS.

    Attributes
    ----------
    dict: dictionary
        dictionary with site tags as keys and complex matrices as values
    mat_shape: (num_orb, num_orb)
        shape of hopping matrices
    """
    def __init__(self, num_orb):
        """
        Initialize hop_dict object.

        :param num_orb: integer
            number of orbitals
        """
        self.dict = {}
        self.mat_shape = (num_orb, num_orb)

    @staticmethod
    def __get_minus_rn(rn):
        """Get minus cell index."""
        return tuple([-v for v in rn])

    def set_mat(self, rn, hop_mat: np.ndarray):
        """
        Add hopping matrix to dictionary or update an existing hopping matrix.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :param hop_mat: (num_orb, num_orb) complex128 array
            hopping matrix
        :raises CellIndexLenError: if len(rn) != 2 or 3
        :raises ValueError: if hop_mat.shape does not match number of orbitals
            or hopping matrix at (0, 0, 0) is not Hermitian
        :raises PCHopDiagonalError: if on-site energies are included in hopping
            matrix
        """
        # Check cell index
        try:
            rn = correct_coord(rn)
        except exc.CoordLenError as err:
            raise exc.CellIndexLenError(err.coord) from err

        # Check matrix size
        hop_mat = np.array(hop_mat)
        if hop_mat.shape != self.mat_shape:
            raise ValueError(f"Shape of hopping matrix {hop_mat.shape} does not "
                             f"match {self.mat_shape}")

        # Special check for (0, 0, 0) cell
        if rn == (0, 0, 0):
            for i in range(hop_mat.shape[0]):
                if abs(hop_mat.item(i, i)) >= 1e-5:
                    raise exc.PCHopDiagonalError(rn, i)
                for j in range(i+1, hop_mat.shape[0]):
                    h_ij = hop_mat.item(i, j)
                    h_ji = hop_mat.item(j, i)
                    if abs(h_ij - h_ji.conjugate()) >= 1e-5:
                        raise ValueError(f"Hopping matrix at (0, 0, 0) is not "
                                         f"Hermitian")

        # Set hopping matrix
        if rn in self.dict.keys():
            self.dict[rn] = hop_mat
        else:
            minus_rn = self.__get_minus_rn(rn)
            if minus_rn in self.dict.keys():
                self.dict[minus_rn] = hop_mat.T.conj()
            else:
                self.dict[rn] = hop_mat

    def set_zero_mat(self, rn):
        """
        Add zero hopping matrix to dictionary.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        zero_mat = np.zeros(self.mat_shape, dtype=complex)
        self.set_mat(rn, zero_mat)

    def set_element(self, rn, element, hop):
        """
        Add single hopping to hopping matrix.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :param element: (orb_i, orb_j)
            element indices
        :param hop: complex float
            hopping value
        :raises CellIndexLenError: if len(rn) != 2 or 3
        :raises ValueError: if element indices are out of range
        :raises PCHopDiagonalError: if on-site energy is given as input
        """
        # Check cell index
        try:
            rn = correct_coord(rn)
        except exc.CoordLenError as err:
            raise exc.CellIndexLenError(err.coord) from err

        # Check element indices
        if element[0] not in range(self.mat_shape[0]) or \
                element[1] not in range(self.mat_shape[1]):
            raise ValueError(f"Element {element} out of range {self.mat_shape}")

        # Check for on-site energy
        if rn == (0, 0, 0) and element[0] == element[1]:
            raise exc.PCHopDiagonalError(rn, element[0])

        # Set matrix element
        if rn in self.dict.keys():
            self.dict[rn][element[0], element[1]] = hop
        else:
            minus_rn = self.__get_minus_rn(rn)
            if minus_rn in self.dict.keys():
                self.dict[minus_rn][element[1], element[0]] = hop.conjugate()
            else:
                self.dict[rn] = np.zeros(self.mat_shape, dtype=complex)
                self.dict[rn][element[0], element[1]] = hop

        # Special treatment for (0, 0, 0) cell
        if rn == (0, 0, 0):
            self.dict[rn][element[1], element[0]] = hop.conjugate()

    def delete_mat(self, rn):
        """
        Delete hopping matrix from dictionary.

        :param rn: (ra, rb, rc)
            cell index of hopping matrix
        :returns: None
        :raises CellIndexLenError: if len(rn) != 2 or 3
        """
        # Check cell index
        try:
            rn = correct_coord(rn)
        except exc.CoordLenError as err:
            raise exc.CellIndexLenError(err.coord) from err

        # Delete hopping matrix
        if rn in self.dict.keys():
            self.dict.pop(rn, None)
        else:
            minus_rn = self.__get_minus_rn(rn)
            if minus_rn in self.dict.keys():
                self.dict.pop(minus_rn, None)
