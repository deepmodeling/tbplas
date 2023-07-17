# cython: language_level=3
# cython: warn.undeclared=True
# cython: warn.unreachable=True
# cython: warn.maybe_uninitialized=True
# cython: warn.unused=True
# cython: warn.unused_arg=True
# cython: warn.unused_result=True
# cython: warn.multiple_declarators=True

import cython
import numpy as np


#-------------------------------------------------------------------------------
#      Functions for converting index between pc and sc representations
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long _id_pc2sc(int [::1] dim, int num_orb_pc, int [::1] id_pc):
    """
    Convert orbital or vacancy index from primitive cell representation
    to super cell representation.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    id_pc: (4,) int32 array
        index in primitive cell representation

    Returns
    -------
    id_sc: int64
        index in super cell representation
    """
    cdef long num_pc, id_sc
    num_pc = id_pc[0] * dim[1] * dim[2] + id_pc[1] * dim[2] + id_pc[2]
    id_sc = num_pc * num_orb_pc + id_pc[3]
    return id_sc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long _id_pc2sc_vac(int [::1] dim, int num_orb_pc, int [::1] id_pc, 
                        long [::1] vac_id_sc):
    """
    Convert orbital index from primitive cell representation to super cell
    representation in presence of vacancies.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    id_pc: (4,) int32 array
        orbital index in primitive cell representation
    vac_id_sc: (num_vac,) int64 array
        indices of vacancies in super cell representation

    Returns
    -------
    id_sc: int64
        orbital index in super cell representation
        -1 if id_pc corresponds to a vacancy

    NOTES
    -----
    vac_id_sc should be sorted in increasing order. Otherwise the 'else'
    clause will be activated too early, and the results will be WRONG!
    """
    cdef long id_sc, offset, vac
    cdef int iv, num_vac

    # Get id_sc without considering vacancies.
    id_sc = _id_pc2sc(dim, num_orb_pc, id_pc)

    # Count the number of vacancies that fall before id_sc
    # and subtract it from the result.
    # If id_sc (id_pc) corresponds to a vacancy, return -1.
    num_vac = vac_id_sc.shape[0]
    if id_sc < vac_id_sc[0]:
        offset = 0
    elif id_sc > vac_id_sc[num_vac-1]:
        offset = num_vac
    else:
        offset = 0
        for iv in range(num_vac):
            vac = vac_id_sc[iv]
            if vac < id_sc:
                offset += 1
            elif vac == id_sc:
                return -1
            else:
                break
    id_sc -= offset
    return id_sc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long _id_pc2sc_vac2(int [::1] dim, int num_orb_pc, int [::1] id_pc, 
                         long [::1] vac_id_sc):
    """Another version of pc2sc_vac based on bi-section method."""
    cdef long id_sc, offset
    cdef int num_vac
    cdef int i0, i1, im
    cdef long x0, x1, xm

    # Get id_sc without considering vacancies.
    id_sc = _id_pc2sc(dim, num_orb_pc, id_pc)

    # Initialize variables
    num_vac = vac_id_sc.shape[0]
    i0, i1 = 0, num_vac - 1
    x0, x1 = vac_id_sc[i0], vac_id_sc[i1]

    # Check for end points
    if id_sc < x0:
        offset = 0
    elif id_sc == x0:
        offset = -1
    elif id_sc == x1:
        offset = -1
    elif id_sc > x1:
        offset = num_vac
    else:
        # Bi-section algorithm for remaining parts
        offset = i0 + 1
        while i1 - i0 > 1:
            im = i0 + (i1 - i0) // 2
            xm = vac_id_sc[im]
            if xm < id_sc:
                i0 = im
            elif xm > id_sc:
                i1 = im
            else:
                offset = -1
                break
        if offset != -1:
            offset = i0 + 1

    # Apply the offset
    if offset == -1:
        id_sc = -1
    else:
        id_sc -= offset
    return id_sc


@cython.boundscheck(False)
@cython.wraparound(False)
def id_pc2sc(int [::1] dim, int num_orb_pc, int [::1] id_pc,
             long [::1] vac_id_sc):
    """
    Common interface to _id_pc2sc and _id_pc2sc_vac.

    See the documentation of these functions for more details.
    """
    if vac_id_sc.shape[0] == 0:
        return _id_pc2sc(dim, num_orb_pc, id_pc)
    else:
        return _id_pc2sc_vac2(dim, num_orb_pc, id_pc, vac_id_sc)


@cython.boundscheck(False)
@cython.wraparound(False)
def check_id_sc_array(long num_orb_sc, long [::1] id_sc_array):
    """
    Check for errors in id_sc_array and return information of the 0th
    encountered error.

    Parameters
    ----------
    num_orb_sc: int64
        number of orbitals in the super cell
    id_sc_array (num_orb,) int64 array
        orbital indices in sc representation

    Returns
    -------
    status: (2,) int32 array
        0th element is the error type:
            -1: IDSCIndexError
             0: NO ERROR
        1th element is the index of 0th illegal id_sc
    """
    cdef long num_orb, io
    cdef int [::1] status

    status = np.zeros(2, dtype=np.int32)
    num_orb = id_sc_array.shape[0]
    for io in range(num_orb):
        if not (0 <= id_sc_array[io] < num_orb_sc):
            status[0] = -1
            status[1] = io
            break
    return np.asarray(status)


@cython.boundscheck(False)
@cython.wraparound(False)
def check_id_pc_array(int [::1] dim, int num_orb_pc, int [:,::1] id_pc_array,
                      long [::1] vac_id_sc):
    """
    Check for errors in id_pc_array and return information of the 0th
    encountered error.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    id_pc_array: (num_orb, 4) int32 array
        orbital indices in primitive cell representation
    vac_id_sc: (num_vac,) int64 array
        indices of vacancies in super cell representation

    Returns
    -------
    status: (3,) int32 array
        0th element is the error type:
            -2: IDPCIndexError
            -1: IDPCVacError
             0: NO ERROR
        1st element is the index of 0th illegal id_pc
        2nd element is the index of wrong element in 0th illegal id_pc
    """
    cdef long num_orb, io
    cdef int i_dim
    cdef int [::1] dim_ext
    cdef int [::1] status
    cdef int num_vac

    # Extend dim by adding num_orb_pc as the 3rd dimension
    dim_ext = np.zeros(4, dtype=np.int32)
    for i_dim in range(3):
        dim_ext[i_dim] = dim[i_dim]
    dim_ext[3] = num_orb_pc

    status = np.zeros(3, dtype=np.int32)
    num_orb = id_pc_array.shape[0]
    num_vac = vac_id_sc.shape[0]
    for io in range(num_orb):
        # Check for IDPCIndexError
        for i_dim in range(4):
            if not (0 <= id_pc_array[io, i_dim] < dim_ext[i_dim]):
                status[0] = -2
                status[1] = io
                status[2] = i_dim

        # Check for IDPCVacError
        if status[0] == -2:
            break
        else:
            if num_vac == 0:
                pass
            else:
                if _id_pc2sc_vac2(dim, num_orb_pc, id_pc_array[io],
                                  vac_id_sc) == -1:
                    status[0] = -1
                    status[1] = io
                    break
    return np.asarray(status)


@cython.boundscheck(False)
@cython.wraparound(False)
def id_sc2pc_array(int [:,::1] orb_id_pc, long [::1] id_sc_array):
    """
    Convert an array of orbital indices from sc representation to pc
    representation.

    Parameters
    ----------
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of all the orbitals in super cell in pc representation
    id_sc_array (num_orb,) int64 array
        orbital indices in sc representation

    Returns
    -------
    id_pc_array: (num_orb, 4) int32 array
        orbital indices in primitive cell representation
    """
    cdef long num_orb, io
    cdef int i_dim
    cdef long id_sc
    cdef int [:,::1] id_pc_array

    # Allocate arrays
    num_orb = id_sc_array.shape[0]
    id_pc_array = np.zeros((num_orb, 4), dtype=np.int32)

    # Convert orbital indices from sc to pc representation
    for io in range(num_orb):
        id_sc = id_sc_array[io]
        for i_dim in range(4):
            id_pc_array[io, i_dim] = orb_id_pc[id_sc, i_dim]
    return np.asarray(id_pc_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def id_pc2sc_array(int [::1] dim, int num_orb_pc, int [:,::1] id_pc_array,
                   long [::1] vac_id_sc):
    """
    Convert an array of orbital indices from pc representation to sc
    representation in presence of vacancies.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    id_pc_array: (num_orb, 4) int32 array
        orbital indices in primitive cell representation
    vac_id_sc: (num_vac,) int64 array
        indices of vacancies in super cell representation

    Returns
    -------
    id_sc_array: (num_orb,) int64 array
        orbital indices in super cell representation
    """
    cdef long num_orb, io
    cdef long [::1] id_sc_array

    # Allocate arrays
    num_orb = id_pc_array.shape[0]
    id_sc_array = np.zeros(num_orb, dtype=np.int64)

    # Convert orbital indices from pc to sc representation
    if vac_id_sc.shape[0] == 0:
        for io in range(num_orb):
            id_sc_array[io] = _id_pc2sc(dim, num_orb_pc, id_pc_array[io])
    else:
        for io in range(num_orb):
            id_sc_array[io] = _id_pc2sc_vac2(dim, num_orb_pc, id_pc_array[io],
                                             vac_id_sc)
    return np.asarray(id_sc_array)


#-------------------------------------------------------------------------------
#       Functions for constructing attributes of OrbitalSet class
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def build_vac_id_sc(int [::1] dim, int num_orb_pc, int [:,::1] vac_id_pc):
    """
    Build the indices of vacancies in super cell representation.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    vac_id_pc: (num_vac, 4) int32 arrray
        indices of vacancies in pc representation
    
    Returns
    -------
    vac_id_sc: (num_vac,) int64 array
        indices of vacancies in sc representation.
    """
    cdef int iv, num_vac
    cdef long [::1] vac_id_sc
 
    num_vac = vac_id_pc.shape[0]
    vac_id_sc = np.zeros(num_vac, dtype=np.int64)
    for iv in range(num_vac):
        vac_id_sc[iv] = _id_pc2sc(dim, num_orb_pc, vac_id_pc[iv])
    return np.asarray(vac_id_sc)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_orb_id_pc(int [::1] dim, int num_orb_pc, long [::1] vac_id_sc):
    """
    Build the indices of orbitals in primitive cell representation.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    vac_id_sc: (num_vac,) int64 arrray
        indices of vacancies in sc representation
    
    Returns
    -------
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals in pc representation
    """
    cdef long num_orb_sc, ptr
    cdef int ia, ib, ic, io
    cdef int [::1] id_work
    cdef int [:,::1] orb_id_pc

    if vac_id_sc.shape[0] == 0:
        num_orb_sc = np.prod(dim) * num_orb_pc
    else:
        num_orb_sc = np.prod(dim) * num_orb_pc - vac_id_sc.shape[0]
    id_work = np.zeros(4, dtype=np.int32)
    orb_id_pc = np.zeros((num_orb_sc, 4), dtype=np.int32)
    ptr = 0

    if vac_id_sc.shape[0] == 0:
        for ia in range(dim[0]):
            for ib in range(dim[1]):
                for ic in range(dim[2]):
                    for io in range(num_orb_pc):
                        orb_id_pc[ptr, 0] = ia
                        orb_id_pc[ptr, 1] = ib
                        orb_id_pc[ptr, 2] = ic
                        orb_id_pc[ptr, 3] = io
                        ptr += 1
    else:
        for ia in range(dim[0]):
            id_work[0] = ia
            for ib in range(dim[1]):
                id_work[1] = ib
                for ic in range(dim[2]):
                    id_work[2] = ic
                    for io in range(num_orb_pc):
                        id_work[3] = io
                        if _id_pc2sc_vac2(dim, num_orb_pc, id_work,
                                          vac_id_sc) != -1:
                            orb_id_pc[ptr, 0] = ia
                            orb_id_pc[ptr, 1] = ib
                            orb_id_pc[ptr, 2] = ic
                            orb_id_pc[ptr, 3] = io
                            ptr += 1
    return np.asarray(orb_id_pc)


#-------------------------------------------------------------------------------
#              Functions for building arrays for SuperCell class
#-------------------------------------------------------------------------------
cdef int _wrap_pbc(int ji, int ni, int pbc_i):
    """
    Wrap primitive cell index back into the 0th period of super cell based on
    super cell dimension and boundary condition.

    Suppose that we have a super cell created by repeating the primitive cell
    3 times along given direction (ni=3). Then the 7th primitive cell (ji=7),
    which resides in the 2nd period of super cell, is wrapped to the 1st
    primitive cell in the 0th period (ALL INDICES COUNTED FROM 0).
        0th period: [0, 1, 2]
        1st period: [3, 4, 5]
        2nd period: [6, 7, 8]
        3rd period: [9, 10, 11]
        ... ...    ... ...

    Under periodic boundary condition (pbc_i == 1), ji is always wrapped back
    into [0, ni-1].

    For open boundary condition (pbc_i == 0), ji will be dropped by setting it
    to -1 when it falls out of [0, ni-1].

    Parameters
    ----------
    ji: int32
        index of primitive cell before wrapping
    ni: int32
        dimension of super cell along given direction
    pbc_i: int32
        whether to enforce periodic condition along given direction

    Returns
    -------
    ji_new: int32
        index of primitive cell after wrapping
    """
    cdef int ji_new
    ji_new = ji
    if pbc_i == 0:
        if ji_new < 0 or ji_new >= ni:
            ji_new = -1
        else:
            pass
    else:
        while ji_new < 0:
            ji_new += ni
        while ji_new >= ni:
            ji_new -= ni
    return ji_new


cdef int _fast_div(int ji, int ni):
    """
    A faster function to evaluate ji//ni when ji mainly falls in [0, ni).

    Parameters
    ----------
    ji: int32
        dividend
    ni: int32
        divisor

    Returns
    -------
    result: int32
        quotient, ji // ni
    """
    cdef int result
    cdef int bound
    if 0 <= ji < ni:
        result = 0
    elif ji >= ni:
        result = 1
        bound = ni
        while True:
            if ji < bound:
                result -= 1
                break
            elif ji == bound:
                break
            else:
                result += 1
                bound += ni
    else:
        result = -1
        bound = -ni
        while True:
            if ji >= bound:
                break
            else:
                result -= 1
                bound -= ni
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _zero_rn(int [:,::1] pc_hop_ind, int ih):
    """
    Check if the given hopping term has rn == (0, 0, 0).

    Parameters
    ----------
    pc_hop_ind: (num_hop_pc, 5) int32 array
        reduced hopping indices of primitive cell
    ih: int32
        index of the hopping term to check

    Returns
    -------
    is_zero: int32
        1 if rn == (0, 0, 0), 0 otherwise.
    """
    cdef int is_zero, i
    is_zero = 1
    for i in range(3):
        if pc_hop_ind[ih, i] != 0:
            is_zero = 0
            break
    return is_zero


@cython.boundscheck(False)
@cython.wraparound(False)
def build_orb_pos(double [:,::1] pc_lattice, double [:,::1] pc_orb_pos,
                  int [:,::1] orb_id_pc):
    """
    Build the array of Cartesian coordinates for all the orbitals in
    super cell.

    Parameters
    ----------
    pc_lattice: (3, 3) float64 array
        lattice vectors of primitive cell in NM
    pc_orb_pos: (num_orb_pc, 3) float64 array
        FRACTIONAL coordinates of orbitals in primitive cell
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation

    Returns
    -------
    orb_pos: (num_orb_sc, 3) float64 array
        CARTESIAN coordinates of all orbitals in super cell in NM
    """
    cdef long num_orb_sc, io
    cdef double [::1] pos_frac
    cdef double ra, rb, rc
    cdef double [:,::1] orb_pos

    num_orb_sc = orb_id_pc.shape[0]
    orb_pos = np.zeros((num_orb_sc, 3), dtype=np.float64)
    for io in range(num_orb_sc):
        pos_frac = pc_orb_pos[orb_id_pc[io, 3]]
        ra = pos_frac[0] + orb_id_pc[io, 0]
        rb = pos_frac[1] + orb_id_pc[io, 1]
        rc = pos_frac[2] + orb_id_pc[io, 2]
        orb_pos[io, 0] = ra * pc_lattice[0, 0] \
                       + rb * pc_lattice[1, 0] \
                       + rc * pc_lattice[2, 0]
        orb_pos[io, 1] = ra * pc_lattice[0, 1] \
                       + rb * pc_lattice[1, 1] \
                       + rc * pc_lattice[2, 1]
        orb_pos[io, 2] = ra * pc_lattice[0, 2] \
                       + rb * pc_lattice[1, 2] \
                       + rc * pc_lattice[2, 2]
    return np.asarray(orb_pos)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_orb_eng(double [::1] pc_orb_eng, int [:,::1] orb_id_pc):
    """
    Build the array of energies for all the orbitals in super cell.

    Parameters
    ----------
    pc_orb_eng: (num_orb_pc,) float64 array
        energies of orbitals in primitive cell in eV
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation

    Returns
    -------
    orb_eng: (num_orb_sc,) float64 array
        energies of all orbitals in super cell in eV
    """
    cdef long num_orb_sc, io
    cdef double [::1] orb_eng

    num_orb_sc = orb_id_pc.shape[0]
    orb_eng = np.zeros(num_orb_sc, dtype=np.float64)
    for io in range(num_orb_sc):
        orb_eng[io] = pc_orb_eng[orb_id_pc[io, 3]]
    return np.asarray(orb_eng)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_hop_gen(int [:,::1] pc_hop_ind, double complex [::1] pc_hop_eng,
                  int [::1] dim, int [::1] pbc, int num_orb_pc,
                  int [:,::1] orb_id_pc, long [::1] vac_id_sc,
                  double [:,::1] sc_lattice, double [:,::1] sc_orb_pos):
    """
    Build the arrays of hopping terms for constructing sparse Hamiltonian
    and dr in CSR format in general cases.

    Parameters
    ----------
    pc_hop_ind: (num_hop_pc, 5) int32 array
        reduced hopping indices of primitive cell
    pc_hop_eng: (num_orb_pc,) complex128 array
        reduced hopping energies of primitive cell in eV
    dim: (3,) int32 array
        dimension of the super cell
    pbc: (3,) int32 array
        periodic conditions
    num_orb_pc: int32
        number of orbitals in primitive cell
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation
    vac_id_sc: (num_vac,) int64 array
        indices of vacancies in sc representation
    sc_lattice: (3, 3) float64 array
        CARTESIAN lattice vectors of supercell in NM
    sc_orb_pos: (num_orb_sc, 3) float64 array
        CARTESIAN coordinates of orbitals of super cell in NM

    Returns
    -------
    hop_i: (num_hop_sc,) int64 array
        reduced orbital indices of bra <i| in sc representation
    hop_j: (num_hop_sc,) int64 array
        reduced orbital indices of ket |j> in sc representation
    hop_v: (num_hop_sc,) complex128 array
        reduced hopping energy of <i|H|j> in eV
    dr: (num_hop_sc, 3) float64 array
        reduced distances corresponding to hop_i and hop_j in NM

    NOTES
    -----
    Dimension of the super cell along each direction should be no smaller
    than a minimum value. Otherwise the result will be wrong. See the
    documentation of 'OrbitalSet' class for more details.

    The hopping terms have been reduced by the conjugate relation. So only
    half of the terms are stored in hop_* and dr.
    """
    # Loop counters and bounds
    cdef long num_orb_sc, io
    cdef int num_hop_pc, ih
    cdef int num_vec

    # Cell and orbital IDs
    cdef int ja0, jb0, jc0
    cdef int ja, jb, jc
    cdef int na, nb, nc
    cdef int [::1] id_pc_j
    cdef long id_sc_i, id_sc_j

    # Results
    cdef long num_hop_sc, ptr
    cdef long [::1] hop_i, hop_j
    cdef double complex [::1] hop_v
    cdef double [:,::1] dr

    # Get the number of hopping terms and allocate arrays
    num_hop_pc = pc_hop_ind.shape[0]
    num_hop_sc = num_hop_pc * np.prod(dim)
    hop_i = np.zeros(num_hop_sc, dtype=np.int64)
    hop_j = np.zeros(num_hop_sc, dtype=np.int64)
    hop_v = np.zeros(num_hop_sc, dtype=np.complex128)
    dr = np.zeros((num_hop_sc, 3), dtype=np.float64)

    # Initialize variables
    num_vac = vac_id_sc.shape[0]
    num_orb_sc = orb_id_pc.shape[0]
    id_pc_j = np.zeros(4, dtype=np.int32)
    ptr = 0

    for io in range(num_orb_sc):
        # Get id_sc for bra
        id_sc_i = io

        # Loop over hopping terms in primitive cell
        for ih in range(num_hop_pc):
            if orb_id_pc[io, 3] == pc_hop_ind[ih, 3]:
                # Apply boundary condition
                ja0 = orb_id_pc[io, 0] + pc_hop_ind[ih, 0]
                jb0 = orb_id_pc[io, 1] + pc_hop_ind[ih, 1]
                jc0 = orb_id_pc[io, 2] + pc_hop_ind[ih, 2]
                ja = _wrap_pbc(ja0, dim[0], pbc[0])
                jb = _wrap_pbc(jb0, dim[1], pbc[1])
                jc = _wrap_pbc(jc0, dim[2], pbc[2])

                # Drop the hopping term if it is out of boundary
                if ja == -1 or jb == -1 or jc == -1:
                    pass
                else:
                    id_pc_j[0] = ja
                    id_pc_j[1] = jb
                    id_pc_j[2] = jc
                    id_pc_j[3] = pc_hop_ind[ih, 4]
                    if num_vac == 0:
                        id_sc_j = _id_pc2sc(dim, num_orb_pc, id_pc_j)
                    else:
                        id_sc_j = _id_pc2sc_vac2(dim, num_orb_pc, id_pc_j,
                                                 vac_id_sc)

                    # Check if id_sc_j corresponds to a vacancy
                    if num_vac == 0 or id_sc_j != -1:
                        hop_i[ptr] = id_sc_i
                        hop_j[ptr] = id_sc_j
                        hop_v[ptr] = pc_hop_eng[ih]
                        na = _fast_div(ja0, dim[0])
                        nb = _fast_div(jb0, dim[1])
                        nc = _fast_div(jc0, dim[2])
                        dr[ptr, 0] = sc_orb_pos[id_sc_j, 0] \
                                   - sc_orb_pos[id_sc_i, 0] \
                                   + na * sc_lattice[0, 0] \
                                   + nb * sc_lattice[1, 0] \
                                   + nc * sc_lattice[2, 0]
                        dr[ptr, 1] = sc_orb_pos[id_sc_j, 1] \
                                   - sc_orb_pos[id_sc_i, 1] \
                                   + na * sc_lattice[0, 1] \
                                   + nb * sc_lattice[1, 1] \
                                   + nc * sc_lattice[2, 1]
                        dr[ptr, 2] = sc_orb_pos[id_sc_j, 2] \
                                   - sc_orb_pos[id_sc_i, 2] \
                                   + na * sc_lattice[0, 2] \
                                   + nb * sc_lattice[1, 2] \
                                   + nc * sc_lattice[2, 2]
                        ptr += 1
    hop_i = hop_i[:ptr]
    hop_j = hop_j[:ptr]
    hop_v = hop_v[:ptr]
    dr = dr[:ptr]
    return np.asarray(hop_i), np.asarray(hop_j), np.asarray(hop_v), np.asarray(dr)


@cython.boundscheck(False)
@cython.wraparound(False)
def split_pc_hop(int [:,::1] pc_hop_ind, double complex [::1] pc_hop_eng,
                 int [::1] pbc):
    """
    Split the hopping terms of primitive cell into periodic and free
    categories.

    Parameters
    ----------
    pc_hop_ind: (num_hop, 5) int32 array
        indices of hopping terms in the primitive cell
    pc_hop_eng: (num_hop,) complex128 array
        energies of hopping terms in the primitive cell
    pbc: (3,) int32 array
        periodic boundary conditions

    Returns
    -------
    ind_pbc: (num_hop_pbc, 5) int32 array
        indices of periodic hopping terms
    eng_pbc: (num_hop_pbc,) complex128 array
        energies of periodic hopping terms
    ind_free: (num_hop_free, 5) int32 array
        indices of free hopping terms
    eng_free: (num_hop_free,) complex128 array
        energies of free hopping terms
    """
    # Loop counters and bounds
    cdef int num_hop, num_hop_pbc, num_hop_free
    cdef int ih, j, ptr_pbc, ptr_free
    cdef int [::1] status

    # Results
    cdef int [:,::1] ind_pbc, ind_free
    cdef double complex [::1] eng_pbc, eng_free

    # Determine pbc and free hopping terms
    # status: 0-free, 1-pbc
    num_hop = pc_hop_ind.shape[0]
    status = np.zeros(num_hop, dtype=np.int32)
    for ih in range(num_hop):
        status[ih] = 1
        for j in range(3):
            if pbc[j] == 0 and pc_hop_ind[ih, j] != 0:
                status[ih] = 0
                break

    # Allocate results
    num_hop_pbc = np.sum(status)
    num_hop_free = num_hop - num_hop_pbc
    ind_pbc = np.zeros((num_hop_pbc, 5), dtype=np.int32)
    ind_free = np.zeros((num_hop_free, 5), dtype=np.int32)
    eng_pbc = np.zeros(num_hop_pbc, dtype=np.complex128)
    eng_free = np.zeros(num_hop_free, dtype=np.complex128)

    # Splitting hopping terms
    ptr_pbc = 0
    ptr_free = 0
    for ih in range(num_hop):
        if status[ih] == 1:
            for j in range(5):
                ind_pbc[ptr_pbc, j] = pc_hop_ind[ih, j]
            eng_pbc[ptr_pbc] = pc_hop_eng[ih]
            ptr_pbc += 1
        else:
            for j in range(5):
                ind_free[ptr_free, j] = pc_hop_ind[ih, j]
            eng_free[ptr_free] = pc_hop_eng[ih]
            ptr_free += 1

    return np.asarray(ind_pbc), np.asarray(eng_pbc), np.asarray(ind_free), np.asarray(eng_free)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_hop_pbc(int [:,::1] pc_hop_ind, double complex [::1] pc_hop_eng,
                  int [::1] dim, int [::1] pbc, int num_orb_pc,
                  double [:,::1] sc_lattice, double [:,::1] sc_orb_pos):
    """
    Build the arrays of hopping terms arising from periodic hopping terms in the
    primitive cell for constructing sparse Hamiltonian and dr in CSR format.

    Parameters
    ----------
    pc_hop_ind: (num_hop_pc, 5) int32 array
        reduced hopping indices of primitive cell
    pc_hop_eng: (num_orb_pc,) complex128 array
        reduced hopping energies of primitive cell in eV
    dim: (3,) int32 array
        dimension of the super cell
    pbc: (3,) int32 array
        periodic conditions
    num_orb_pc: int32
        number of orbitals in primitive cell
    sc_lattice: (3, 3) float64 array
        CARTESIAN lattice vectors of supercell in NM
    sc_orb_pos: (num_orb_sc, 3) float64 array
        CARTESIAN coordinates of orbitals of super cell in NM

    Returns
    -------
    hop_i: (num_hop_sc,) int64 array
        reduced orbital indices of bra <i| in sc representation
    hop_j: (num_hop_sc,) int64 array
        reduced orbital indices of ket |j> in sc representation
    hop_v: (num_hop_sc,) complex128 array
        reduced hopping energy of <i|H|j> in eV
    dr: (num_hop_sc, 3) float64 array
        reduced distances corresponding to hop_i and hop_j in NM

    NOTES
    -----
    See 'build_hop'.
    """
    # Loop counters and bounds
    cdef int num_hop_pc, ih
    cdef long num_hop_sc, ptr

    # Cell and orbital IDs
    cdef int ia, ib, ic
    cdef int zero_rn
    cdef int ja0, jb0, jc0
    cdef int na, nb, nc
    cdef int [::1] offset_pc_i, offset_pc_j
    cdef long offset_sc_i, offset_sc_j
    cdef long id_sc_i, id_sc_j

    # Results
    cdef long [::1] hop_i, hop_j
    cdef double complex [::1] hop_v
    cdef double [:,::1] dr

    # Get the number of hopping terms and allocate arrays
    num_hop_pc = pc_hop_ind.shape[0]
    num_hop_sc = num_hop_pc * np.prod(dim)
    hop_i = np.zeros(num_hop_sc, dtype=np.int64)
    hop_j = np.zeros(num_hop_sc, dtype=np.int64)
    hop_v = np.zeros(num_hop_sc, dtype=np.complex128)
    dr = np.zeros((num_hop_sc, 3), dtype=np.float64)

    # Initialize variables
    offset_pc_i = np.zeros(4, dtype=np.int32)
    offset_pc_j = np.zeros(4, dtype=np.int32)
    ptr = 0

    for ia in range(dim[0]):
        offset_pc_i[0] = ia
        for ib in range(dim[1]):
            offset_pc_i[1] = ib
            for ic in range(dim[2]):
                offset_pc_i[2] = ic
                offset_pc_i[3] = 0

                # Evaluate the offset for orbital i
                offset_sc_i = _id_pc2sc(dim, num_orb_pc, offset_pc_i)

                for ih in range(num_hop_pc):
                    # Evaluate the offset for orbital j
                    zero_rn = _zero_rn(pc_hop_ind, ih)
                    if zero_rn == 1:
                        offset_sc_j = offset_sc_i
                    else:
                        ja0 = ia + pc_hop_ind[ih, 0]
                        jb0 = ib + pc_hop_ind[ih, 1]
                        jc0 = ic + pc_hop_ind[ih, 2]
                        offset_pc_j[0] = _wrap_pbc(ja0, dim[0], pbc[0])
                        offset_pc_j[1] = _wrap_pbc(jb0, dim[1], pbc[1])
                        offset_pc_j[2] = _wrap_pbc(jc0, dim[2], pbc[2])
                        offset_pc_j[3] = 0
                        offset_sc_j = _id_pc2sc(dim, num_orb_pc, offset_pc_j)

                    # Fill hopping terms
                    id_sc_i = offset_sc_i + pc_hop_ind[ih, 3]
                    id_sc_j = offset_sc_j + pc_hop_ind[ih, 4]
                    hop_i[ptr] = id_sc_i
                    hop_j[ptr] = id_sc_j
                    hop_v[ptr] = pc_hop_eng[ih]
                    if zero_rn == 1:
                        na, nb, nc = 0, 0, 0
                    else:
                        na = _fast_div(ja0, dim[0])
                        nb = _fast_div(jb0, dim[1])
                        nc = _fast_div(jc0, dim[2])
                    dr[ptr, 0] = sc_orb_pos[id_sc_j, 0] \
                               - sc_orb_pos[id_sc_i, 0] \
                               + na * sc_lattice[0, 0] \
                               + nb * sc_lattice[1, 0] \
                               + nc * sc_lattice[2, 0]
                    dr[ptr, 1] = sc_orb_pos[id_sc_j, 1] \
                               - sc_orb_pos[id_sc_i, 1] \
                               + na * sc_lattice[0, 1] \
                               + nb * sc_lattice[1, 1] \
                               + nc * sc_lattice[2, 1]
                    dr[ptr, 2] = sc_orb_pos[id_sc_j, 2] \
                               - sc_orb_pos[id_sc_i, 2] \
                               + na * sc_lattice[0, 2] \
                               + nb * sc_lattice[1, 2] \
                               + nc * sc_lattice[2, 2]
                    ptr += 1
    return np.asarray(hop_i), np.asarray(hop_j), np.asarray(hop_v), np.asarray(dr)


@cython.boundscheck(False)
@cython.wraparound(False)
def find_equiv_hopping(long [::1] hop_i, long [::1] hop_j, long bra, long ket):
    """
    Find the index of equivalent hopping term of <bra|H|ket> in hop_i and hop_j,
    i.e. the same term or its conjugate counterpart.

    Parameters
    ----------
    hop_i, hop_j: (num_hop_sc,) int64 array
        row and column indices of hopping terms
        reduced by a half using the conjugate relation
    bra: int64
        index of bra of the hopping term in sc representation
    ket: int64
        index of ket of the hopping term in sc representation

    Returns
    -------
    id_same: int64
        index of the same hopping term, -1 if not found
    id_conj: int64
        index of the conjugate hopping term, -1 if not found
    """
    cdef long num_hop_sc, ih, bra_old, ket_old
    cdef long id_same, id_conj

    id_same, id_conj = -1, -1
    num_hop_sc = hop_i.shape[0]
    for ih in range(num_hop_sc):
        bra_old, ket_old = hop_i[ih], hop_j[ih]
        if bra_old == bra and ket_old == ket:
            id_same = ih
            break
        elif bra_old == ket and ket_old == bra:
            id_conj = ih
            break
        else:
            pass
    return id_same, id_conj


@cython.boundscheck(False)
@cython.wraparound(False)
def get_orb_id_trim(int [:,::1] orb_id_pc, long [::1] hop_i, long [::1] hop_j):
    """
    Get the indices of orbitals to trim in primitive cell representation.

    Parameters
    ----------
    orb_id_pc: (num_orb_sc, 4) int32 array
        orbital indices in primitive cell representation
    hop_i: (num_hop_sc,) int64 array
        row indices of hopping terms reduced by conjugate relation
    hop_j: (num_hop_sc,) int64 array
        column indices of hopping terms reduced by conjugate relation

    Returns
    -------
    orb_id_trim: (num_orb_trim, 4) int32 array
        indices of orbitals to trim in primitive cell representation
    """
    cdef long num_orb_sc, io
    cdef long num_hop_sc, ih
    cdef long [::1] hop_count
    cdef long num_orb_trim, counter, i_dim
    cdef int [:,::1] orb_id_trim

    num_orb_sc = orb_id_pc.shape[0]
    num_hop_sc = hop_i.shape[0]

    # Count the number or hopping terms for each orbital
    hop_count = np.zeros(num_orb_sc, dtype=np.int64)
    for ih in range(num_hop_sc):
        hop_count[hop_i[ih]] = hop_count[hop_i[ih]] + 1
        hop_count[hop_j[ih]] = hop_count[hop_j[ih]] + 1

    # Determine the number of orbitals to trim
    # Since we have reduced hop_i and hop_j, so dangling orbitals have
    # hopping terms <= 1.
    num_orb_trim = 0
    for io in range(num_orb_sc):
        if hop_count[io] <= 1:
            num_orb_trim += 1

    # Collect orbitals to trim
    orb_id_trim = np.zeros((num_orb_trim, 4), dtype=np.int32)
    counter = 0
    for io in range(num_orb_sc):
        if hop_count[io] <= 1:
            for i_dim in range(4):
                orb_id_trim[counter, i_dim] = orb_id_pc[io, i_dim]
            counter += 1
    return np.asarray(orb_id_trim)


#-------------------------------------------------------------------------------
#             Functions for building arrays for SCInterHopping class
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def build_inter_dr(long [:,::1] hop_ind,
                   double [:,::1] pos_bra, double [:,::1] pos_ket,
                   double [:, ::1] sc_lat_ket):
    """
    Build the array of hopping distances for an 'SCInterHopping' instance.

    Parameters
    ----------
    hop_ind: (num_hop, 5) int64 array
        hopping indices
    pos_bra: (num_hop, 3) float64 array
        Cartesian coordinates of orbitals of the 'bra' super cell in nm
    pos_ket: (num_hop, 3) float64 array
        Cartesian coordinates of orbitals of the 'ket' super cell in nm
    sc_lat_ket: (3, 3) float64 array
        Cartesian coordinates of 'ket' super cell lattice vectors in nm

    Returns
    -------
    dr: (num_hop, 3) float64 array
        hopping distances in nm
    """
    cdef long num_hop, ih
    cdef long id_bra, id_ket
    cdef long na, nb, nc
    cdef double [:,::1] dr

    num_hop = hop_ind.shape[0]
    dr = np.zeros((num_hop, 3), dtype=np.float64)
    for ih in range(num_hop):
        id_bra = hop_ind[ih, 3]
        id_ket = hop_ind[ih, 4]
        na = hop_ind[ih, 0]
        nb = hop_ind[ih, 1]
        nc = hop_ind[ih, 2]
        dr[ih, 0] = pos_ket[id_ket, 0] \
                  - pos_bra[id_bra, 0] \
                  + na * sc_lat_ket[0, 0] \
                  + nb * sc_lat_ket[1, 0] \
                  + nc * sc_lat_ket[2, 0]
        dr[ih, 1] = pos_ket[id_ket, 1] \
                  - pos_bra[id_bra, 1] \
                  + na * sc_lat_ket[0, 1] \
                  + nb * sc_lat_ket[1, 1] \
                  + nc * sc_lat_ket[2, 1]
        dr[ih, 2] = pos_ket[id_ket, 2] \
                  - pos_bra[id_bra, 2] \
                  + na * sc_lat_ket[0, 2] \
                  + nb * sc_lat_ket[1, 2] \
                  + nc * sc_lat_ket[2, 2]
    return np.asarray(dr)


#-------------------------------------------------------------------------------
#                        Functions for testing purposes
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def test_acc_sps(int [::1] dim, int num_orb_pc, int [:,::1] orb_id_pc):
    """
    In this test, we convert the orbital index as sc->pc->sc, to check if the
    input is restored.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation

    Returns
    -------
    result: int64
        total absolute error
    """
    cdef long id_sc, id_sc2
    cdef long result

    result = 0
    for id_sc in range(orb_id_pc.shape[0]):
        id_sc2 = _id_pc2sc(dim, num_orb_pc, orb_id_pc[id_sc])
        result += abs(id_sc - id_sc2)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def test_acc_psp(int [::1] dim, int num_orb_pc, int [:,::1] orb_id_pc):
    """
    In this test, we convert the orbital index as pc->sc->pc, to check if the
    input is restored.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation

    Returns
    -------
    result: int64
        total absolute error
    """
    cdef int [::1] id_pc, id_pc2
    cdef long id_sc
    cdef long result
    cdef int i

    result = 0
    for id_sc in range(orb_id_pc.shape[0]):
        id_pc = orb_id_pc[id_sc]
        id_pc2 = orb_id_pc[_id_pc2sc(dim, num_orb_pc, id_pc)]
        for i in range(4):
            result += abs(id_pc[i] - id_pc2[i])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def test_acc_sps_vac(int [::1] dim, int num_orb_pc, int [:,::1] orb_id_pc,
                     long [::1] vac_id_sc):
    """
    In this test, we convert the orbital index as sc->pc->sc, to check if the
    input is restored, when there are vacancies.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation
    vac_id_sc: (num_orb_sc,) int64 array
        indices of vacancies of super cell in sc representation

    Returns
    -------
    result: int64
        total absolute error
    """
    cdef long id_sc, id_sc2
    cdef long result

    result = 0
    for id_sc in range(orb_id_pc.shape[0]):
        id_sc2 = _id_pc2sc_vac(dim, num_orb_pc, orb_id_pc[id_sc], vac_id_sc)
        result += abs(id_sc - id_sc2)
        id_sc2 = _id_pc2sc_vac2(dim, num_orb_pc, orb_id_pc[id_sc], vac_id_sc)
        result += abs(id_sc - id_sc2)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def test_acc_psp_vac(int [::1] dim, int num_orb_pc, int [:,::1] orb_id_pc,
                     long [::1] vac_id_sc):
    """
    In this test, we convert the orbital index as pc->sc->pc, to check if the
    input is restored, when there are vacancies.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation
    vac_id_sc: (num_orb_sc,) int64 array
        indices of vacancies of super cell in sc representation

    Returns
    -------
    result: int64
        total absolute error
    """
    cdef int [::1] id_pc, id_pc2
    cdef long id_sc
    cdef long result
    cdef int i

    result = 0
    for id_sc in range(orb_id_pc.shape[0]):
        id_pc = orb_id_pc[id_sc]
        id_pc2 = orb_id_pc[_id_pc2sc_vac(dim, num_orb_pc, id_pc, vac_id_sc)]
        for i in range(4):
            result += abs(id_pc[i] - id_pc2[i])
        id_pc2 = orb_id_pc[_id_pc2sc_vac2(dim, num_orb_pc, id_pc, vac_id_sc)]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def test_speed_pc2sc(int [::1] dim, int num_orb_pc, int [:,::1] orb_id_pc):
    """
    Test the speed of _id_pc2sc.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation

    Returns
    -------
    None.
    """
    cdef long i, id_sc
    for i in range(orb_id_pc.shape[0]):
        id_sc = _id_pc2sc(dim, num_orb_pc, orb_id_pc[i])


@cython.boundscheck(False)
@cython.wraparound(False)
def test_speed_pc2sc_vac(int [::1] dim, int num_orb_pc, int [:,::1] orb_id_pc,
                   long [::1] vac_id_sc):
    """
    Test the speed of _id_pc2sc_vac.

    Parameters
    ----------
    dim: (3,) int32 array
        dimension of the super cell
    num_orb_pc: int32
        number of orbitals in primitive cell
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation
    vac_id_sc: (num_orb_sc,) int64 array
        indices of vacancies of super cell in sc representation

    Returns
    -------
    None.
    """
    cdef long i, id_sc
    for i in range(orb_id_pc.shape[0]):
        id_sc = _id_pc2sc_vac(dim, num_orb_pc, orb_id_pc[i], vac_id_sc)


@cython.boundscheck(False)
@cython.wraparound(False)
def test_speed_sc2pc(int [:,::1] orb_id_pc):
    """
    Test the speed of _id_sc2pc.

    Parameters
    ----------
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals of super cell in pc representation

    Returns
    -------
    None.
    """
    cdef long id_sc
    cdef int [::1] id_pc
    for id_sc in range(orb_id_pc.shape[0]):
        id_pc = orb_id_pc[id_sc]
