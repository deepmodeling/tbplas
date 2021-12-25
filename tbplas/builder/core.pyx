# cython: language_level=3
# cython: warn.undeclared=True
# cython: warn.unreachable=True
# cython: warn.maybe_uninitialized=True
# cython: warn.unused=True
# cython: warn.unused_arg=True
# cython: warn.unused_result=True
# cython: warn.multiple_declarators=True

import cython
from libc.math cimport cos, sin, pi, sqrt, exp
import numpy as np


#-------------------------------------------------------------------------------
#                         Functions for primitive cell
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def set_ham(double [:,::1] orb_pos, double [::1] orb_eng,
            int [:,::1] hop_ind, double complex [::1] hop_eng,
            double [::1] kpt, double complex [:,::1] ham_k):
    """
    Set up Hamiltonian for given k-point using convention I:
        phase = exp(i*dot(k, R+rj-ri))

    Parameters
    ----------
    orb_pos: (num_orb, 3) float64 array
        FRACTIONAL coordinates of orbitals
    orb_eng: (num_orb,) float64 array
        onsite energies of orbitals in eV
    hop_ind: (num_hop, 5) int32 array
        reduced hopping indices
    hop_eng: (num_hop,) complex128 array
        reduced hopping energies in eV
    kpt: (3,) float64 array
        FRACTIONAL coordinate of k-point
    ham_k: (num_orb, num_orb) complex128 array
        incoming Hamiltonian in eV
        Should be initialized as a zero matrix before calling this function.

    Returns
    -------
    None. Results are stored in ham_k.

    NOTES
    -----
    hop_ind and hop_eng contains only half of the full hopping terms.
    Conjugate terms are added automatically to ensure Hermitianity.
    """
    cdef int ra, rb, rc
    cdef double k_dot_r, phase
    cdef double complex hij
    cdef int ii, jj, ih

    # Set on-site energies
    for ii in range(orb_eng.shape[0]):
        ham_k[ii, ii] = orb_eng[ii]

    # Set hopping terms
    for ih in range(hop_ind.shape[0]):
        # Extract data
        ra, rb, rc = hop_ind[ih, 0], hop_ind[ih, 1], hop_ind[ih, 2]
        ii, jj = hop_ind[ih, 3], hop_ind[ih, 4]

        # Calculate phase
        k_dot_r = kpt[0] * (ra + orb_pos[jj, 0] - orb_pos[ii, 0]) + \
                  kpt[1] * (rb + orb_pos[jj, 1] - orb_pos[ii, 1]) + \
                  kpt[2] * (rc + orb_pos[jj, 2] - orb_pos[ii, 2])
        phase = 2 * pi * k_dot_r

        # Set Hamiltonian
        # Conjugate terms are added automatically.
        hij = hop_eng[ih] * (cos(phase) + 1j * sin(phase))
        ham_k[ii, jj] = ham_k[ii, jj] + hij
        ham_k[jj, ii] = ham_k[jj, ii] + hij.conjugate()


@cython.boundscheck(False)
@cython.wraparound(False)
def set_ham2(double [::1] orb_eng,
             int [:,::1] hop_ind, double complex [::1] hop_eng,
             double [::1] kpt, double complex [:,::1] ham_k):
    """
    Set up Hamiltonian for given k-point using convention II:
        phase = exp(i*dot(k, R))

    Parameters
    ----------
    orb_eng: (num_orb,) float64 array
        onsite energies of orbitals in eV
    hop_ind: (num_hop, 5) int32 array
        reduced hopping indices
    hop_eng: (num_hop,) complex128 array
        reduced hopping energies in eV
    kpt: (3,) float64 array
        FRACTIONAL coordinate of k-point
    ham_k: (num_orb, num_orb) complex128 array
        incoming Hamiltonian in eV
        Should be initialized as a zero matrix before calling this function.

    Returns
    -------
    None. Results are stored in ham_k.

    NOTES
    -----
    hop_ind and hop_eng contains only half of the full hopping terms.
    Conjugate terms are added automatically to ensure Hermitianity.
    """
    cdef int ra, rb, rc
    cdef double k_dot_r, phase
    cdef double complex hij
    cdef int ii, jj, ih

    # Set on-site energies
    for ii in range(orb_eng.shape[0]):
        ham_k[ii, ii] = orb_eng[ii]

    # Set hopping terms
    for ih in range(hop_ind.shape[0]):
        # Extract data
        ra, rb, rc = hop_ind[ih, 0], hop_ind[ih, 1], hop_ind[ih, 2]
        ii, jj = hop_ind[ih, 3], hop_ind[ih, 4]

        # Calculate phase
        k_dot_r = kpt[0] * ra + kpt[1] * rb + kpt[2] * rc
        phase = 2 * pi * k_dot_r

        # Set Hamiltonian
        # Conjugate terms are added automatically.
        hij = hop_eng[ih] * (cos(phase) + 1j * sin(phase))
        ham_k[ii, jj] = ham_k[ii, jj] + hij
        ham_k[jj, ii] = ham_k[jj, ii] + hij.conjugate()


@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_hop(int [:,::1] hop_ind, double complex [::1] hop_eng,
               double eng_cutoff, long num_wan):
    """
    Reduce hopping terms from seedname_hr.dat of Wannier90 and extract
    on-site energies.

    Parameters
    ----------
    hop_ind: (num_hop, 5) int32 array
        raw hopping indices
    hop_eng: (num_hop,) complex128 array
        raw hopping energies in eV
    eng_cutoff: float64
        energy cutoff for hopping terms in eV
    num_wan: int64
        number of Wannier functions

    Returns
    -------
    orb_eng: (num_wann,) float64 array
        on-site energies of orbitals in eV
    hop_ind_re: (num_hop_re, 5) int32 array
        reduced hopping indices
    hop_eng_re: (num_hop_re,) complex128
        reduced hopping energies in eV
    """
    # Loop counters and bounds
    cdef int ih, num_hop

    # Intermediate variables
    cdef int ra, rb, rc, ii, jj
    cdef int kk, is_kept, num_hop_re, ptr
    cdef int [::1] status

    # Results
    cdef double [::1] orb_eng
    cdef int [:,::1] hop_ind_re
    cdef double complex [::1] hop_eng_re

    num_hop = hop_ind.shape[0]
    status = np.zeros(num_hop, dtype=np.int32)
    orb_eng = np.zeros(num_wan, dtype=np.float64)

    # Extract on-site energies and build status
    for ih in range(num_hop):
        # Extract data
        ra, rb, rc = hop_ind[ih, 0], hop_ind[ih, 1], hop_ind[ih, 2]
        ii, jj = hop_ind[ih, 3], hop_ind[ih, 4]

        # Check if this is an on-site term
        if ra == rb == rc == 0 and ii == jj:
            orb_eng[ii] = hop_eng[ih].real
        else:
            # Check whether to keep this hopping term
            is_kept = 1
            if abs(hop_eng[ih]) < eng_cutoff:
                is_kept = 0
            else:
                for kk in range(ih):
                    if hop_ind[kk, 0] == -ra and \
                       hop_ind[kk, 1] == -rb and \
                       hop_ind[kk, 2] == -rc and \
                       hop_ind[kk, 3] == jj and \
                       hop_ind[kk, 4] == ii:
                        is_kept = 0
                        break
            status[ih] = is_kept

    # Reduce hopping terms
    num_hop_re = np.sum(status)
    hop_ind_re = np.zeros((num_hop_re, 5), dtype=np.int32)
    hop_eng_re = np.zeros(num_hop_re, dtype=np.complex128)
    ptr = 0
    for ih in range(num_hop):
        if status[ih] == 1:
            for kk in range(5):
                hop_ind_re[ptr, kk] = hop_ind[ih, kk]
            hop_eng_re[ptr] = hop_eng[ih]
            ptr += 1

    return np.asarray(orb_eng), np.asarray(hop_ind_re), np.asarray(hop_eng_re)


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
def id_pc2sc(int [::1] dim, int num_orb_pc, int [::1] id_pc,
             long [::1] vac_id_sc):
    """
    Common interface to _id_pc2sc and _id_pc2sc_vac.

    See the documentation of these functions for more details.
    """
    if vac_id_sc is None:
        return _id_pc2sc(dim, num_orb_pc, id_pc)
    else:
        return _id_pc2sc_vac(dim, num_orb_pc, id_pc, vac_id_sc)


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
                      int [:,::1] vac_id_pc):
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
    vac_id_pc: (num_vac, 4) int32 array
        indices of vacancies in primitive cell representation

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

    # Extend dim by adding num_orb_pc as the 3rd dimension
    dim_ext = np.zeros(4, dtype=np.int32)
    for i_dim in range(3):
        dim_ext[i_dim] = dim[i_dim]
    dim_ext[3] = num_orb_pc

    status = np.zeros(3, dtype=np.int32)
    num_orb = id_pc_array.shape[0]
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
            if vac_id_pc is None:
                pass
            else:
                if _check_vac(id_pc_array[io, 0], id_pc_array[io, 1],
                              id_pc_array[io, 2], id_pc_array[io, 3],
                              vac_id_pc) == 1:
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
    for io in range(num_orb):
        if vac_id_sc is None:
            id_sc_array[io] = _id_pc2sc(dim, num_orb_pc, id_pc_array[io])
        else:
            id_sc_array[io] = _id_pc2sc_vac(dim, num_orb_pc, id_pc_array[io],
                                            vac_id_sc)
    return np.asarray(id_sc_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def wrap_id_pc_array(int [:,::1] orb_id_pc, int [::1] dim, int [::1] pbc):
    """
    Wrap orbital indices in primitive cell representation using periodic
    condition.

    Parameters
    ----------
    orb_id_pc: (num_orb, 4) int32 array
        orbital indices in pc representation
    dim: (3,) int32 array
        dimension of super cell
    pbc: (3,) int32 array
        whether periodic condition is enabled along 3 directions

    Returns
    -------
    None. orb_id_pc is modified.
    """
    cdef long num_orb, i_o
    cdef int i_dim, ji_new

    num_orb = orb_id_pc.shape[0]
    for i_o in range(num_orb):
        for i_dim in range(3):
            ji_new = _wrap_pbc(orb_id_pc[i_o, i_dim], dim[i_dim], pbc[i_dim])
            orb_id_pc[i_o, i_dim] = ji_new


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
cdef int _check_vac(int ia, int ib, int ic, int io, int [:,::1] vac_id_pc):
    """
    Check if the given orbital is a vacancy.

    Parameters
    ----------
    ia, ib, ic, io: int32
        index of orbital in pc representation
    vac_id_pc: (num_vac, 4) int32 arrray
        indices of vacancies in pc representation

    Returns
    -------
    result: int32
        1 if the orbital is a vacancy. Otherwise 0.
    """
    cdef int iv, result
    result = 0
    for iv in range(vac_id_pc.shape[0]):
        if ia == vac_id_pc[iv, 0] and \
           ib == vac_id_pc[iv, 1] and \
           ic == vac_id_pc[iv, 2] and \
           io == vac_id_pc[iv, 3]:
               result = 1
               break
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def build_orb_id_pc(int [::1] dim, int num_orb_pc, int [:,::1] vac_id_pc):
    """
    Build the indices of orbitals in primitive cell representation.

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
    orb_id_pc: (num_orb_sc, 4) int32 array
        indices of orbitals in pc representation
    """
    cdef long num_orb_sc, ptr
    cdef int ia, ib, ic, io
    cdef int [:,::1] orb_id_pc

    if vac_id_pc is None:
        num_orb_sc = np.prod(dim) * num_orb_pc
    else:
        num_orb_sc = np.prod(dim) * num_orb_pc - vac_id_pc.shape[0]
    orb_id_pc = np.zeros((num_orb_sc, 4), dtype=np.int32)
    ptr = 0

    for ia in range(dim[0]):
        for ib in range(dim[1]):
            for ic in range(dim[2]):
                for io in range(num_orb_pc):
                    if vac_id_pc is None or \
                            _check_vac(ia, ib, ic, io, vac_id_pc) == 0:
                        orb_id_pc[ptr, 0] = ia
                        orb_id_pc[ptr, 1] = ib
                        orb_id_pc[ptr, 2] = ic
                        orb_id_pc[ptr, 3] = io
                        ptr += 1
    return np.asarray(orb_id_pc)


#-------------------------------------------------------------------------------
#       Functions for building arrays for SuperCell/InterHopping classes
#-------------------------------------------------------------------------------
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
cdef long _get_num_hop_sc(int [:,::1] pc_hop_ind,
                          int [::1] dim, int [::1] pbc, int num_orb_pc,
                          int [:,::1] orb_id_pc, long [::1] vac_id_sc):
    """
    Determine the number of hopping terms in the super cell.

    This function is a reduced version of 'build_hop'. But we just count
    the number of hopping terms, rather than actually filling the arrays.
    
    Parameters
    ----------
    pc_hop_ind: (num_hop_pc, 5) int32 array
        reduced hopping indices of primitive cell
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

    Returns
    -------
    num_hop_sc: int64
        number of hopping terms in the super cell

    NOTES
    -----
    Dimension of the super cell along each direction should be no smaller
    than a minimum value. Otherwise the result will be wrong. See the
    documentation of 'OrbitalSet' class for more details.

    The hopping terms have been reduced by the conjugate relation. So only
    half of the terms are taken into consideration.
    """
    # Loop counters and bounds
    cdef long num_orb_sc, io
    cdef int num_hop_pc, ih

    # Cell and orbital IDs
    cdef int ja, jb, jc
    cdef int [::1] id_pc_j
    cdef long id_sc_j

    # Results
    cdef long num_hop_sc

    # Initialize variables
    num_orb_sc = orb_id_pc.shape[0]
    num_hop_pc = pc_hop_ind.shape[0]
    id_pc_j = np.zeros(4, dtype=np.int32)
    num_hop_sc = 0

    for io in range(num_orb_sc):
        for ih in range(num_hop_pc):
            if orb_id_pc[io, 3] == pc_hop_ind[ih, 3]:
                ja = _wrap_pbc(orb_id_pc[io, 0]+pc_hop_ind[ih, 0],
                               dim[0], pbc[0])
                jb = _wrap_pbc(orb_id_pc[io, 1]+pc_hop_ind[ih, 1],
                               dim[1], pbc[1])
                jc = _wrap_pbc(orb_id_pc[io, 2]+pc_hop_ind[ih, 2],
                               dim[2], pbc[2])
                if ja == -1 or jb == -1 or jc == -1:
                    pass
                else:
                    if vac_id_sc is None:
                        num_hop_sc += 1
                    else: 
                        id_pc_j[0] = ja
                        id_pc_j[1] = jb
                        id_pc_j[2] = jc
                        id_pc_j[3] = pc_hop_ind[ih, 4]
                        id_sc_j = _id_pc2sc_vac(dim, num_orb_pc, id_pc_j,
                                                vac_id_sc)
                        if id_sc_j != -1:
                            num_hop_sc += 1
    return num_hop_sc


@cython.boundscheck(False)
@cython.wraparound(False)
def build_hop(int [:,::1] pc_hop_ind, double complex [::1] pc_hop_eng,
              int [::1] dim, int [::1] pbc, int num_orb_pc,
              int [:,::1] orb_id_pc, long [::1] vac_id_sc,
              double [:,::1] sc_lattice, double [:,::1] sc_orb_pos,
              int data_kind):
    """
    Build the arrays of hopping terms for constructing sparse Hamiltonian
    and dr in CSR format.

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
    data_kind: int32
        which arrays to generate
        0 - hop_i, hop_j, hop_v
        1 - dr

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
    num_hop_sc = _get_num_hop_sc(pc_hop_ind,
                                 dim, pbc, num_orb_pc,
                                 orb_id_pc, vac_id_sc)
    if data_kind == 0:
        hop_i = np.zeros(num_hop_sc, dtype=np.int64)
        hop_j = np.zeros(num_hop_sc, dtype=np.int64)
        hop_v = np.zeros(num_hop_sc, dtype=np.complex128)
    else:
        dr = np.zeros((num_hop_sc, 3), dtype=np.float64)

    # Initialize variables
    num_orb_sc = orb_id_pc.shape[0]
    num_hop_pc = pc_hop_ind.shape[0]
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
                    if vac_id_sc is None:
                        id_sc_j = _id_pc2sc(dim, num_orb_pc, id_pc_j)
                    else:
                        id_sc_j = _id_pc2sc_vac(dim, num_orb_pc, id_pc_j,
                                                vac_id_sc)

                    # Check if id_sc_j corresponds to a vacancy
                    if vac_id_sc is None or id_sc_j != -1:
                        if data_kind == 0:
                            hop_i[ptr] = id_sc_i
                            hop_j[ptr] = id_sc_j
                            hop_v[ptr] = pc_hop_eng[ih]
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

    # Conditional return
    if data_kind == 0:
        return np.asarray(hop_i), np.asarray(hop_j), np.asarray(hop_v)
    else:
        return np.asarray(dr)


@cython.boundscheck(False)
@cython.wraparound(False)
def check_hop(long [::1] hop_i, long [::1] hop_j):
    """
    Check if there are diagonal, duplicate or conjugate terms in hop_i and hop_j
    from SuperCell.get_hop().

    Parameters
    ----------
    hop_i: (num_hop,) int64 array
        row indices of hopping terms
    hop_j: (num_hop,) int64 array
        column indices of hopping terms

    Returns
    -------
    status: (3,) int32 array
        0th element is the error type:
            -3: diagonal term found
            -2: conjugate terms found
            -1: duplicate terms found
             0: NO ERROR
        1st and 2nd elements are the indices of diagonal, duplicate or conjugate
        terms
    """
    cdef long num_hop, ih1, ih2
    cdef long i_ref, j_ref, i_chk, j_chk
    cdef int [::1] status

    status = np.zeros(3, dtype=np.int32)
    num_hop = hop_i.shape[0]

    for ih1 in range(num_hop-1):
        i_ref, j_ref = hop_i[ih1], hop_j[ih1]
        # Check for diagonal terms
        if i_ref == j_ref:
            status[0] = -3
            status[1] = ih1
            status[2] = ih1
            break

        # Check for duplicate or conjugate terms
        for ih2 in range(ih1+1, num_hop):
            i_chk, j_chk = hop_i[ih2], hop_j[ih2]
            if i_ref == i_chk and j_ref == j_chk:
                status[0] = -1
                status[1] = ih1
                status[2] = ih2
                break
            elif i_ref == j_chk and j_ref == i_chk:
                status[0] = -2
                status[1] = ih1
                status[2] = ih2
                break
    return np.asarray(status)


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


@cython.boundscheck(False)
@cython.wraparound(False)
def check_inter_hop(long [::1] hop_i, long [::1] hop_j):
    """
    Check if there are duplicate terms in hop_i and hop_j from
    InterHop.get_hop().

    Parameters
    ----------
    hop_i: (num_hop,) int64 array
        row indices of hopping terms
    hop_j: (num_hop,) int64 array
        column indices of hopping terms

    Returns
    -------
    status: (3,) int32 array
        0th element is the error type:
            -1: dunplicate terms found
             0: NO ERROR
        1st and 2nd elements are the indices of the duplicate terms
    """
    cdef long num_hop, ih1, ih2
    cdef long i_ref, j_ref, i_chk, j_chk
    cdef int [::1] status

    status = np.zeros(3, dtype=np.int32)
    num_hop = hop_i.shape[0]

    for ih1 in range(num_hop-1):
        i_ref, j_ref = hop_i[ih1], hop_j[ih1]
        for ih2 in range(ih1+1, num_hop):
            i_chk, j_chk = hop_i[ih2], hop_j[ih2]
            if i_ref == i_chk and j_ref == j_chk:
                status[0] = -1
                status[1] = ih1
                status[2] = ih2
                break
    return np.asarray(status)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_inter_dr(long [::1] hop_i, long [::1] hop_j,
                   double [:,::1] pos_bra, double [:,::1] pos_ket,
                   int [:,::1] id_ket_pc, int [::1] dim_ket,
                   double [:, ::1] sc_lat_ket):
    """
    Build the array of hopping distances for an 'InterHopping' instance.

    Parameters
    ----------
    hop_i: (num_hop,) int64 array
        row indices of hopping terms
    hop_j: (num_hop,) int64 array
        column indices of hopping terms
    pos_bra: (num_hop, 3) float64 array
        Cartesian coordiantes of orbitals of the 'bra' super cell in nm
    pos_ket: (num_hop, 3) float64 array
        Cartesian coordiantes of orbitals of the 'ket' super cell in nm
    id_ket_pc: (num_hop, 4) int32 array
        indices of orbitals of the 'ket' super cell in pc representation
    dim_ket: (3,) int32 array
        dimension of 'ket' super cell
    sc_lat_ket: (3, 3) float64 array
        Cartesian coordiantes of 'ket' super cell lattice vectors

    Returns
    -------
    dr: (num_hop, 3) float64 array
        hopping distances in nm
    """
    cdef long num_hop, ih
    cdef long id_bra, id_ket
    cdef int na, nb, nc
    cdef double [:,::1] dr

    num_hop = hop_i.shape[0]
    dr = np.zeros((num_hop, 3), dtype=np.float64)
    for ih in range(num_hop):
        id_bra = hop_i[ih]
        id_ket = hop_j[ih]
        na = _fast_div(id_ket_pc[ih, 0], dim_ket[0])
        nb = _fast_div(id_ket_pc[ih, 1], dim_ket[1])
        nc = _fast_div(id_ket_pc[ih, 2], dim_ket[2])
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


#-------------------------------------------------------------------------------
#                   Functions for advanced Sample utilities
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def set_mag_field(long [::1] hop_i, long [::1] hop_j,
                  double complex [::1] hop_v, double [:,::1] dr,
                  double [:,::1] orb_pos, double intensity):
    """
    Add magnetic field perpendicular to xOy-plane via Peierls substitution.

    Parameters
    ----------
    hop_i, hop_j: (num_hop_sc,) int64 array
        row and column indices of hopping terms
        reduced by a half using the conjugate relation
    hop_v: (num_hop_sc,) complex128 array
        hopping energies in accordance with hop_i and hop_j in eV
    dr: (num_hop_sc, 3) float64 array
        distances corresponding to hop_i and hop_j in NM
    orb_pos: (num_orb_sc, 3) float64 array
        CARTESIAN coordinates of all orbitals in super cell in NM
    intensity: float
        intensity of magnetic field in Tesla

    Returns
    -------
    None. Results are saved in hop_v.
    """
    cdef long num_hop_sc, ih, ii, jj
    cdef double dx, ytot, phase

    num_hop_sc = hop_i.shape[0]
    for ih in range(num_hop_sc):
        ii, jj, dx = hop_i[ih], hop_j[ih], dr[ih, 0]
        ytot = orb_pos[jj, 1] + orb_pos[ii, 1]
        phase = pi * intensity * dx * ytot / 4135.666734
        hop_v[ih] = hop_v[ih] * (cos(phase) + 1j * sin(phase))


@cython.boundscheck(False)
@cython.wraparound(False)
def get_rescale(double [::1] orb_eng, long [::1] hop_i, long [::1] hop_j,
                double complex [::1] hop_v):
    """
    Estimate the rescale factor by
        rescale = np.max(np.sum(np.abs(ham_dense), axis=1))

    Parameters
    ----------
    orb_eng: (num_orb_sc,) float64 array
        on-site energies of orbitals in eV
    hop_i, hop_j: (num_hop_sc,) int64 array
        row and column indices of hopping terms
        reduced by a half using the conjugate relation
    hop_v: (num_hop_sc,) complex128 array
        hopping energies in accordance with hop_i and hop_j in eV

    Returns
    -------
    rescale: float64
        rescale factor
    """
    cdef long num_orb_sc, num_hop_sc, io, ih, ii, jj
    cdef double [::1] ham_sum
    cdef double hop_abs, rescale

    num_orb_sc = orb_eng.shape[0]
    num_hop_sc = hop_i.shape[0]

    ham_sum = np.zeros(num_orb_sc, dtype=np.float64)
    for io in range(num_orb_sc):
        ham_sum[io] = orb_eng[io]
    for ih in range(num_hop_sc):
        ii, jj = hop_i[ih], hop_j[ih]
        hop_abs = sqrt(hop_v[ih].real**2 + hop_v[ih].imag**2)
        ham_sum[ii] = ham_sum[ii] + hop_abs
        ham_sum[jj] = ham_sum[jj] + hop_abs
    rescale = np.max(ham_sum)
    return rescale


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long _get_free_ptr(long i, long [::1] indptr, int [::1] num_data_count):
    """
    Get a free pointer (unoccupied position) in the i-th row of a CSR matrix
    to put a new element in indices and data.

    Parameters
    ----------
    i: int64
        row index
    indptr: (num_row+1,) int64 array
        'indptr' attribute of CSR matrix
    num_data_count: (num_row,) int32 array
        number of occupied positions for each row

    Returns
    -------
    ptr: int64
        the first unoccupied position in i-th row

    NOTES
    -----
    This function also updates num_data_count accordingly.
    """
    cdef long ptr
    ptr = indptr[i] + num_data_count[i]
    num_data_count[i] = num_data_count[i] + 1
    return ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def build_ham_dxy_fast(double [::1] orb_eng, long [::1] hop_i, long [::1] hop_j,
                       double complex [::1] hop_v, double [:,::1] dr,
                       double cutoff):
    """
    Build the indptr, indices, hop, dx and dy arrays for conductivity
    calculations using TBPM.

    Parameters
    ----------
    orb_eng: (num_orb_sc,) float64 array
        on-site energies for all orbitals in eV
    hop_i, hop_j: (num_hop_sc,) int64 array
        row and column indices of hopping terms
        reduced by a half using the conjugate relation
    hop_v: (num_hop_sc,) complex128 array
        hopping energies in accordance with hop_i and hop_j in eV
    dr: (num_hop_sc, 3) float64 array
        distances corresponding to hop_i and hop_j in NM
    cutoff: float64
        cutoff for orbital energies in eV

    Returns
    -------
    indptr: int64 array
        shared 'indtpr' array for hop, dx and dy
    indices: int64 array
        shared 'indices' array for hop, dx and dy
    hop: complex128 array
        'data' array for constructing CSR Hamiltonian
    dx: float64 array
        'data' array for constructing CSR dx matrix
    dy: float64 array
        'data' array for constructing CSR dy matrix

    NOTES
    -----
    This function is equivalent to 'build_ham_csr2', but twice faster.
    Also, it uses less memory. So you should use this function in most
    cases.
    """
    # Common loop bounds and pointers
    cdef long num_orb_sc, num_hop_sc
    cdef long io, ih, ptr

    # indptr
    cdef long [::1] indptr

    # indices and data
    cdef long num_data_tot, ii, jj
    cdef long [::1] indices
    cdef double complex [::1] hop
    cdef double [::1] dx, dy
    cdef int [::1] num_data_count

    # Build indptr
    num_orb_sc = orb_eng.shape[0]
    num_hop_sc = hop_i.shape[0]
    indptr = np.zeros(num_orb_sc+1, dtype=np.int64)
    for io in range(num_orb_sc):
        if abs(orb_eng[io]) >= cutoff:
            ptr = io + 1
            indptr[ptr] = indptr[ptr] + 1
    for ih in range(num_hop_sc):
        ptr = hop_i[ih] + 1
        indptr[ptr] = indptr[ptr] + 1
        ptr = hop_j[ih] + 1
        indptr[ptr] = indptr[ptr] + 1
    indptr[0] = 0
    for io in range(1, indptr.shape[0]):
        indptr[io] = indptr[io-1] + indptr[io]

    # Allocate indices and data
    num_data_tot = indptr[indptr.shape[0]-1]
    indices = np.zeros(num_data_tot, dtype=np.int64)
    hop = np.zeros(num_data_tot, dtype=np.complex128)
    dx = np.zeros(num_data_tot, dtype=np.float64)
    dy = np.zeros(num_data_tot, dtype=np.float64)
    num_data_count = np.zeros(num_orb_sc, dtype=np.int32)

    # Fill indices and data
    # Diagonal part
    for io in range(num_orb_sc):
        if abs(orb_eng[io]) >= cutoff:
            ptr = _get_free_ptr(io, indptr, num_data_count)
            indices[ptr] = io
            hop[ptr] = orb_eng[io]
            dx[ptr] = 0.0
            dy[ptr] = 0.0

    # Off-diagonal part
    for ih in range(num_hop_sc):
        ii, jj = hop_i[ih], hop_j[ih]
        # Original part
        ptr = _get_free_ptr(ii, indptr, num_data_count)
        indices[ptr] = jj
        hop[ptr] = hop_v[ih]
        dx[ptr] = dr[ih, 0]
        dy[ptr] = dr[ih, 1]

        # Conjugate part
        ptr = _get_free_ptr(jj, indptr, num_data_count)
        indices[ptr] = ii
        hop[ptr] = hop_v[ih].conjugate()
        dx[ptr] = -dr[ih, 0]
        dy[ptr] = -dr[ih, 1]

    return np.asarray(indptr), np.asarray(indices), np.asarray(hop), \
        np.asarray(dx), np.asarray(dy)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_ham_dxy_safe(double [::1] orb_eng, long [::1] hop_i, long [::1] hop_j,
                       double complex [::1] hop_v, double [:,::1] dr,
                       double cutoff):
    """
    Build the indptr, indices, hop, dx and dy arrays for conductivity
    calculations using TBPM.

    Parameters
    ----------
    orb_eng: (num_orb_sc,) float64 array
        on-site energies for all orbitals in eV
    hop_i, hop_j: (num_hop_sc,) int64 array
        row and column indices of hopping terms
        reduced by a half using the conjugate relation
    hop_v: (num_hop_sc,) complex128 array
        hopping energies in accordance with hop_i and hop_j in eV
    dr: (num_hop_sc, 3) float64 array
        distances corresponding to hop_i and hop_j in NM
    cutoff: float64
        cutoff for orbital energies in eV

    Returns
    -------
    indptr: int64 array
        shared 'indtpr' array for hop, dx and dy
    indices: int64 array
        shared 'indices' array for hop, dx and dy
    hop: complex128 array
        'data' array for constructing CSR Hamiltonian
    dx: float64 array
        'data' array for constructing CSR dx matrix
    dy: float64 array
        'data' array for constructing CSR dy matrix

    NOTES
    -----
    This function is provided as an alternative to 'build_ham_dxy_fast' and
    is mainly for testing and reference purposes. According to out test,
    build_ham_dxy_fast takes about 1.48s for Graphene with 10^7 atoms while
    this function takes 2.83s. Also, this function uses more memory. So, do
    not use this function in production runs.
    """
    # Common loop bounds and counters
    cdef long num_orb_sc, num_hop_sc
    cdef long io, ih, ptr

    # indices and data
    cdef long num_nz, num_data_tot
    cdef long [::1] hop_i_full, hop_j_full
    cdef double complex [::1] hop
    cdef double [::1] dx, dy
    cdef long [::1] ind

    # indptr
    cdef long [::1] indptr

    # Get the number of non-zero orbital energies
    num_orb_sc = orb_eng.shape[0]
    num_nz = 0
    for io in range(num_orb_sc):
        if abs(orb_eng[io]) >= cutoff:
            num_nz += 1

    # Allocate indices and data
    num_hop_sc = hop_i.shape[0]
    num_data_tot = num_nz + 2 * num_hop_sc
    hop_i_full = np.zeros(num_data_tot, dtype=np.int64)
    hop_j_full = np.zeros(num_data_tot, dtype=np.int64)
    hop = np.zeros(num_data_tot, dtype=np.complex128)
    dx = np.zeros(num_data_tot, dtype=np.float64)
    dy = np.zeros(num_data_tot, dtype=np.float64)

    # Fill indices and data
    # Diagonal part
    ptr = 0
    for io in range(num_orb_sc):
        if abs(orb_eng[io]) >= cutoff:
            hop_i_full[ptr] = io
            hop_j_full[ptr] = io
            hop[ptr] = orb_eng[io]
            dx[ptr] = 0.0
            dy[ptr] = 0.0
            ptr += 1

    # Off-diagonal part
    for ih in range(num_hop_sc):
        # Original part
        ptr = ih + num_nz
        hop_i_full[ptr] = hop_i[ih]
        hop_j_full[ptr] = hop_j[ih]
        hop[ptr] = hop_v[ih]
        dx[ptr] = dr[ih, 0]
        dy[ptr] = dr[ih, 1]

        # Conjugate part
        ptr += num_hop_sc
        hop_i_full[ptr] = hop_j[ih]
        hop_j_full[ptr] = hop_i[ih]
        hop[ptr] = hop_v[ih].conjugate()
        dx[ptr] = -dr[ih, 0]
        dy[ptr] = -dr[ih, 1]

    # Sort arrays
    ind = np.argsort(hop_i_full)
    hop_i_full = hop_i_full.base[ind]
    hop_j_full = hop_j_full.base[ind]
    hop = hop.base[ind]
    dx = dx.base[ind]
    dy = dy.base[ind]

    # Build intptr
    indptr = np.zeros(num_orb_sc+1, dtype=np.int64)
    for ih in range(hop_i_full.shape[0]):
        ptr = hop_i_full[ih] + 1
        indptr[ptr] = indptr[ptr] + 1
    indptr[0] = 0
    for io in range(1, indptr.shape[0]):
        indptr[io] = indptr[io-1] + indptr[io]

    return np.asarray(indptr), np.asarray(hop_j_full), np.asarray(hop), \
        np.asarray(dx), np.asarray(dy)


@cython.boundscheck(False)
@cython.wraparound(False)
def sort_col_csr(long [::1] indptr, long [::1] indices,
                 double complex [::1] hop, double [::1] dx, double [::1] dy):
    """
    Sort column indices and corresponding data for results from
    'build_ham_dxy_fast' and 'build_ham_dxy_safe'.

    Parameters
    ----------
    indptr: int64 array
        shared 'indtpr' array for hop, dx and dy
    indices: int64 array
        shared 'indices' array for hop, dx and dy
    hop: complex128 array
        'data' array for constructing CSR Hamiltonian
    dx: float64 array
        'data' array for constructing CSR dx matrix
    dy: float64 array
        'data' array for constructing CSR dy matrix

    Returns
    -------
    None. The incoming arrays are updated.

    NOTES
    -----
    Sorting the columns of CSR matrices is mainly for testing purposes. It is
    slow, yet does not improve the efficiency of TBPM calculations.
    """
    cdef long i_row, ptr0, ptr1
    cdef long [::1] ind_buf
    cdef double complex [::1] hop_buf
    cdef double [::1] dx_buf, dy_buf
    cdef long [::1] ind_sort
    cdef long offset, i_buf, ptr2

    for i_row in range(indptr.shape[0]-1):
        ptr0, ptr1 = indptr[i_row], indptr[i_row+1]

        # Back up the row
        ind_buf = np.copy(indices[ptr0:ptr1])
        hop_buf = np.copy(hop[ptr0:ptr1])
        dx_buf = np.copy(dx[ptr0:ptr1])
        dy_buf = np.copy(dy[ptr0:ptr1])

        # Sor the row
        ind_sort = np.argsort(ind_buf)
        for offset in range(ind_sort.shape[0]):
            ptr2 = ptr0 + offset
            i_buf = ind_sort[offset]
            indices[ptr2] = ind_buf[i_buf]
            hop[ptr2] = hop_buf[i_buf]
            dx[ptr2] = dx_buf[i_buf]
            dy[ptr2] = dy_buf[i_buf]


@cython.boundscheck(False)
@cython.wraparound(False)
def build_hop_k(double complex [::1] hop_v, double [:,::1] dr,
                double [::1] kpt, double complex [::1] hop_k):
    """
    Build the arrays of hopping terms for constructing sparse Hamiltonian
    in CSR format for given k-point.

    The arrays hop_i and hop_j are identical to that of build_hop, while hop_k
    differs from hop_v by element-dependent phase factors. So we create these
    arrays in Python by calling build_hop, then updating hop_k using this
    function.

    Parameters
    ----------
    hop_v: (num_hop_sc,) complex128 array
        reduced hopping energy of <i|H|j> in eV
    dr: (num_hop_sc, 3) float64 array
        reduced distances corresponding to hop_i and hop_j in NM
    kpt: (3,) float64 array
        CARTESIAN coordinate of k-point in 1/NM
    hop_k: (num_hop_sc,) complex128 array
        incoming hop_k to be set up in eV

    Returns
    -------
    None. Results are saved in hop_k.

    NOTES
    -----
    Dimension of the super cell along each direction should be no smaller
    than a minimum value. Otherwise the result will be wrong. See the
    documentation of 'OrbitalSet' class for more details.

    The hopping terms have been reduced by the conjugate relation. So only
    half of the terms are stored in hop_k.
    """
    cdef long num_hop_sc, ih
    cdef double phase

    num_hop_sc = hop_v.shape[0]
    for ih in range(num_hop_sc):
        phase = dr[ih, 0] * kpt[0] + dr[ih, 1] * kpt[1] + dr[ih, 2] * kpt[2]
        hop_k[ih] = hop_v[ih] * (cos(phase) + 1j * sin(phase))


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_ham(double [::1] orb_eng, long [::1] hop_i, long [::1] hop_j,
             double complex [::1] hop_v, double complex [:,::1] ham_dense):
    """
    Fill dense Hamiltonian with on-site energies and hopping terms.

    Parameters
    ----------
    orb_eng: (num_orb_sc,) float64 array
        on-site energies or all orbitals in eV
    hop_i, hop_j: (num_hop_sc,) int64 array
        row and column indices of hopping terms
        reduced by a half using the conjugate relation
    hop_v: (num_hop_sc,) complex128 array
        hopping energies in accordance with hop_i and hop_j in eV
    ham_dense: (num_orb_sc, num_orb_sc) complex128 array
        incoming dense Hamiltonian in eV

    Returns
    -------
    None. Results are stored in ham_dense.

    NOTES
    -----
    Dimension of the super cell along each direction should be no smaller
    than a minimum value. Otherwise the result will be wrong. See the
    documentation of 'OrbitalSet' class for more details.
    """
    cdef long num_orb_sc, io
    cdef long num_hop_sc, ih
    cdef long ii, jj

    num_orb_sc = orb_eng.shape[0]
    num_hop_sc = hop_i.shape[0]

    for io in range(num_orb_sc):
        ham_dense[io, io] = orb_eng[io]

    for ih in range(num_hop_sc):
        ii, jj = hop_i[ih], hop_j[ih]
        ham_dense[ii, jj] = ham_dense[ii, jj] + hop_v[ih]
        ham_dense[jj, ii] = ham_dense[jj, ii] + hop_v[ih].conjugate()


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


@cython.boundscheck(False)
@cython.wraparound(False)
def dyn_pol_q(double [:,::1] bands, double complex [:,:,::1] states,
              long [::1] kq_map,
              double beta, double mu, double [::1] omegas,
              long iq, double [::1] q_point, double [:,::1] orb_pos,
              double complex [:,::1] dyn_pol):
    """
    Calculate dynamic polarizability for regular q-point on k-mesh using
    Lindhard function, for cross-validation with FORTRAN version.

    Parmaters
    ---------
    bands: (num_kpt, num_orb) float64 array
        eigenvalues on regular k-grid
    states: (num_kpt, num_orb, num_orb) complex128 array
        eigenstates on regular k-grid
    kq_map: (num_kpt,) int64 array
        map of k+q grid to k-grid
    beta: double
        Lindhard.beta
    mu: double
        Lindhard.mu
    omegas: (num_omega,) float64 array
        frequencies on which dyn_pol is evaluated
    iq: int64
        index of q-point
    q_point: (3,) float64
        CARTESIAN coordinates of q-point in 1/NM
    orb_pos: (num_orb, 3) float64
        CARTESIAN coordinates of orbitals in NM
    dyn_pol: (num_qpt, num_omega)
        dynamic polarizability

    Returns
    -------
    None. Results are saved in dyn_pol.
    """
    cdef long num_omega, num_kpt, num_orb
    cdef long iw, ik, ikqp, jj, ll, ib
    cdef double k_dot_r
    cdef double complex [::1] phase
    cdef double omega, eng, eng_q, f, f_q
    cdef double complex prod, dp_sum
    cdef double [:,:,::1] delta_eng
    cdef double complex [:,:,::1] prod_df

    num_omega = omegas.shape[0]
    num_kpt = bands.shape[0]
    num_orb = bands.shape[1]
    phase = np.zeros(num_orb, dtype=np.complex128)
    delta_eng = np.zeros((num_kpt, num_orb, num_orb), dtype=np.float64)
    prod_df = np.zeros((num_kpt, num_orb, num_orb), dtype=np.complex128)

    # Build reusable arrays
    for ib in range(num_orb):
        k_dot_r = q_point[0] * orb_pos[ib, 0] \
                + q_point[1] * orb_pos[ib, 1] \
                + q_point[2] * orb_pos[ib, 2]
        phase[ib] = cos(k_dot_r) + 1j * sin(k_dot_r)

    for ik in range(num_kpt):
        ikqp = kq_map[ik]
        for jj in range(num_orb):
            eng = bands[ik, jj]
            f = 1.0 / (1.0 + exp(beta * (eng - mu)))
            for ll in range(num_orb):
                eng_q = bands[ikqp, ll]
                delta_eng[ik, jj, ll] = eng - eng_q
                f_q = 1.0 / (1.0 + exp(beta * (eng_q - mu)))
                prod = 0.0
                for ib in range(num_orb):
                    prod += states[ikqp, ll, ib].conjugate() * states[ik, jj, ib] * phase[ib]
                prod_df[ik, jj, ll] = prod * prod.conjugate() * (f - f_q)

    # Evaluate dyn_pol
    for iw in range(num_omega):
        omega = omegas[iw]
        dp_sum = 0.0
        for ik in range(num_kpt):
            for jj in range(num_orb):
                for ll in range(num_orb):
                    dp_sum += prod_df[ik, jj, ll] / \
                              (delta_eng[ik, jj, ll] + omega + 0.005j)
        dyn_pol[iq, iw] = dp_sum


@cython.boundscheck(False)
@cython.wraparound(False)
def dyn_pol_q_arb(double [:,::1] bands, double complex [:,:,::1] states,
                  double [:,::1] bands_kq, double complex [:,:,::1] states_kq,
                  double beta, double mu, double [::1] omegas,
                  long iq, double [::1] q_point, double [:,::1] orb_pos,
                  double complex [:,::1] dyn_pol):
    """
    Calculate dynamic polarizability for arbitrary q-point using Lindhard
    function, for cross-validation with FORTRAN version.

    Parmaters
    ---------
    bands: (num_kpt, num_orb) float64 array
        eigenvalues on regular k-grid
    states: (num_kpt, num_orb, num_orb) complex128 array
        eigenstates on regular k-grid
    bands_kq: (num_kpt, num_orb) float64 array
        eigenvalues on k+q grid
    states_kq: (num_kpt, num_orb, num_orb) complex128 array
        eigenstates on k+q grid
    beta: double
        Lindhard.beta
    mu: double
        Lindhard.mu
    omegas: (num_omega,) float64 array
        frequencies on which dyn_pol is evaluated
    iq: int64
        index of q-point
    q_point: (3,) float64
        CARTESIAN coordinates of q-point in 1/NM
    orb_pos: (num_orb, 3) float64
        CARTESIAN coordinates of orbitals in NM
    dyn_pol: (num_qpt, num_omega)
        dynamic polarizability

    Returns
    -------
    None. Results are saved in dyn_pol.
    """
    cdef long num_omega, num_kpt, num_orb
    cdef long iw, ik, jj, ll, ib
    cdef double k_dot_r
    cdef double complex [::1] phase
    cdef double omega, eng, eng_q, f, f_q
    cdef double complex prod, dp_sum
    cdef double [:,:,::1] delta_eng
    cdef double complex [:,:,::1] prod_df

    num_omega = omegas.shape[0]
    num_kpt = bands.shape[0]
    num_orb = bands.shape[1]
    phase = np.zeros(num_orb, dtype=np.complex128)
    delta_eng = np.zeros((num_kpt, num_orb, num_orb), dtype=np.float64)
    prod_df = np.zeros((num_kpt, num_orb, num_orb), dtype=np.complex128)

    # Build reusable arrays
    for ib in range(num_orb):
        k_dot_r = q_point[0] * orb_pos[ib, 0] \
                + q_point[1] * orb_pos[ib, 1] \
                + q_point[2] * orb_pos[ib, 2]
        phase[ib] = cos(k_dot_r) + 1j * sin(k_dot_r)

    for ik in range(num_kpt):
        for jj in range(num_orb):
            eng = bands[ik, jj]
            f = 1.0 / (1.0 + exp(beta * (eng - mu)))
            for ll in range(num_orb):
                eng_q = bands_kq[ik, ll]
                delta_eng[ik, jj, ll] = eng - eng_q
                f_q = 1.0 / (1.0 + exp(beta * (eng_q - mu)))
                prod = 0.0
                for ib in range(num_orb):
                    prod += states_kq[ik, ll, ib].conjugate() * states[ik, jj, ib] * phase[ib]
                prod_df[ik, jj, ll] = prod * prod.conjugate() * (f - f_q)

    # Evaluate dyn_pol
    for iw in range(num_omega):
        omega = omegas[iw]
        dp_sum = 0.0
        for ik in range(num_kpt):
            for jj in range(num_orb):
                for ll in range(num_orb):
                    dp_sum += prod_df[ik, jj, ll] / \
                              (delta_eng[ik, jj, ll] + omega + 0.005j)
        dyn_pol[iq, iw] = dp_sum
