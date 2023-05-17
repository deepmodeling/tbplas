# cython: language_level=3
# cython: warn.undeclared=True
# cython: warn.unreachable=True
# cython: warn.maybe_uninitialized=True
# cython: warn.unused=True
# cython: warn.unused_arg=True
# cython: warn.unused_result=True
# cython: warn.multiple_declarators=True

import cython
from libc.math cimport cos, sin, pi, sqrt
import numpy as np


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
#                   Functions for advanced Sample utilities
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def set_mag_field(long [::1] hop_i, long [::1] hop_j,
                  double complex [::1] hop_v, double [:,::1] dr,
                  double [:,::1] orb_pos, double intensity, long gauge):
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
    gauge: int64
        gauge of vector field

    Returns
    -------
    None. Results are saved in hop_v.
    """
    cdef long num_hop_sc, ih, ii, jj
    cdef double dx, dy, sx, sy, phase
    cdef double factor = pi / 4135.666734

    num_hop_sc = hop_i.shape[0]
    if gauge == 0:
        for ih in range(num_hop_sc):
            ii, jj = hop_i[ih], hop_j[ih]
            dx = dr[ih, 0]
            sy = orb_pos[jj, 1] + orb_pos[ii, 1]
            phase = factor * intensity * dx * sy
            hop_v[ih] = hop_v[ih] * (cos(phase) + 1j * sin(phase))
    elif gauge == 1:
        for ih in range(num_hop_sc):
            ii, jj = hop_i[ih], hop_j[ih]
            dy = dr[ih, 1]
            sx = orb_pos[jj, 0] + orb_pos[ii, 0]
            phase = -factor * intensity * dy * sx
            hop_v[ih] = hop_v[ih] * (cos(phase) + 1j * sin(phase))
    else:
        for ih in range(num_hop_sc):
            ii, jj = hop_i[ih], hop_j[ih]
            dx, dy = dr[ih, 0], dr[ih, 1]
            sx = orb_pos[jj, 0] + orb_pos[ii, 0]
            sy = orb_pos[jj, 1] + orb_pos[ii, 1]
            phase = factor * 0.5 * intensity * (dx * sx - dy * sy)
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
