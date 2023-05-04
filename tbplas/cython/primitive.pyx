# cython: language_level=3
# cython: warn.undeclared=True
# cython: warn.unreachable=True
# cython: warn.maybe_uninitialized=True
# cython: warn.unused=True
# cython: warn.unused_arg=True
# cython: warn.unused_result=True
# cython: warn.multiple_declarators=True

import cython
from libc.math cimport cos, sin, pi
import numpy as np


#-------------------------------------------------------------------------------
#                         Functions for primitive cell
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def set_ham(double [:,::1] orb_pos, double [::1] orb_eng,
            int [:,::1] hop_ind, double complex [::1] hop_eng,
            long convention, double [::1] kpt, double complex [:,::1] ham_k):
    """
    Set up Hamiltonian for given k-point using convention I:
        phase = exp(i*dot(k, R+rj-ri))
    or convention II:
        phase = exp(i*dot(k, R))

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
    convention: int64
        convention for setting up the Hamiltonian
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
        if convention == 1:
            k_dot_r = kpt[0] * (ra + orb_pos[jj, 0] - orb_pos[ii, 0]) + \
                      kpt[1] * (rb + orb_pos[jj, 1] - orb_pos[ii, 1]) + \
                      kpt[2] * (rc + orb_pos[jj, 2] - orb_pos[ii, 2])
        else:
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
    cdef int kk, num_hop_re, ptr
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
            if abs(hop_eng[ih]) < eng_cutoff:
                status[ih] = 0
            else:
                status[ih] = 1

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
