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
