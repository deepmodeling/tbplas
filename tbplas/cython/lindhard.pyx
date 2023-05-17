# cython: language_level=3
# cython: warn.undeclared=True
# cython: warn.unreachable=True
# cython: warn.maybe_uninitialized=True
# cython: warn.unused=True
# cython: warn.unused_arg=True
# cython: warn.unused_result=True
# cython: warn.multiple_declarators=True

import cython
from libc.math cimport cos, sin, exp
import numpy as np


#-------------------------------------------------------------------------------
#                         Functions for Lindhard class
#-------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def build_kmesh_grid(long [::1] kmesh_size):
    """
    Build the 'kmesh_grid' attribute of 'Lindhard' class.

    Parameters
    ----------
    kmesh_size: (3,) int64 array
        dimension of kmesh

    Returns
    -------
    kmesh_grid: (num_kpt, 3) int64 array
        grid coordinates of k-points on kmesh
    """
    cdef long num_kpt, ka, kb, kc, ptr
    cdef long [:,::1] kmesh_grid

    num_kpt = np.prod(kmesh_size)
    kmesh_grid = np.zeros((num_kpt, 3), dtype=np.int64)
    ptr = 0

    for ka in range(kmesh_size[0]):
        for kb in range(kmesh_size[1]):
            for kc in range(kmesh_size[2]):
                kmesh_grid[ptr, 0] = ka
                kmesh_grid[ptr, 1] = kb
                kmesh_grid[ptr, 2] = kc
                ptr += 1
    return np.asarray(kmesh_grid)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_kq_map(long [::1] kmesh_size, long [:,::1] kmesh_grid,
                 long [::1] q_point):
    """
    Remap k-points on k+q mesh to kmesh.

    Parameters
    ----------
    kmesh_size: (3,) int64 array
        dimension of kmesh
    kmesh_grid: (num_kpt, 3) int64 array
        grid coordinates of k-points on kmesh
    q_point: (3,) int64 array
        grid coordinate of q-point

    Returns
    -------
    kq_map: (num_kpt,) int64 array
        indices of k+q points in kmesh
    """
    cdef long num_kpt, ik, ka, kb, kc
    cdef long [::1] kq_map
    
    num_kpt = kmesh_grid.shape[0]
    kq_map = np.zeros(num_kpt, dtype=np.int64)
    for ik in range(num_kpt):
        ka = (q_point[0] + kmesh_grid[ik, 0]) % kmesh_size[0]
        kb = (q_point[1] + kmesh_grid[ik, 1]) % kmesh_size[1]
        kc = (q_point[2] + kmesh_grid[ik, 2]) % kmesh_size[2]
        kq_map[ik] = ka * kmesh_size[1] * kmesh_size[2] + kb * kmesh_size[2] + kc
    return np.asarray(kq_map)


@cython.boundscheck(False)
@cython.wraparound(False)
def prod_dp(double [:,::1] bands, double complex [:,:,::1] states,
            long [::1] kq_map, double beta, double mu,
            double [::1] q_point, double [:,::1] orb_pos,
            long k_min, long k_max,
            double [:,:,::1] delta_eng,
            double complex [:,:,::1] prod_df):
    """
    Calculate delta_eng and prod_df for regular q-point.

    Parameters
    ----------
    bands: (num_kpt, num_orb) float64 array
        eigenvalues on regular k-grid in eV
    states: (num_kpt, num_orb, num_orb) complex128 array
        eigenstates on regular k-grid
    kq_map: (num_kpt,) int64 array
        map of k+q grid to k-grid
    beta: double
        Lindhard.beta
    mu: double
        Lindhard.mu
    q_point: (3,) float64
        CARTESIAN coordinates of q-point in 1/NM
    orb_pos: (num_orb, 3) float64
        CARTESIAN coordinates of orbitals in NM
    k_min: int64
        lower bound of k-index assigned to this process
    k_max: int64
        upper bound of k-index assigned to this process
    delta_eng: (num_kpt, num_orb, num_orb) float64 array
        energy difference for evaluating dyn_pol
    prod_df: (num_kpt, num_orb, num_orb) complex128 array
        prod_df for evaluating dyn_pol

    Returns
    -------
    None. Results are saved in delta_eng and prod_df.
    """
    cdef long num_orb
    cdef long ik, ikqp, jj, ll, ib
    cdef double k_dot_r
    cdef double complex [::1] phase
    cdef double eng, eng_q, f, f_q
    cdef double complex prod

    num_orb = bands.shape[1]
    phase = np.zeros(num_orb, dtype=np.complex128)

    for ib in range(num_orb):
        k_dot_r = q_point[0] * orb_pos[ib, 0] \
                + q_point[1] * orb_pos[ib, 1] \
                + q_point[2] * orb_pos[ib, 2]
        phase[ib] = cos(k_dot_r) + 1j * sin(k_dot_r)

    # NOTE: the actual range of k_index is [k_min, k_max]
    for ik in range(k_min, k_max+1):
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


@cython.boundscheck(False)
@cython.wraparound(False)
def prod_dp_arb(double [:,::1] bands, double complex [:,:,::1] states,
                double [:,::1] bands_kq, double complex [:,:,::1] states_kq, 
                double beta, double mu,
                double [::1] q_point, double [:,::1] orb_pos,
                long k_min, long k_max,
                double [:,:,::1] delta_eng,
                double complex [:,:,::1] prod_df):
    """
    Calculate delta_eng and prod_df for arbitrary q-point.

    Parameters
    ---------
    bands: (num_kpt, num_orb) float64 array
        eigenvalues on regular k-grid in eV
    states: (num_kpt, num_orb, num_orb) complex128 array
        eigenstates on regular k-grid
    bands_kq: (num_kpt, num_orb) float64 array
        eigenvalues on k+q grid in eV
    states_kq: (num_kpt, num_orb, num_orb) complex128 array
        eigenstates on k+q grid
    beta: double
        Lindhard.beta
    mu: double
        Lindhard.mu
    q_point: (3,) float64
        CARTESIAN coordinates of q-point in 1/NM
    orb_pos: (num_orb, 3) float64
        CARTESIAN coordinates of orbitals in NM
    k_min: int64
        lower bound of k-index assigned to this process
    k_max: int64
        upper bound of k-index assigned to this process
    delta_eng: (num_kpt, num_orb, num_orb) float64 array
        energy difference for evaluating dyn_pol
    prod_df: (num_kpt, num_orb, num_orb) complex128 array
        prod_df for evaluating dyn_pol

    Returns
    -------
    None. Results are saved in delta_eng and prod_df.
    """
    cdef long num_orb
    cdef long ik, jj, ll, ib
    cdef double k_dot_r
    cdef double complex [::1] phase
    cdef double eng, eng_q, f, f_q
    cdef double complex prod

    num_orb = bands.shape[1]
    phase = np.zeros(num_orb, dtype=np.complex128)

    for ib in range(num_orb):
        k_dot_r = q_point[0] * orb_pos[ib, 0] \
                + q_point[1] * orb_pos[ib, 1] \
                + q_point[2] * orb_pos[ib, 2]
        phase[ib] = cos(k_dot_r) + 1j * sin(k_dot_r)

    # NOTE: the actual range of k_index is [k_min, k_max]
    for ik in range(k_min, k_max+1):
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


@cython.boundscheck(False)
@cython.wraparound(False)
def dyn_pol(double [:,:,::1] delta_eng, double complex [:,:,::1] prod_df,
            double [::1] omegas, double delta,
            long k_min, long k_max, long iq,
            double complex [:,::1] dyn_pol):
    """
    Calculate dynamic polarizability using Lindhard function,
    for cross-validation with FORTRAN version.

    Parameters
    ---------
    delta_eng: (num_kpt, num_orb, num_orb) float64 array
        energy difference for evaluating dyn_pol
    prod_df: (num_kpt, num_orb, num_orb) complex128 array
        prod_df for evaluating dyn_pol
    omegas: (num_omega,) float64 array
        frequencies on which dyn_pol is evaluated in eV
    delta: double
        broadening parameter in eV
    k_min: int64
        lower bound of k-index index assigned to this process
    k_max: int64
        upper bound of k-index index assigned to this process
    iq: int64
        index of q-point
    dyn_pol: (num_qpt, num_omega)
        dynamic polarizability

    Returns
    -------
    None. Results are saved in dyn_pol.
    """
    cdef long num_omega, num_orb
    cdef long iw, ik, jj, ll
    cdef double omega
    cdef double complex dp_sum

    num_omega = omegas.shape[0]
    num_orb = delta_eng.shape[1]

    # NOTE: the actual range of k-index is [k_min, k_max]
    for iw in range(num_omega):
        omega = omegas[iw]
        dp_sum = 0.0
        for ik in range(k_min, k_max+1):
            for jj in range(num_orb):
                for ll in range(num_orb):
                    dp_sum += prod_df[ik, jj, ll] / \
                              (delta_eng[ik, jj, ll] + omega + 1j * delta)
        dyn_pol[iq, iw] = dp_sum


@cython.boundscheck(False)
@cython.wraparound(False)
def prod_ac(double [:,::1] bands, double complex [:,:,::1] states,
            int [:,::1] hop_ind, double complex [::1] hop_eng,
            double [:,::1] hop_dr, double [:,::1] kmesh,
            double beta, double mu, int [::1] comp,
            long k_min, long k_max,
            double [:,:,::1] delta_eng, double complex [:,:,::1] prod_df):
    """
    Calculate delta_eng and prod_df for ac_cond.

    Parameters
    ----------
    bands: (num_kpt, num_orb) float64 array
        eigenvalues on regular k-grid in eV
    states: (num_kpt, num_orb, num_orb) complex128 array
        eigenstates on regular k-grid
    hop_ind: (num_hop, 2) int32 array
        orbital indices of reduced hopping terms
    hop_eng: (num_hop,) complex128 array
        hopping energies of reduced hopping terms in eV
    hop_dr: (num_hop, 3) float64 array
        hopping distances in CARTESIAN coordinates in NM
    kmesh: (num_kpt, 3) float64 array
        CARTESIAN coordinates of k-points in 1/NM
    beta: double
        Lindhard.beta
    mu: double
        Lindhard.mu
    comp: (2,) int32 array
        components of AC conductivity to calculate
    k_min: int64
        lower bound of k-index assigned to this process
    k_max: int64
        upper bound of k-index assigned to this process
    delta_eng: (num_kpt, num_orb, num_orb) float64 array
        energy difference for evaluating ac_cond
    prod_df: (num_kpt, num_orb, num_orb) complex128 array
        prod_df for evaluating ac_cond

    Returns
    -------
    None. Results are saved in delta_eng and prod_df.
    """
    cdef long num_hop, num_orb
    cdef long ik, ih, mm, nn, ib1, ib2
    cdef double k_dot_r
    cdef double complex phase
    cdef double complex [:,::1] vmat1, vmat2
    cdef double eng_m, eng_n, f_m, f_n
    cdef double complex prod1, prod2

    num_hop = hop_ind.shape[0]
    num_orb = bands.shape[1]
    vmat1 = np.zeros((num_orb, num_orb), dtype=np.complex128)
    vmat2 = np.zeros((num_orb, num_orb), dtype=np.complex128)

    # NOTE: the actual range of k_index is [k_min, k_max]
    for ik in range(k_min, k_max+1):
        # Build vmat in Bloch basis via Fourier transform
        for ib1 in range(num_orb):
            for ib2 in range(num_orb):
                vmat1[ib1, ib2] = 0.0
                vmat2[ib1, ib2] = 0.0
        for ih in range(num_hop):
            k_dot_r = kmesh[ik, 0] * hop_dr[ih, 0] + \
                      kmesh[ik, 1] * hop_dr[ih, 1] + \
                      kmesh[ik, 2] * hop_dr[ih, 2]
            phase = (cos(k_dot_r) + 1j * sin(k_dot_r)) * hop_eng[ih]
            ib1, ib2 = hop_ind[ih, 0], hop_ind[ih, 1]
            vmat1[ib1, ib2] = vmat1[ib1, ib2] + 1j * phase * hop_dr[ih, comp[0]]
            vmat1[ib2, ib1] = vmat1[ib2, ib1] - 1j * phase.conjugate() * hop_dr[ih, comp[0]]
            vmat2[ib1, ib2] = vmat2[ib1, ib2] + 1j * phase * hop_dr[ih, comp[1]]
            vmat2[ib2, ib1] = vmat2[ib2, ib1] - 1j * phase.conjugate() * hop_dr[ih, comp[1]]

        # Build delta_eng and prod_df
        for mm in range(num_orb):
            eng_m = bands[ik, mm]
            f_m = 1.0 / (1.0 + exp(beta * (eng_m - mu)))
            for nn in range(num_orb):
                eng_n = bands[ik, nn]
                delta_eng[ik, mm, nn] = eng_m - eng_n
                f_n = 1.0 / (1.0 + exp(beta * (eng_n - mu)))
                prod1, prod2 = 0.0, 0.0
                for ib1 in range(num_orb):
                    for ib2 in range(num_orb):
                        prod1 += states[ik, nn, ib1].conjugate() * vmat1[ib1, ib2] * states[ik, mm, ib2]
                        prod2 += states[ik, mm, ib1].conjugate() * vmat2[ib1, ib2] * states[ik, nn, ib2]
                if abs(eng_m - eng_n) >= 1.0e-7:
                    prod_df[ik, mm, nn] = prod1 * prod2 * (f_m - f_n) / (eng_m - eng_n)
                # else:
                #     prod_df[ik, mm, nn] = prod1 * prod2 * -beta * f_n * (1 - f_n)


@cython.boundscheck(False)
@cython.wraparound(False)
def ac_cond(double [:,:,::1] delta_eng, double complex [:,:,::1] prod_df,
            double [::1] omegas, double delta,
            long k_min, long k_max, double complex [::1] ac_cond):
    """
    Evaluate full AC conductivity using Kubo-Greenwood formula.

    Parameters
    ---------
    delta_eng: (num_kpt, num_orb, num_orb) float64 array
        energy difference for evaluating ac_cond
    prod_df: (num_kpt, num_orb, num_orb) complex128 array
        prod_df for evaluating ac_cond
    omegas: (num_omega,) float64 array
        frequencies on which dyn_pol is evaluated in eV
    delta: double
        broadening parameter in eV
    k_min: int64
        lower bound of k-index for this process
    k_max: int64
        upper bound of k-index for this process
    ac_cond: (num_omega,) complex128 array
        full ac conductivity

    Returns
    -------
    None. Results are saved in ac_cond.
    """
    cdef long num_omega, num_orb
    cdef long iw, ik, mm, nn
    cdef double omega
    cdef double complex ac_sum

    num_omega = omegas.shape[0]
    num_orb = delta_eng.shape[1]

    # NOTE: the actual range of k-index is [k_min, k_max]
    for iw in range(num_omega):
        omega = omegas[iw]
        ac_sum = 0.0
        for ik in range(k_min, k_max+1):
            for mm in range(num_orb):
                for nn in range(num_orb):
                    ac_sum += prod_df[ik, mm, nn] / \
                              (delta_eng[ik, mm, nn] - omega - 1j * delta)
        ac_cond[iw] = ac_sum
