! ------------------------------------------------------
! fortran subroutines for analysis, callable from python
! ------------------------------------------------------

! calculates everything after the calculation of the trace
SUBROUTINE cond_from_trace(mu_mn, n_kernel, mu, n_mu, H_rescale, beta, &
                           NE_integral, fermi_precision, prefactor, cond)
    USE const, ONLY: PI
    USE kpm
    USE funcs, ONLY: Fermi_dist
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: n_kernel, n_mu, NE_integral
    REAL(KIND=8), INTENT(IN) :: H_rescale, beta, fermi_precision, prefactor
    REAL(KIND=8), INTENT(IN), DIMENSION(n_mu) :: mu
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_kernel, n_kernel) :: mu_mn
    !output
    REAL(KIND=8), INTENT(OUT), DIMENSION(n_mu) :: cond

    !declare vars
    INTEGER :: i, j, k, NE
    REAL(KIND=8), DIMENSION(NE_integral-1) :: energy
    COMPLEX(KIND=8), DIMENSION(n_kernel, n_kernel) :: Gamma_mn
    REAL(KIND=8), DIMENSION(NE_integral-1) :: sum_gamma_mu
    REAL(KIND=8) :: dcx, fd, en, div, dE, sum

    cond = 0D0
    dE = PI / NE_integral
    NE = NE_integral - 1

    PRINT*, "  Calculating sum"
    DO k = 1, NE
        energy(k) = k * dE
        CALL get_gamma_mn(energy(k), n_kernel, Gamma_mn)

        sum = 0D0
        !$OMP PARALLEL DO PRIVATE(i) REDUCTION(+: sum)
        DO j = 1, n_kernel
            DO i = 1, n_kernel
                sum = sum + DBLE(Gamma_mn(i,j) * mu_mn(i,j))
            END DO
        END DO
        !$OMP END PARALLEL DO
        sum_gamma_mu(k) = sum
    END DO

    PRINT*, "  Final integral"
    DO i = 1, n_mu
        dcx = 0D0

        !$OMP PARALLEL DO SIMD PRIVATE(en, div, fd) REDUCTION(+: dcx)
        DO k = 1, NE
            div = DSIN(energy(k))**3
            en = DCOS(energy(k)) * H_rescale
            fd = Fermi_dist(beta, mu(i), en, fermi_precision)
            dcx = dcx + sum_gamma_mu(k) * fd * dE / div
        END DO
        !$OMP END PARALLEL DO SIMD

        cond(i) = dcx * prefactor / H_rescale / H_rescale
    END DO
END SUBROUTINE cond_from_trace


! calculate dynamic polarizability for regular q-point on k-mesh using
! Lindhard function
subroutine dyn_pol_q(eng, num_orb, num_kpt, wfn, kq_map, &
                     beta, mu, omegas, num_omega, delta, &
                     i_q, q_point, orb_pos, &
                     dyn_pol, num_qpt)
    implicit none

    ! input and output
    real(kind=8), intent(in) :: eng(num_orb, num_kpt)
    integer, intent(in) :: num_orb, num_kpt
    complex(kind=8), intent(in) :: wfn(num_orb, num_orb, num_kpt)
    integer, intent(in) :: kq_map(num_kpt)
    real(kind=8), intent(in) :: beta, mu
    real(kind=8), intent(in) :: omegas(num_omega)
    real(kind=8), intent(in) :: delta
    integer, intent(in) :: num_omega, i_q
    real(kind=8), intent(in) :: q_point(3)
    real(kind=8), intent(in) :: orb_pos(3, num_orb)
    complex(kind=8), intent(inout) :: dyn_pol(num_omega, num_qpt)
    integer, intent(in) :: num_qpt

    ! local variables
    integer :: i_w, i_k, i_kpq, jj, ll, ib
    real(kind=8) :: k_dot_r
    complex(kind=8) :: phase(num_orb)
    real(kind=8) :: omega, f, f_q
    complex(kind=8) :: prod, dp_sum
    real(kind=8) :: delta_eng(num_orb, num_orb, num_kpt)
    complex(kind=8) :: prod_df(num_orb, num_orb, num_kpt)
    complex(kind=8) :: cdelta

    ! build reusable arrays
    !$OMP PARALLEL DO PRIVATE(k_dot_r)
    do ib = 1, num_orb
        k_dot_r = q_point(1) * orb_pos(1, ib) &
                + q_point(2) * orb_pos(2, ib) &
                + q_point(3) * orb_pos(3, ib)
        phase(ib) = dcmplx(cos(k_dot_r), sin(k_dot_r))
    end do
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(i_kpq, jj, f, ll, f_q, prod, ib)
    do i_k = 1, num_kpt
        i_kpq = kq_map(i_k)
        do jj = 1, num_orb
            f = 1.0 / (1.0 + exp(beta * (eng(jj, i_k) - mu)))
            do ll = 1, num_orb
                delta_eng(ll, jj, i_k) = eng(jj, i_k) - eng(ll, i_kpq)
                f_q = 1.0 / (1.0 + exp(beta * (eng(ll, i_kpq) - mu)))
                prod = 0.0
                do ib = 1, num_orb
                    prod = prod + conjg(wfn(ib, ll, i_kpq)) * wfn(ib, jj, i_k) * phase(ib)
                end do
                prod_df(ll, jj, i_k) = prod * conjg(prod) * (f - f_q)
            end do
        end do
    end do
    !$OMP END PARALLEL DO

    ! calculate dyn_pol
    cdelta = dcmplx(0.0D0, delta)
    do i_w = 1, num_omega
        omega = omegas(i_w)
        dp_sum = dcmplx(0.0D0, 0.0D0)
        !$OMP PARALLEL DO PRIVATE(jj, ll) REDUCTION(+: dp_sum)
        do i_k = 1, num_kpt
            do jj = 1, num_orb
                do ll = 1, num_orb
                dp_sum = dp_sum + prod_df(ll, jj, i_k) &
                       / (delta_eng(ll, jj, i_k) + omega + cdelta)
                end do
            end do
        end do
        !$OMP END PARALLEL DO
        dyn_pol(i_w, i_q) = dp_sum
    end do
end subroutine dyn_pol_q


! calculate dynamic polarizability for arbitrary q-point using Lindhard function
subroutine dyn_pol_q_arb(eng, num_orb, num_kpt, wfn, eng_kq, wfn_kq, &
                         beta, mu, omegas, num_omega, delta, &
                         i_q, q_point, orb_pos, &
                         dyn_pol, num_qpt)
    implicit none

    ! input and output
    real(kind=8), intent(in) :: eng(num_orb, num_kpt)
    integer, intent(in) :: num_orb, num_kpt
    complex(kind=8), intent(in) :: wfn(num_orb, num_orb, num_kpt)
    real(kind=8), intent(in) :: eng_kq(num_orb, num_kpt)
    complex(kind=8), intent(in) :: wfn_kq(num_orb, num_orb, num_kpt)
    real(kind=8), intent(in) :: beta, mu
    real(kind=8), intent(in) :: omegas(num_omega)
    real(kind=8), intent(in) :: delta
    integer, intent(in) :: num_omega, i_q
    real(kind=8), intent(in) :: q_point(3)
    real(kind=8), intent(in) :: orb_pos(3, num_orb)
    complex(kind=8), intent(inout) :: dyn_pol(num_omega, num_qpt)
    integer, intent(in) :: num_qpt

    ! local variables
    integer :: i_w, i_k, jj, ll, ib
    real(kind=8) :: k_dot_r
    complex(kind=8) :: phase(num_orb)
    real(kind=8) :: omega, f, f_q
    complex(kind=8) :: prod, dp_sum
    real(kind=8) :: delta_eng(num_orb, num_orb, num_kpt)
    complex(kind=8) :: prod_df(num_orb, num_orb, num_kpt)
    complex(kind=8) :: cdelta

    ! build reusable arrays
    !$OMP PARALLEL DO PRIVATE(k_dot_r)
    do ib = 1, num_orb
        k_dot_r = q_point(1) * orb_pos(1, ib) &
                + q_point(2) * orb_pos(2, ib) &
                + q_point(3) * orb_pos(3, ib)
        phase(ib) = dcmplx(cos(k_dot_r), sin(k_dot_r))
    end do
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(jj, f, ll, f_q, prod, ib)
    do i_k = 1, num_kpt
        do jj = 1, num_orb
            f = 1.0 / (1.0 + exp(beta * (eng(jj, i_k) - mu)))
            do ll = 1, num_orb
                delta_eng(ll, jj, i_k) = eng(jj, i_k) - eng_kq(ll, i_k)
                f_q = 1.0 / (1.0 + exp(beta * (eng_kq(ll, i_k) - mu)))
                prod = 0.0
                do ib = 1, num_orb
                    prod = prod + conjg(wfn_kq(ib, ll, i_k)) * wfn(ib, jj, i_k) * phase(ib)
                end do
                prod_df(ll, jj, i_k) = prod * conjg(prod) * (f - f_q)
            end do
        end do
    end do
    !$OMP END PARALLEL DO

    ! calculate dyn_pol
    cdelta = dcmplx(0.0D0, delta)
    do i_w = 1, num_omega
        omega = omegas(i_w)
        dp_sum = dcmplx(0.0D0, 0.0D0)
        !$OMP PARALLEL DO PRIVATE(jj, ll) REDUCTION(+: dp_sum)
        do i_k = 1, num_kpt
            do jj = 1, num_orb
                do ll = 1, num_orb
                dp_sum = dp_sum + prod_df(ll, jj, i_k) &
                       / (delta_eng(ll, jj, i_k) + omega + cdelta)
                end do
            end do
        end do
        !$OMP END PARALLEL DO
        dyn_pol(i_w, i_q) = dp_sum
    end do
end subroutine dyn_pol_q_arb


! calculate real part of ac conductivity using Lindhard function
subroutine ac_cond_real(eng, num_orb, num_kpt, wfn, hop_ind, num_hop, hop_eng, &
                        hop_dr, kmesh, beta, mu, omegas, num_omega, delta, &
                        ac_cond)
    implicit none

    ! input and output
    real(kind=8), intent(in) :: eng(num_orb, num_kpt)
    integer, intent(in) :: num_orb, num_kpt
    complex(kind=8), intent(in) :: wfn(num_orb, num_orb, num_kpt)
    integer(kind=4), intent(in) :: hop_ind(2, num_hop)
    integer, intent(in) :: num_hop
    complex(kind=8), intent(in) :: hop_eng(num_hop)
    real(kind=8), intent(in) :: hop_dr(3, num_hop)
    real(kind=8), intent(in) :: kmesh(3, num_kpt)
    real(kind=8), intent(in) :: beta, mu
    real(kind=8), intent(in) :: omegas(num_omega)
    integer, intent(in) :: num_omega
    real(kind=8), intent(in) :: delta
    real(kind=8), intent(inout) :: ac_cond(num_omega)

    ! local variables
    integer :: i_w, i_k, i_h, jj, ll, ib1, ib2
    real(kind=8) :: k_dot_r
    complex(kind=8) :: phase
    complex(kind=8) :: jmat(num_orb, num_orb)
    real(kind=8) :: omega, f_j, f_l
    complex(kind=8) :: prod, ac_sum
    real(kind=8) :: delta_eng(num_orb, num_orb, num_kpt)
    complex(kind=8) :: prod_df(num_orb, num_orb, num_kpt)
    complex(kind=8), parameter :: m1j = (0.0D0, -1.0D0)
    complex(kind=8) :: cdelta

    ! build reusable arrays
    do i_k = 1, num_kpt
        ! build jmat in Bloch basis via Fourier transform
        ! NOTE: this must be done in serial mode bdecause different threads
        ! may have the same ib1 and ib2.
        ! NOTE: jmat should be initialzed as exactly zero. jmat = jmat * 0.0
        ! is incorrect under GCC.
        jmat = 0.0
        do i_h = 1, num_hop
            k_dot_r = kmesh(1, i_k) * hop_dr(1, i_h) &
                    + kmesh(2, i_k) * hop_dr(2, i_h) &
                    + kmesh(3, i_k) * hop_dr(3, i_h)
            phase = dcmplx(cos(k_dot_r), sin(k_dot_r)) * hop_eng(i_h) * hop_dr(1, i_h)
            ib1 = hop_ind(1, i_h)
            ib2 = hop_ind(2, i_h)
            jmat(ib1, ib2) = jmat(ib1, ib2) + m1j * phase
            jmat(ib2, ib1) = jmat(ib2, ib1) - m1j * conjg(phase)
        end do

        ! build delta_eng and prod_df
        !$OMP PARALLEL DO PRIVATE(f_j, ll, f_l, prod, ib1, ib2)
        do jj = 1, num_orb
            f_j = 1.0 / (1.0 + exp(beta * (eng(jj, i_k) - mu)))
            do ll = 1, num_orb
                delta_eng(ll, jj, i_k) = eng(jj, i_k) - eng(ll, i_k)
                f_l = 1.0 / (1.0 + exp(beta * (eng(ll, i_k) - mu)))
                prod = 0.0
                do ib1 = 1, num_orb
                    do ib2 = 1, num_orb
                        prod = prod + conjg(wfn(ib1, jj, i_k)) * jmat(ib1, ib2) * wfn(ib2, ll, i_k)
                    end do
                end do
                prod_df(ll, jj, i_k) = prod * conjg(prod) * (f_j - f_l)
            end do
        end do
        !$OMP END PARALLEL DO
    end do

    ! calculate real part of ac_cond
    cdelta = dcmplx(0.0D0, delta)
    do i_w = 1, num_omega
        omega = omegas(i_w)
        ac_sum = dcmplx(0.0D0, 0.0D0)
        !$OMP PARALLEL DO PRIVATE(jj, ll) REDUCTION(+: ac_sum)
        do i_k = 1, num_kpt
            do jj = 1, num_orb
                do ll = 1, num_orb
                ac_sum = ac_sum + prod_df(ll, jj, i_k) &
                       / (delta_eng(ll, jj, i_k) + omega + cdelta)
                end do
            end do
        end do
        !$OMP END PARALLEL DO
        ac_cond(i_w) = aimag(ac_sum)
    end do
end subroutine ac_cond_real
