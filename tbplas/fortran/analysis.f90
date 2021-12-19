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
subroutine dyn_pol_q(eng, num_orb, num_kpt, wfn, kq_map, beta, mu, &
                     omegas, num_omega, i_q, dyn_pol, num_qpt)
    implicit none

    ! input and output
    real(kind=8), intent(in) :: eng(num_orb, num_kpt)
    integer, intent(in) :: num_orb, num_kpt
    complex(kind=8), intent(in) :: wfn(num_orb, num_orb, num_kpt)
    integer, intent(in) :: kq_map(num_kpt)
    real(kind=8), intent(in) :: beta, mu
    real(kind=8), intent(in) :: omegas(num_omega)
    integer, intent(in) :: num_omega, i_q
    complex(kind=8), intent(inout) :: dyn_pol(num_omega, num_qpt)
    integer, intent(in) :: num_qpt

    ! local variables
    integer :: i_w, i_k, i_kpq, jj, ll
    real(kind=8) :: omega, f_q, f
    complex(kind=8) :: prod, dp_sum
    complex(kind=8), parameter :: eta = (0.0D0, 0.005D0)

    do i_w = 1, num_omega
        omega = omegas(i_w)
        dp_sum = (0.0D0, 0.0D0)
        !$OMP PARALLEL DO PRIVATE(i_kpq, jj, ll, f_q, f, prod) REDUCTION(+: dp_sum)
        do i_k = 1, num_kpt
            i_kpq = kq_map(i_k)
            do jj = 1, num_orb
                do ll = 1, num_orb
                    f_q = 1.0 / (1.0 + exp(beta * (eng(jj, i_kpq) - mu)))
                    f = 1.0 / (1.0 + exp(beta * (eng(ll, i_k) - mu)))
                    prod = abs(dot_product(wfn(:, jj, i_kpq), wfn(:, ll, i_k)))**2
                    dp_sum = dp_sum + prod * (f_q - f) &
                           / (eng(jj, i_kpq) - eng(ll, i_k) - omega - eta)
                end do
            end do
        end do
        !$OMP END PARALLEL DO
        dyn_pol(i_w, i_q) = dp_sum
    end do
end subroutine dyn_pol_q


! calculate dynamic polarizability for arbitrary q-point using Lindhard function
subroutine dyn_pol_q_arb(eng, num_orb, num_kpt, wfn, eng_kq, wfn_kq, beta, mu, &
                         omegas, num_omega, i_q, dyn_pol, num_qpt)
    implicit none

    ! input and output
    real(kind=8), intent(in) :: eng(num_orb, num_kpt)
    integer, intent(in) :: num_orb, num_kpt
    complex(kind=8), intent(in) :: wfn(num_orb, num_orb, num_kpt)
    real(kind=8), intent(in) :: eng_kq(num_orb, num_kpt)
    complex(kind=8), intent(in) :: wfn_kq(num_orb, num_orb, num_kpt)
    real(kind=8), intent(in) :: beta, mu
    real(kind=8), intent(in) :: omegas(num_omega)
    integer, intent(in) :: num_omega, i_q
    complex(kind=8), intent(inout) :: dyn_pol(num_omega, num_qpt)
    integer, intent(in) :: num_qpt

    ! local variables
    integer :: i_w, i_k, jj, ll
    real(kind=8) :: omega, f_q, f
    complex(kind=8) :: prod, dp_sum
    complex(kind=8), parameter :: eta = (0.0D0, 0.005D0)

    do i_w = 1, num_omega
        omega = omegas(i_w)
        dp_sum = (0.0D0, 0.0D0)
        !$OMP PARALLEL DO PRIVATE(jj, ll, f_q, f, prod) REDUCTION(+: dp_sum)
        do i_k = 1, num_kpt
            do jj = 1, num_orb
                do ll = 1, num_orb
                    f_q = 1.0 / (1.0 + exp(beta * (eng_kq(jj, i_k) - mu)))
                    f = 1.0 / (1.0 + exp(beta * (eng(ll, i_k) - mu)))
                    prod = abs(dot_product(wfn_kq(:, jj, i_k), wfn(:, ll, i_k)))**2
                    dp_sum = dp_sum + prod * (f_q - f) &
                           / (eng_kq(jj, i_k) - eng(ll, i_k) - omega - eta)
                end do
            end do
        end do
        !$OMP END PARALLEL DO
        dyn_pol(i_w, i_q) = dp_sum
    end do
end subroutine dyn_pol_q_arb
