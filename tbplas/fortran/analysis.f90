! ------------------------------------------------------
! fortran subroutines for analysis, callable from python
! ------------------------------------------------------

! calculates everything after the calculation of the trace
SUBROUTINE cond_from_trace(mu_mn, n_kernel, mu, n_mu, H_rescale, beta, &
                           NE_integral, fermi_precision, rank, cond)
    USE const, ONLY: PI
    USE kpm
    USE funcs, ONLY: Fermi_dist
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: n_kernel, n_mu, NE_integral
    REAL(KIND=8), INTENT(IN) :: H_rescale, beta, fermi_precision
    REAL(KIND=8), INTENT(IN), DIMENSION(n_mu) :: mu
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_kernel, n_kernel) :: mu_mn
    INTEGER, INTENT(IN) :: rank
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
        IF (MODULO(k, 64) == 0) THEN
            IF (rank == 0) PRINT *, "Calculating for energy ", k, " of ", NE
        END IF

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
        IF (rank == 0) PRINT *, "Calculating for energy ", i, " of ", n_mu

        dcx = 0D0

        !$OMP PARALLEL DO SIMD PRIVATE(en, div, fd) REDUCTION(+: dcx)
        DO k = 1, NE
            div = DSIN(energy(k))**3
            en = DCOS(energy(k)) * H_rescale
            fd = Fermi_dist(beta, mu(i), en, fermi_precision)
            dcx = dcx + sum_gamma_mu(k) * fd * dE / div
        END DO
        !$OMP END PARALLEL DO SIMD

        cond(i) = dcx
    END DO
END SUBROUTINE cond_from_trace
