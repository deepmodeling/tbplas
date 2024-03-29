! --------------------------------
! helper functions for calculation
! --------------------------------

MODULE funcs

    IMPLICIT NONE

CONTAINS

! get coefficients of current operator
SUBROUTINE current_coefficient(hop, dr, n_hop, value, cur_coefs)
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: n_hop
    REAL(KIND=8), INTENT(IN) :: value
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: hop
    REAL(KIND=8), INTENT(IN), DIMENSION(n_hop) :: dr
    ! output
    COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_hop) :: cur_coefs

    ! declare vars
    INTEGER :: i
    COMPLEX(KIND=8) :: alpha

    alpha = CMPLX(0D0, value, KIND=8)

    !$OMP PARALLEL DO SIMD
    DO i = 1, n_hop
        cur_coefs(i) = alpha * hop(i) * dr(i)
    END DO
    !$OMP END PARALLEL DO SIMD
END SUBROUTINE current_coefficient

! The actual Fermi distribution
PURE FUNCTION Fermi_dist(beta, Ef, energy, eps)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: beta, Ef, energy, eps
    ! output
    REAL(KIND=8) :: Fermi_dist

    ! declare vars
    REAL(KIND=8) :: x

    IF (energy >= Ef) THEN
        x = EXP(beta * (Ef - energy))
        Fermi_dist = x / (1 + x)
    ELSE
        x = EXP(beta * (energy - Ef))
        Fermi_dist = 1 / (1 + x)
    END IF

    ! IF (Fermi_dist < eps) THEN
    !     Fermi_dist = 0
    ! END IF
END FUNCTION Fermi_dist

! compute Chebyshev expansion coefficients of Fermi operator
SUBROUTINE get_Fermi_cheb_coef(cheb_coef, n_cheb, nr_Fermi, &
                               beta, mu, one_minus_Fermi, eps)
    USE const
    USE kpm, ONLY: jackson_kernel
    USE fft, ONLY : fft1d_inplace
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: nr_Fermi
    LOGICAL, INTENT(IN) :: one_minus_Fermi ! if true: compute 1-Fermi operator
    REAL(KIND=8), INTENT(IN) :: beta, mu, eps
    ! output
    REAL(KIND=8), INTENT(OUT), DIMENSION(nr_Fermi) :: cheb_coef
    INTEGER, INTENT(OUT) :: n_cheb

    ! declare vars
    INTEGER :: i
    REAL(KIND=8) :: r0, energy
    COMPLEX(KIND=8), DIMENSION(nr_Fermi) :: cheb_coef_complex
    REAL(KIND=8), DIMENSION(nr_Fermi) :: kernel

    !---------------------------------------------------------------------------
    ! NOTE: 'SIMD' should be disabled for 'OMP PARALLEL DO' instruction.
    ! Otherwise it will cause ifort to abort complaining about internal segfault
    ! errors. For gfortran there won't be such problem. (yhli)
    !---------------------------------------------------------------------------
    r0 = 2 * PI / nr_Fermi

    IF (one_minus_Fermi) THEN ! compute coeffs for one minus Fermi operator
        !$OMP PARALLEL DO PRIVATE(energy)
        DO i = 1, nr_Fermi
            energy = COS((i - 1) * r0)
            cheb_coef_complex(i) = 1D0 - Fermi_dist(beta, mu, energy, eps)
        END DO
        !$OMP END PARALLEL DO
    ELSE ! compute coeffs for Fermi operator
        !$OMP PARALLEL DO PRIVATE(energy)
        DO i = 1, nr_Fermi
            energy = COS((i - 1) * r0)
            cheb_coef_complex(i) = Fermi_dist(beta, mu, energy, eps)
        END DO
        !$OMP END PARALLEL DO
    END IF

    ! Fourier transform result
    CALL fft1d_inplace(cheb_coef_complex, -1)

    ! Get number of nonzero elements
    n_cheb = nr_Fermi / 2

    CALL jackson_kernel(kernel, n_cheb)

    !$OMP PARALLEL DO
    DO i = 1, n_cheb
        cheb_coef(i) = kernel(i) * DBLE(cheb_coef_complex(i)) / n_cheb
    END DO
    !$OMP END PARALLEL DO

    DO i = 1, n_cheb
        IF((ABS(cheb_coef(i)) < eps) .AND. (ABS(cheb_coef(i + 1)) < eps)) THEN
            n_cheb = i
            EXIT
        END IF
    END DO
END SUBROUTINE get_Fermi_cheb_coef

SUBROUTINE density_coef(n_wf, site_x, site_y, site_z, q_point, &
                        s_density_q, s_density_min_q)
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: n_wf
    REAL(KIND=8), INTENT(IN), DIMENSION(n_wf) :: site_x, site_y, site_z
    REAL(KIND=8), INTENT(IN), DIMENSION(3) :: q_point
    ! output
    COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_wf) :: s_density_q
    COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_wf) :: s_density_min_q

    ! declare vars
    INTEGER :: i
    REAL(KIND=8) :: power

    !$OMP PARALLEL DO SIMD PRIVATE(power)
    DO i = 1, n_wf
        power = q_point(1)*site_x(i) + &
                q_point(2)*site_y(i) + &
                q_point(3)*site_z(i)
        s_density_q(i) = CMPLX(COS(power), SIN(power), KIND=8)
        s_density_min_q(i) = CMPLX(COS(power), -SIN(power), KIND=8)
    END DO
    !$OMP END PARALLEL DO SIMD
END SUBROUTINE density_coef

! Green's function G00(E) using Haydock recursion method
SUBROUTINE green_function(energy, delta, coefa, coefb, n_depth, g00)
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: n_depth
    REAL(KIND=8), INTENT(IN) :: energy, delta
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: coefa
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: coefb
    ! output
    COMPLEX(KIND=8), INTENT(OUT) :: g00

    ! declare variables
    COMPLEX(KIND=8) :: E_cmplx
    INTEGER :: i

    E_cmplx = CMPLX(energy, delta, KIND=8)
    g00 = CMPLX(0D0, 0D0, KIND=8)

    DO i = n_depth, 1, -1
        g00 = 1D0 / (E_cmplx - coefa(i) - coefb(i)**2 * g00)
    END DO
END SUBROUTINE green_function

! Checks the norm of time-dependent wave function during propagation
subroutine check_norm(wf_t, norm_ref, eps)
    use ieee_arithmetic, only: ieee_is_nan, ieee_is_finite
    use math, only: inner_prod
    implicit none
    complex(kind=8), dimension(:), intent(in) :: wf_t
    complex(kind=8), intent(in) :: norm_ref
    real(kind=8), intent(in) :: eps
    real(kind=8) :: dnorm
    dnorm = abs(inner_prod(wf_t, wf_t) - norm_ref)
    if (ieee_is_nan(dnorm) .or. .not. ieee_is_finite(dnorm) .or. dnorm > eps) then
        write (*,*) "=============================================================================="
        write (*,*) "=                                   ERROR                                    ="
        write (*,*) "= Wave function norm gets broken during proparation. First of all, check the ="
        write (*,*) "= Hermitianity of the Hamiltonian. Then try increasing 'rescale' in input.   ="
        write (*,*) "=============================================================================="
        stop
    end if
end subroutine check_norm

END MODULE funcs
