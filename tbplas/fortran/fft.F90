! ---------------------------------------
! FFT functions using Cooley-Tukey method
! ---------------------------------------

#include "flags.h"

MODULE fft

    IMPLICIT NONE
#ifdef FFTW
    ! the fftw flag we use here: FFTW_ESTIMATE | FFTW_DESTROY_INPUT
    INTEGER, PARAMETER, PRIVATE :: FFTW_FLAG = 65
#endif

CONTAINS

! Cooley-Tukey FFT in non-recursive form
SUBROUTINE fft1d_inplace(x, sgn)
    USE const
    IMPLICIT NONE
    ! input and in-place output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: x
    INTEGER, INTENT(IN) :: sgn
#ifdef FFTW
    ! declare vars
    INTEGER(KIND=8) :: plan

    CALL dfftw_plan_dft_1d(plan, SIZE(x), x, x, sgn, FFTW_FLAG)
    CALL dfftw_execute_dft(plan, x, x)
    CALL dfftw_destroy_plan(plan)
#else
    ! declare vars
    INTEGER :: n, i, j, k, ncur, ntmp, itmp
    REAL(KIND=8) :: e
    COMPLEX(KIND=8) :: ctmp

    n = SIZE(x)
    ncur = n

    DO
        ntmp = ncur
        e = 2.0 * pi / ncur
        ncur = ncur / 2
        IF ( ncur < 1 ) EXIT
        !$OMP PARALLEL DO PRIVATE(i, itmp, ctmp)
        DO j = 1, ncur
            DO i = j, n, ntmp
                itmp = i + ncur
                ctmp = x(i) - x(itmp)
                x(i) = x(i) + x(itmp)
                x(itmp) = ctmp * EXP(CMPLX(0.0, sgn*e*(j-1), KIND=8))
            END DO
        END DO
        !$OMP END PARALLEL DO
    END DO

    j = 1
    DO i = 1, n - 1
        IF ( i < j ) THEN
            ctmp = x(j)
            x(j) = x(i)
            x(i) = ctmp
        END IF
        k = n/2
        DO WHILE( k < j )
            j = j - k
            k = k / 2
        END DO
        j = j + k
    END DO
#endif
END SUBROUTINE fft1d_inplace

END MODULE fft
