! -------------------------------
! some constants for FORTRAN part
! -------------------------------

MODULE const

    IMPLICIT NONE
    REAL(KIND=8), PARAMETER :: pi=3.14159265358979323846D0
    COMPLEX(KIND=8), PARAMETER :: img = CMPLX(0D0, 1D0, KIND=8)
    COMPLEX(KIND=8), PARAMETER :: zero_cmp = CMPLX(0D0, 0D0, KIND=8)
    COMPLEX(KIND=8), PARAMETER :: one_cmp = CMPLX(1D0, 0D0, KIND=8)

END MODULE const
