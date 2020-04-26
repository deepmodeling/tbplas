! -----------------------------------------
! math functions, using intrinsic functions
! -----------------------------------------

#include "flags.h"

MODULE math

    IMPLICIT NONE

    ! parallel addition
    INTERFACE OPERATOR(.pAdd.)
        MODULE PROCEDURE vav_d, vav_z
    END INTERFACE OPERATOR(.pAdd.)
    ! parallel subtraction
    INTERFACE OPERATOR(.pSub.)
        MODULE PROCEDURE vsv_d, vsv_z
    END INTERFACE OPERATOR(.pSub.)
    ! parallel multipltcation
    INTERFACE OPERATOR(.pMul.)
        MODULE PROCEDURE dvmdv, zvmzv, dsmdv, dsmzv, zsmzv
    END INTERFACE OPERATOR(.pMul.)
    ! parallel division
    INTERFACE OPERATOR(.pDiv.)
        MODULE PROCEDURE dvddv, zvdzv, dvdds, zvdds, zvdzs
    END INTERFACE OPERATOR(.pDiv.)
    ! a*X
    INTERFACE self_mul
        MODULE PROCEDURE dsmdv_self, dsmzv_self, zsmzv_self
    END INTERFACE self_mul
    ! X/a
    INTERFACE self_div
        MODULE PROCEDURE dvdds_self, zvdds_self, zvdzs_self
    END INTERFACE self_div

    ! inner_prod: CONJG(X) dot Y
    INTERFACE inner_prod
        MODULE PROCEDURE dot_z, dot_d
    END INTERFACE inner_prod
    ! norm
    INTERFACE norm
        MODULE PROCEDURE norm_d, norm_z
    END INTERFACE norm
    ! parallel copy
    INTERFACE copy
        MODULE PROCEDURE copy_d, copy_z
    END INTERFACE copy
    ! Y = a*X+Y
    INTERFACE axpy
        MODULE PROCEDURE axpy_d, axpy_z, axpy_dz
    END INTERFACE axpy
    ! Y = a*X+b*Y
    INTERFACE axpby
        MODULE PROCEDURE axpby_d, axpby_z, axpby_dz
    END INTERFACE axpby

    PRIVATE :: dot_d, dot_z, norm_d, norm_z, copy_d, copy_z
    PRIVATE :: axpy_d, axpy_z, axpy_dz, axpby_d, axpby_z, axpby_dz
    PRIVATE :: vav_d, vav_z, vsv_d, vsv_z, dvmdv, zvmzv, dvddv, zvdzv
    PRIVATE :: dsmdv, dsmzv, zsmzv, dvdds, zvdds, zvdzs
    PRIVATE :: dsmdv_self, dsmzv_self, zsmzv_self
    PRIVATE :: dvdds_self, zvdds_self, zvdzs_self
#ifdef BLAS
    ! declare BLAS functions
    REAL(KIND=8), EXTERNAL, PRIVATE :: ddot, dnrm2, dznrm2
    COMPLEX(KIND=8), EXTERNAL, PRIVATE :: zdotc
#endif

CONTAINS

FUNCTION vav_d(X, Y) RESULT(Z)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    REAL(KIND=8), DIMENSION(SIZE(X)) :: Z
#ifdef VML
    CALL vdadd(SIZE(X), X, Y, Z)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = X(i) + Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION vav_d

FUNCTION vav_z(X, Y) RESULT(Z)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Z
#ifdef VML
    CALL vzadd(SIZE(X), X, Y, Z)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = X(i) + Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION vav_z

FUNCTION vsv_d(X, Y) RESULT(Z)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    REAL(KIND=8), DIMENSION(SIZE(X)) :: Z
#ifdef VML
    CALL vdsub(SIZE(X), X, Y, Z)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = X(i) - Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION vsv_d

FUNCTION vsv_z(X, Y) RESULT(Z)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Z
#ifdef VML
    CALL vzsub(SIZE(X), X, Y, Z)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = X(i) - Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION vsv_z

FUNCTION dvmdv(X, Y) RESULT(Z)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    REAL(KIND=8), DIMENSION(SIZE(X)) :: Z
#ifdef VML
    CALL vdmul(SIZE(X), X, Y, Z)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = X(i) * Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION dvmdv

FUNCTION zvmzv(X, Y) RESULT(Z)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Z
#ifdef VML
    CALL vzmul(SIZE(X), X, Y, Z)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = X(i) * Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION zvmzv

FUNCTION dsmdv(a, X) RESULT(Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    REAL(KIND=8), DIMENSION(SIZE(X)) :: Y

    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i)
    END DO
    !$OMP END PARALLEL DO SIMD
END FUNCTION dsmdv

FUNCTION dsmzv(a, X) RESULT(Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Y

    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i)
    END DO
    !$OMP END PARALLEL DO SIMD
END FUNCTION dsmzv

FUNCTION zsmzv(a, X) RESULT(Y)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN) :: a
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Y

    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i)
    END DO
    !$OMP END PARALLEL DO SIMD
END FUNCTION zsmzv

FUNCTION dvddv(X, Y) RESULT(Z)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    REAL(KIND=8), DIMENSION(SIZE(X)) :: Z
#ifdef VML
    CALL vddiv(SIZE(X), X, Y, Z)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = X(i) / Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION dvddv

FUNCTION zvdzv(X, Y) RESULT(Z)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Z
#ifdef VML
    CALL vzdiv(SIZE(X), X, Y, Z)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = X(i) / Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION zvdzv

FUNCTION dvdds(X, b) RESULT(Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X
    REAL(KIND=8), INTENT(IN) :: b
    ! output
    REAL(KIND=8), DIMENSION(SIZE(X)) :: Y

    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = X(i) / b
    END DO
    !$OMP END PARALLEL DO SIMD
END FUNCTION dvdds

FUNCTION zvdds(X, b) RESULT(Y)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    REAL(KIND=8), INTENT(IN) :: b
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Y

    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = X(i) / b
    END DO
    !$OMP END PARALLEL DO SIMD
END FUNCTION zvdds

FUNCTION zvdzs(X, b) RESULT(Y)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    COMPLEX(KIND=8), INTENT(IN) :: b
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Y

    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = X(i) / b
    END DO
    !$OMP END PARALLEL DO SIMD
END FUNCTION zvdzs

SUBROUTINE dsmdv_self(a, X)
    IMPLICIT NONE
    ! input and in-place output
    REAL(KIND=8), INTENT(IN) :: a
    REAL(KIND=8), INTENT(INOUT), DIMENSION(:) :: X
#ifdef BLAS
    CALL dscal(SIZE(X), a, X, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        X(i) = a * X(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE dsmdv_self

SUBROUTINE dsmzv_self(a, X)
    IMPLICIT NONE
    ! input and in-place output
    REAL(KIND=8), INTENT(IN) :: a
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: X
#ifdef BLAS
    CALL zdscal(SIZE(X), a, X, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        X(i) = a * X(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE dsmzv_self

SUBROUTINE zsmzv_self(a, X)
    IMPLICIT NONE
    ! input and in-place output
    COMPLEX(KIND=8), INTENT(IN) :: a
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: X
#ifdef BLAS
    CALL zscal(SIZE(X), a, X, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        X(i) = a * X(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE zsmzv_self

SUBROUTINE dvdds_self(X, b)
    IMPLICIT NONE
    ! input and in-place output
    REAL(KIND=8), INTENT(INOUT), DIMENSION(:) :: X
    REAL(KIND=8), INTENT(IN) :: b
#ifdef BLAS
    ! declare vars
    REAL(KIND=8) :: a

    a = 1D0 / b
    CALL dscal(SIZE(X), a, X, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        X(i) = X(i) / b
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE dvdds_self

SUBROUTINE zvdds_self(X, b)
    IMPLICIT NONE
    ! input and in-place output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: X
    REAL(KIND=8), INTENT(IN) :: b
#ifdef BLAS
    ! declare vars
    REAL(KIND=8) :: a

    a = 1D0 / b
    CALL zdscal(SIZE(X), a, X, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        X(i) = X(i) / b
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE zvdds_self

SUBROUTINE zvdzs_self(X, b)
    IMPLICIT NONE
    ! input and in-place output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: X
    COMPLEX(KIND=8), INTENT(IN) :: b
#ifdef BLAS
    ! declare vars
    COMPLEX(KIND=8) :: a

    a = CMPLX(1D0, KIND=8) / b
    CALL zscal(SIZE(X), a, X, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        X(i) = X(i) / b
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE zvdzs_self

FUNCTION dot_d(X, Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    REAL(KIND=8) :: dot_d
#ifdef BLAS
    dot_d = ddot(SIZE(X), X, 1, Y, 1)
#else
    dot_d = DOT_PRODUCT(X, Y)
#endif
END FUNCTION dot_d

FUNCTION dot_z(X, Y)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    COMPLEX(KIND=8) :: dot_z
#ifdef BLAS
    dot_z = zdotc(SIZE(X), X, 1, Y, 1)
#else
    dot_z = DOT_PRODUCT(X, Y)
#endif
END FUNCTION dot_z

FUNCTION norm_d(X)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    REAL(KIND=8) :: norm_d
#ifdef BLAS
    norm_d = dnrm2(SIZE(X), X, 1)
#else
    norm_d = NORM2(X)
#endif
END FUNCTION norm_d

FUNCTION norm_z(X)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    REAL(KIND=8) :: norm_z
#ifdef BLAS
    norm_z = dznrm2(SIZE(X), X, 1)
#else
    norm_z = 0D0
    norm_z = norm_z + SUM(ABS(X(:)))
#endif
END FUNCTION norm_z

FUNCTION copy_d(X) RESULT(Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    REAL(KIND=8), DIMENSION(SIZE(X)) :: Y
#ifdef BLAS
    CALL dcopy(SIZE(X), X, 1, Y, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = X(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION copy_d

FUNCTION copy_z(X) RESULT(Y)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: Y
#ifdef BLAS
    CALL zcopy(SIZE(X), X, 1, Y, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = X(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END FUNCTION copy_z

SUBROUTINE axpy_d(a, X, Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    REAL(KIND=8), INTENT(INOUT), DIMENSION(:) :: Y
#ifdef BLAS
    CALL daxpy(SIZE(X), a, X, 1, Y, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i) + Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE axpy_d

SUBROUTINE axpy_z(a, X, Y)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN) :: a
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: Y
#ifdef BLAS
    CALL zaxpy(SIZE(X), a, X, 1, Y, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i) + Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE axpy_z

SUBROUTINE axpy_dz(a, X, Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: Y
#ifdef BLAS
    CALL zaxpy(SIZE(X), CMPLX(a, KIND=8), X, 1, Y, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i) + Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE axpy_dz

SUBROUTINE axpby_d(a, X, b, Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a, b
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    REAL(KIND=8), INTENT(INOUT), DIMENSION(:) :: Y
#ifdef BLAS
    CALL daxpby(SIZE(X), a, X, 1, b, Y, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i) + b * Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE axpby_d

SUBROUTINE axpby_z(a, X, b, Y)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN) :: a, b
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: Y
#ifdef BLAS
    CALL zaxpby(SIZE(X), a, X, 1, b, Y, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i) + b * Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE axpby_z

SUBROUTINE axpby_dz(a, X, b, Y)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a, b
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X
    ! output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: Y
#ifdef BLAS
    CALL zaxpby(SIZE(X), CMPLX(a, KIND=8), X, 1, CMPLX(b, KIND=8), Y, 1)
#else
    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Y(i) = a * X(i) + b * Y(i)
    END DO
    !$OMP END PARALLEL DO SIMD
#endif
END SUBROUTINE axpby_dz

! a, b, c here are all REAL. Used in chebshev timestep
FUNCTION axpbypcz(a, X, b, Y, c, Z) RESULT(out)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a, b, c
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y, Z
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(X)) :: out

    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        out(i) = a * X(i) + b * Y(i) + c * Z(i)
    END DO
    !$OMP END PARALLEL DO SIMD
END FUNCTION axpbypcz

! Here, a is COMPLEX and b is REAL. Used only in Haydock_coef
SUBROUTINE axpbypz(a, X, b, Y, Z)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN) :: a
    REAL(KIND=8), INTENT(IN) :: b
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: X, Y
    ! output
    COMPLEX(KIND=8), INTENT(OUT), DIMENSION(SIZE(X)) :: Z

    ! declare vars
    INTEGER :: i

    !$OMP PARALLEL DO SIMD
    DO i = 1, SIZE(X)
        Z(i) = a * X(i) + b * Y(i) + Z(i)
    END DO
    !$OMP END PARALLEL DO SIMD
END SUBROUTINE axpbypz

END MODULE math
