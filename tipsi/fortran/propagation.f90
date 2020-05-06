! -----------------------
! propagation subroutines
! -----------------------

MODULE propagation

    IMPLICIT NONE

CONTAINS

! Apply forward timestep using Chebyshev decomposition
FUNCTION cheb_wf_timestep_fwd(H_csr, Bes, wf_in) RESULT(wf_out)
    USE const
    USE math
    USE csr
    IMPLICIT NONE
    ! input
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: H_csr
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: Bes
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: wf_in
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(wf_in)) :: wf_out

    ! declare vars
    INTEGER :: i
    COMPLEX(KIND=8), DIMENSION(SIZE(wf_in)), TARGET :: Tcheb0, Tcheb1
    COMPLEX(KIND=8), DIMENSION(:), POINTER :: p0, p1, p2

    Tcheb0 = amv(-img, H_csr, wf_in)
    Tcheb1 = amv(-2*img, H_csr, Tcheb0) .pAdd. wf_in(:)
    wf_out = axpbypcz(Bes(1), wf_in, 2*Bes(2), Tcheb0, 2*Bes(3), Tcheb1)

    p0 => Tcheb0
    p1 => Tcheb1
    DO i = 4, SIZE(Bes)
        p2 => p0
        CALL amxpy(-2*img, H_csr, p1, p0) ! p2 = -2*img * H_csr * p1 + p0
        CALL axpy(2*Bes(i), p2, wf_out)   ! wf_out = wf_out + 2*Bes(i) * p2
        p0 => p1
        p1 => p2
    END DO
END FUNCTION cheb_wf_timestep_fwd

! Apply inverse timestep using Chebyshev decomposition
FUNCTION cheb_wf_timestep_inv(H_csr, Bes, wf_in) RESULT(wf_out)
    USE const, ONLY: img
    USE math, ONLY: copy, axpy, axpbypcz, OPERATOR(.pAdd.), OPERATOR(.pMul.)
    USE csr, ONLY: SPARSE_MATRIX_T, amv, amxpy
    IMPLICIT NONE
    ! input
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: H_csr
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: Bes
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: wf_in
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(wf_in)) :: wf_out

    ! declare vars
    INTEGER :: i
    COMPLEX(KIND=8), DIMENSION(SIZE(wf_in)), TARGET :: Tcheb0, Tcheb1
    COMPLEX(KIND=8), DIMENSION(:), POINTER :: p0, p1, p2

    Tcheb0 = amv(img, H_csr, wf_in)
    Tcheb1 = amv(2*img, H_csr, Tcheb0) .pAdd. wf_in(:)
    wf_out = axpbypcz(Bes(1), wf_in, 2*Bes(2), Tcheb0, 2*Bes(3), Tcheb1)

    p0 => Tcheb0
    p1 => Tcheb1
    DO i = 3, SIZE(Bes)
        p2 => p0
        CALL amxpy(2*img, H_csr, p1, p0) ! p2 = 2*img * H_csr * p1 + p0
        CALL axpy(2*Bes(i), p2, wf_out)  ! wf_out = wf_out + 2*Bes(i) * p2
        p0 => p1
        p1 => p2
    END DO
END FUNCTION cheb_wf_timestep_inv

! Fermi-Dirac distribution operator
FUNCTION Fermi(H_csr, cheb_coef, wf_in) RESULT(wf_out)
    USE math, ONLY: copy, axpy, axpbypcz, OPERATOR(.pSub.), OPERATOR(.pMul.)
    USE csr, ONLY: SPARSE_MATRIX_T, amv, amxpby, OPERATOR(*)
    IMPLICIT NONE
    ! input
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: H_csr
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: cheb_coef
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: wf_in
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(wf_in)) :: wf_out

    ! declare vars
    INTEGER :: i
    COMPLEX(KIND=8), DIMENSION(SIZE(wf_in)), TARGET :: Tcheb0, Tcheb1
    COMPLEX(KIND=8), DIMENSION(:), POINTER :: p0, p1, p2

    Tcheb0 = H_csr * wf_in
    Tcheb1 = amv(2D0, H_csr, Tcheb0) .pSub. wf_in(:)
    wf_out = axpbypcz(cheb_coef(1), wf_in, cheb_coef(2), Tcheb0, &
                      cheb_coef(3), Tcheb1)

    p0 => Tcheb0
    p1 => Tcheb1
    DO i = 4, SIZE(cheb_coef)
        p2 => p0
        CALL amxpby(2D0, H_csr, p1, -1D0, p0) ! p2 = 2 * H_csr * p1 - p0
        CALL axpy(cheb_coef(i), p2, wf_out)
        p0 => p1
        p1 => p2
    END DO
END FUNCTION Fermi

! Get Haydock coefficients using Haydock recursion method
SUBROUTINE Haydock_coef(n1, n_depth, H_csr, H_rescale, coefa, coefb)
    USE math, ONLY: inner_prod, norm, axpy, axpbypz, self_div
    USE csr, ONLY: SPARSE_MATRIX_T, amv
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: n_depth
    REAL(KIND=8), INTENT(IN) :: H_rescale
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:), TARGET :: n1
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: H_csr
    ! output
    COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_depth) :: coefa
    REAL(KIND=8), INTENT(OUT), DIMENSION(n_depth) :: coefb

    ! declare vars
    INTEGER :: i
    COMPLEX(KIND=8), DIMENSION(SIZE(n1)), TARGET :: n0, n2
    COMPLEX(KIND=8), DIMENSION(:), POINTER :: p0, p1, p2, p3

    ! get a1
    n2 = amv(H_rescale, H_csr, n1)
    coefa(1) = inner_prod(n1, n2)
    ! get b1
    CALL axpy(-coefa(1), n1, n2)
    coefb(1) = norm(n2)

    p1 => n1
    p2 => n2
    p3 => n0
    ! recursion
    DO i = 2, n_depth
        IF (MODULO(i, 1000) == 0) THEN
            PRINT *, "    Depth    ", i, " of ", n_depth
        END IF

        p0 => p1
        p1 => p2
        CALL self_div(p1, coefb(i-1)) ! p1(:) = p2(:) / coefb(i-1)

        p2 => p3
        p2 = amv(H_rescale, H_csr, p1)
        coefa(i) = inner_prod(p1, p2)

        CALL axpbypz(-coefa(i), p1, -coefb(i-1), p0, p2)
        coefb(i) = norm(p2)
        p3 => p0
    END DO
END SUBROUTINE Haydock_coef

END MODULE propagation
