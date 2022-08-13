! -----------------------
! propagation subroutines
! -----------------------

MODULE propagation

    IMPLICIT NONE

CONTAINS

! Apply forward/backward timestep using Chebyshev decomposition
FUNCTION cheb_wf_timestep(H_csr, Bes, wf_in, fwd) RESULT(wf_out)
    USE const
    USE math
    USE csr
    IMPLICIT NONE
    ! input
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: H_csr
    REAL(KIND=8), INTENT(IN), DIMENSION(:) :: Bes
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: wf_in
    LOGICAL, INTENT(IN) :: fwd
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(wf_in)) :: wf_out

    ! declare vars
    INTEGER :: i
    COMPLEX(KIND=8), DIMENSION(SIZE(wf_in)), TARGET :: Tcheb0, Tcheb1
    COMPLEX(KIND=8), DIMENSION(:), POINTER :: p0, p1, p2
    COMPLEX(KIND=8) :: img_dt

    if (fwd) then
        img_dt = img
    else
        img_dt = -img
    end if
    Tcheb0 = amv(-img_dt, H_csr, wf_in)
    Tcheb1 = amv(-2*img_dt, H_csr, Tcheb0) .pAdd. wf_in(:)
    wf_out = axpbypcz(Bes(1), wf_in, 2*Bes(2), Tcheb0, 2*Bes(3), Tcheb1)

    p0 => Tcheb0
    p1 => Tcheb1
    DO i = 4, SIZE(Bes)
        p2 => p0
        CALL amxpy(-2*img_dt, H_csr, p1, p0) ! p2 = -2*img_dt * H_csr * p1 + p0
        CALL axpy(2*Bes(i), p2, wf_out)   ! wf_out = wf_out + 2*Bes(i) * p2
        p0 => p1
        p1 => p2
    END DO
END FUNCTION cheb_wf_timestep

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
    COMPLEX(KIND=8), DIMENSION(SIZE(n1)), TARGET :: n2, n_buf
    COMPLEX(KIND=8), DIMENSION(:), POINTER :: p1, p2, p_buf, p_swap

    ! Formulation of the method
    !
    ! initial condition:
    !   a1 = <n1|H|n1>
    !   m2 = (H - a1)|n1>
    !   b2 = |m2|
    !   n2 = m2 / b2
    !
    ! iteration:
    !   ai = <ni|H|ni>
    !   mi+1 = (H - ai)|ni> - bi|ni-1>
    !   bi+1 = |mi+1|
    !   ni+1 = mi+1 / bi+1
    !
    ! where mi+1 is the unnormalized wave function at iteration i+1.
    !
    ! In out implementation:
    !   p1, n1 -> ni
    !   p2, n2 -> ni+1
    !   p_buf, n_buf -> H|ni> and mi+1

    ! get initial values for a and b
    n_buf = amv(H_rescale, H_csr, n1)  ! n_buf = H|n1>
    coefa(1) = inner_prod(n1, n_buf)   ! a1 = <n1|H|n1> = <n1|n_buf>
    CALL axpy(-coefa(1), n1, n_buf)    ! m2 = (H - a1)|n1> = |n_buf> - a1|n1>
    coefb(1) = norm(n_buf)             ! b2 = |m2|
    n2 = n_buf / coefb(1)              ! n2 = m2 / b2

    ! initialize pointers
    p1 => n1
    p2 => n2
    p_buf => n_buf

    ! recursion
    DO i = 2, n_depth
        IF (MODULO(i, 1000) == 0) THEN
            PRINT *, "    Depth    ", i, " of ", n_depth
        END IF

        p_buf = amv(H_rescale, H_csr, p2)  ! n_buf = H|ni>
        coefa(i) = inner_prod(p2, p_buf)   ! ai = <ni|H|ni> = <n1|n_buf>
        ! mi+1 = (H - ai)|ni> - bi|ni-1> = n_buf - ai|ni> - bi|ni-1>
        CALL axpbypz(-coefa(i), p2, -coefb(i-1), p1, p_buf)
        coefb(i) = norm(p_buf)             ! bi+1 = |mi+1|
        p_buf = p_buf / coefb(i)           ! ni+1 = mi+1 / bi+1

        ! update pointers
        p_swap => p1
        p1 => p2
        p2 => p_buf
        p_buf => p_swap
    END DO
END SUBROUTINE Haydock_coef

END MODULE propagation
