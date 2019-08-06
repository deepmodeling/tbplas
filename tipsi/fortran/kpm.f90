MODULE kpm
	IMPLICIT NONE

CONTAINS

! Get array of Jackson kernel
!!! the first element has already been divided by 2
SUBROUTINE jackson_kernel(g_J, n_kernel)
    USE const
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN):: n_kernel
	! output
	REAL(KIND=8), INTENT(OUT) :: g_J(0 : n_kernel-1)

	! declare vars
	INTEGER :: k
	REAL(KIND=8):: cotq, q

    g_J(0) = 0.5
    q = PI / n_kernel
    cotq = DCOS(q) / DSIN(q)

	!$OMP PARALLEL DO SIMD
    DO k = 1, n_kernel - 1
        g_J(k) = ((n_kernel - k) * DCOS(q*k) + cotq * DSIN(q*k)) / n_kernel
    END DO
	!$OMP END PARALLEL DO SIMD

END SUBROUTINE jackson_kernel

! calculate Gamma_mn
SUBROUTINE get_gamma_mn(x, n_kernel, Gamma_mn)
	USE const, ONLY: img
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_kernel
	REAL(KIND=8), INTENT(IN) :: x
	! output
	COMPLEX(kind=8), INTENT(OUT) :: Gamma_mn(0:n_kernel-1, 0:n_kernel-1)

	!declare vars
	INTEGER :: i, j
	REAL(KIND=8) :: sx, cx

	sx = DSIN(x)
	cx = DCOS(x)

	!$OMP PARALLEL DO PRIVATE(i)
	DO j = 0, n_kernel-1
		DO i=0, n_kernel-1
			Gamma_mn(i, j) = DCMPLX(cx, -j*sx) * DCOS(i*x) * EXP(img*j*x) &
						   + DCMPLX(cx, i*sx) * DCOS(j*x) * EXP(-img*i*x)
		END DO
	END DO
	!$OMP END PARALLEL DO

END SUBROUTINE get_gamma_mn

END MODULE kpm
