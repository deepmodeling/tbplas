! -----------------------
! propagation subroutines
! -----------------------

MODULE propagation

	IMPLICIT NONE

CONTAINS

! Apply timestep using Chebyshev decomposition
SUBROUTINE cheb_wf_timestep(wf_t, n_wf, Bes, value, H_csr, wf_t1)

	USE const, ONLY: img
	USE csr
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_wf
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_wf) :: wf_t
	REAL(KIND=8), INTENT(IN), DIMENSION(:) :: Bes
	REAL(KIND=8), INTENT(IN) :: value
	TYPE(SPARSE_MATRIX_T), INTENT(IN) :: H_csr
	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_wf) :: wf_t1

	! declare vars
	INTEGER :: i, j
	COMPLEX(KIND=8), DIMENSION(n_wf), TARGET :: Tcheb0, Tcheb1, Tcheb2
	COMPLEX(KIND=8), DIMENSION(:), POINTER :: p0, p1, p2

	CALL csr_mv(wf_t, n_wf, value, H_csr, Tcheb1)

	!$OMP PARALLEL DO
	DO i = 1, n_wf
		Tcheb0(i) = wf_t(i)
		Tcheb1(i) = -img * Tcheb1(i)
		wf_t1(i) = Bes(1) * Tcheb0(i) + 2 * Bes(2) * Tcheb1(i)
	END DO
	!$OMP END PARALLEL DO

	p0 => Tcheb0
	p1 => Tcheb1
	DO j = 3, SIZE(Bes)
		p2 => p0
		CALL csr_mv(p1, n_wf, value, H_csr, Tcheb2)

		!$OMP PARALLEL DO
		DO i = 1, n_wf
			p2(i) = p0(i) - 2 * img * Tcheb2(i)
			wf_t1(i) = wf_t1(i) + 2 * Bes(j) * p2(i)
		END DO
		!$OMP END PARALLEL DO
		p0 => p1
		p1 => p2
	END DO

END SUBROUTINE cheb_wf_timestep

! Fermi-Dirac distribution operator
SUBROUTINE Fermi(wf_in, n_wf, cheb_coef, H_csr, wf_out)

	USE csr
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_wf
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_wf) :: wf_in
	REAL(KIND=8), INTENT(IN), DIMENSION(:) :: cheb_coef
	TYPE(SPARSE_MATRIX_T), INTENT(IN) :: H_csr
	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_wf) :: wf_out

	! declare vars
	INTEGER :: i, j
	COMPLEX(KIND=8), DIMENSION(n_wf), TARGET :: Tcheb0, Tcheb1, Tcheb2
	COMPLEX(KIND=8), DIMENSION(:), POINTER :: p0, p1, p2

	CALL csr_mv(wf_in, n_wf, 1D0, H_csr, Tcheb1)

	!$OMP PARALLEL DO
	DO i = 1, n_wf
		Tcheb0(i) = wf_in(i)
		wf_out(i) = cheb_coef(1) * Tcheb0(i) + cheb_coef(2) * Tcheb1(i)
	END DO
	!$OMP END PARALLEL DO

	p0 => Tcheb0
	p1 => Tcheb1
	DO j = 3, SIZE(cheb_coef)
		p2 => p0
		CALL csr_mv(p1, n_wf, 1D0, H_csr, Tcheb2)

		!$OMP PARALLEL DO
		DO i = 1, n_wf
			p2(i) = 2 * Tcheb2(i) - p0(i)
			wf_out(i) = wf_out(i) + cheb_coef(j) * p2(i)
		END DO
		!$OMP END PARALLEL DO
		p0 => p1
		p1 => p2
	END DO

END SUBROUTINE Fermi

! Get Haydock coefficients using Haydock recursion method
SUBROUTINE Haydock_coef(n1, n_wf, n_depth, H_csr, H_rescale, coefa, coefb)

	USE math, ONLY: inner_prod
	USE csr
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_wf, n_depth
	REAL(KIND=8), INTENT(IN) :: H_rescale
	COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(n_wf) :: n1
	TYPE(SPARSE_MATRIX_T), INTENT(IN) :: H_csr
	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_depth) :: coefa
	REAL(KIND=8), INTENT(OUT), DIMENSION(n_depth) :: coefb

	! declare variables
	INTEGER :: i, j
	COMPLEX(KIND=8), DIMENSION(n_wf) :: n0, n2

	! get a1
	CALL csr_mv(n1, n_wf, H_rescale, H_csr, n2)
	coefa(1) = inner_prod(n1, n2)

	!$OMP PARALLEL DO
	DO j = 1, n_wf
		n2(j) = n2(j) - coefa(1) * n1(j)
	END DO
	!$OMP END PARALLEL DO

	coefb(1) = DSQRT(inner_prod(n2))

	! recursion
	DO i = 2, n_depth

		IF (MODULO(i,1000) == 0) THEN
			PRINT *, "    Depth    ", i, " of ", n_depth
		END IF

		!$OMP PARALLEL DO
		DO j = 1, n_wf
			n0(j) = n1(j)
			n1(j) = n2(j) / coefb(i-1)
		END DO
		!$OMP END PARALLEL DO

		CALL csr_mv(n1, n_wf, H_rescale, H_csr, n2)
		coefa(i) = inner_prod(n1, n2)

		!$OMP PARALLEL DO
		DO j = 1, n_wf
			n2(j) = n2(j) - coefa(i) * n1(j) - coefb(i-1) * n0(j)
		END DO
		!$OMP END PARALLEL DO

		coefb(i) = DSQRT(inner_prod(n2))
	END DO
END SUBROUTINE Haydock_coef

END MODULE propagation
