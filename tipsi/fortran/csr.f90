! -------------------------------
! functions related to csr matrix
! -------------------------------

MODULE csr

	IMPLICIT NONE
	TYPE :: SPARSE_MATRIX_T
		INTEGER, DIMENSION(:), POINTER :: indptr
		INTEGER, DIMENSION(:), POINTER :: indices
		COMPLEX(KIND=8), DIMENSION(:), POINTER :: values
	END TYPE SPARSE_MATRIX_T

CONTAINS

! Build csr matrix
SUBROUTINE make_csr_matrix(n_wf, n_calls, indptr, indices, values, mat_csr)

	IMPLICIT NONE
	! input
	! n_calls is useless, only to make interface consistent
	INTEGER, INTENT(IN) :: n_wf, n_calls
	INTEGER, INTENT(IN), DIMENSION(:), TARGET :: indptr, indices
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(:), TARGET :: values
	! output
	TYPE(SPARSE_MATRIX_T), INTENT(OUT) :: mat_csr

	mat_csr%indptr => indptr
	mat_csr%indices => indices
	mat_csr%values => values

END SUBROUTINE make_csr_matrix

! Calculate value*mat_csr*vec_in
SUBROUTINE csr_mv(vec_in, n_vec, value, mat_csr, vec_out)

	! input
	IMPLICIT NONE
	INTEGER, INTENT(IN) :: n_vec
	REAL(KIND=8), INTENT(IN) :: value
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_vec) :: vec_in
	TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat_csr
	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_vec) :: vec_out

	! declare vars
	INTEGER :: i, j, j_start, j_end, k

	vec_out = 0D0
	!$OMP PARALLEL DO PRIVATE(j,k)
	! Nota bene: fortran indexing is off by 1
	DO i = 1, n_vec
		j_start = mat_csr%indptr(i)
		j_end = mat_csr%indptr(i + 1)
		DO j = j_start, j_end - 1
			k = mat_csr%indices(j + 1)
			vec_out(i) = vec_out(i) &
						+ value * mat_csr%values(j + 1) * vec_in(k + 1)
		END DO
	END DO
	!$OMP END PARALLEL DO

END SUBROUTINE csr_mv

END MODULE csr
