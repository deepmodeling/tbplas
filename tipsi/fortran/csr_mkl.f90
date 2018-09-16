! ---------------------------------------------------------
! functions related to csr matrix, using Sparse BLAS in MKL
! ---------------------------------------------------------

INCLUDE 'mkl_spblas.f90'

MODULE csr

	USE mkl_spblas
	IMPLICIT NONE
	TYPE(MATRIX_DESCR), PRIVATE, PARAMETER :: &
		H_descr = MATRIX_DESCR(type = SPARSE_MATRIX_TYPE_GENERAL, &
							   mode = SPARSE_FILL_MODE_UPPER, &
							   diag = SPARSE_DIAG_NON_UNIT)

CONTAINS

! Build csr matrix
SUBROUTINE make_csr_matrix(n_wf, n_calls, indptr, indices, values, mat_csr)

	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_wf, n_calls
	INTEGER, INTENT(IN), DIMENSION(:), TARGET :: indptr
	INTEGER, INTENT(IN), DIMENSION(:) :: indices
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: values
	! output
	TYPE(SPARSE_MATRIX_T), INTENT(OUT) :: mat_csr

	! declare vars
	INTEGER :: stat
	INTEGER, DIMENSION(:), POINTER :: rows_start, rows_end

	rows_start => indptr(1:n_wf)
	rows_end => indptr(2:n_wf + 1)

	stat = mkl_sparse_z_create_csr(mat_csr, SPARSE_INDEX_BASE_ZERO, n_wf, &
								   n_wf, rows_start, rows_end, indices, values)
	! set optimizations
	stat = mkl_sparse_set_mv_hint(mat_csr, SPARSE_OPERATION_NON_TRANSPOSE, &
								  H_descr, n_calls)
	stat = mkl_sparse_set_memory_hint(mat_csr, SPARSE_MEMORY_AGGRESSIVE)
	stat = mkl_sparse_optimize(mat_csr)

END SUBROUTINE make_csr_matrix

! Calculate value*mat_csr*vec_in
SUBROUTINE csr_mv(vec_in, n_vec, value, mat_csr, vec_out)

	USE const, ONLY: zero_cmp
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_vec
	REAL(KIND=8), INTENT(IN) :: value
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_vec) :: vec_in
	TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat_csr

	! the status
	INTEGER :: stat
	COMPLEX(KIND=8) :: alpha

	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_vec) :: vec_out

	alpha = CMPLX(value, 0D0, KIND=8)
	!use Sparse BLAS in MKL to calculate the product
	stat = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_csr, &
						   H_descr, vec_in, zero_cmp, vec_out)
END SUBROUTINE csr_mv

END MODULE csr
