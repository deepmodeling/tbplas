! -------------------------------
! functions related to csr matrix
! -------------------------------

#ifdef MKL
    INCLUDE 'mkl_spblas.f90'
#endif

MODULE csr

#ifdef MKL
    USE mkl_spblas
    IMPLICIT NONE
    TYPE(MATRIX_DESCR), PRIVATE, PARAMETER :: &
        H_descr = MATRIX_DESCR(type = SPARSE_MATRIX_TYPE_GENERAL, &
                               mode = SPARSE_FILL_MODE_UPPER, &
                               diag = SPARSE_DIAG_NON_UNIT)
#else
    IMPLICIT NONE
    TYPE :: SPARSE_MATRIX_T
        INTEGER, DIMENSION(:), POINTER :: indptr
        INTEGER, DIMENSION(:), POINTER :: indices
        COMPLEX(KIND=8), DIMENSION(:), POINTER :: values
    END TYPE SPARSE_MATRIX_T
#endif
    ! overload operator for mat*vec
    INTERFACE OPERATOR(*)
        MODULE PROCEDURE csr_mv
    END INTERFACE OPERATOR(*)
    INTERFACE amv
        MODULE PROCEDURE amv_d, amv_z
    END INTERFACE amv
    INTERFACE amxpy
        MODULE PROCEDURE amxpy_d, amxpy_z
    END INTERFACE amxpy
    INTERFACE amxpby
        MODULE PROCEDURE amxpby_d, amxpby_z
    END INTERFACE amxpby

    PRIVATE :: csr_mv, amv_d, amv_z, amxpy_d, amxpy_z, amxpby_d, amxpby_z

CONTAINS

! Build csr matrix
FUNCTION make_csr_matrix(indptr, indices, values) RESULT(mat)
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN), DIMENSION(:), TARGET :: indptr, indices
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:), TARGET :: values
    ! output
    TYPE(SPARSE_MATRIX_T) :: mat
#ifdef MKL
    ! declare vars
    INTEGER :: n_wf, t
    INTEGER, DIMENSION(:), POINTER :: rows_start, rows_end

    n_wf = SIZE(indptr) - 1
    rows_start => indptr(1:n_wf)
    rows_end => indptr(2:n_wf + 1)

    t = mkl_sparse_z_create_csr(mat, SPARSE_INDEX_BASE_ZERO, n_wf, &
                                n_wf, rows_start, rows_end, indices, values)
#else
    mat%indptr => indptr
    mat%indices => indices
    mat%values => values
#endif
END FUNCTION make_csr_matrix

! Calculate mat*in
FUNCTION csr_mv(mat, in) RESULT(out)
    USE const, ONLY: zero_cmp, one_cmp
    IMPLICIT NONE
    ! input
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: in
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(in)) :: out
#ifdef MKL
    ! declare vars
    INTEGER :: t

    !use Sparse BLAS in MKL to calculate the product
    t = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, one_cmp, mat, &
                        H_descr, in, zero_cmp, out)
#else
    ! declare vars
    INTEGER :: i, j, k

    out(:) = zero_cmp
    !$OMP PARALLEL DO PRIVATE(j, k)
    ! Note: fortran indexing is off by 1
    DO i = 1, SIZE(in)
        DO j = mat%indptr(i) + 1, mat%indptr(i + 1)
            k = mat%indices(j) + 1
            out(i) = mat%values(j) * in(k) + out(i)
        END DO
    END DO
    !$OMP END PARALLEL DO
#endif
END FUNCTION csr_mv

! Calculate a*mat*in
FUNCTION amv_d(a, mat, in) RESULT(out)
    USE const, ONLY: zero_cmp
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: in
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(in)) :: out
#ifdef MKL
    ! declare vars
    INTEGER :: t
    COMPLEX(KIND=8) :: alpha

    alpha = CMPLX(a, KIND=8)
    !use Sparse BLAS in MKL to calculate the product
    t = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat, &
                        H_descr, in, zero_cmp, out)
#else
    ! declare vars
    INTEGER :: i, j, k

    out(:) = zero_cmp
    !$OMP PARALLEL DO PRIVATE(j, k)
    ! Note: fortran indexing is off by 1
    DO i = 1, SIZE(in)
        DO j = mat%indptr(i) + 1, mat%indptr(i + 1)
            k = mat%indices(j) + 1
            out(i) = a * mat%values(j) * in(k) + out(i)
        END DO
    END DO
    !$OMP END PARALLEL DO
#endif
END FUNCTION amv_d

! Calculate a*mat*in
FUNCTION amv_z(a, mat, in) RESULT(out)
    USE const, ONLY: zero_cmp
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN) :: a
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: in
    ! output
    COMPLEX(KIND=8), DIMENSION(SIZE(in)) :: out
#ifdef MKL
    ! declare vars
    INTEGER :: t

    !use Sparse BLAS in MKL to calculate the product
    t = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, a, mat, &
                        H_descr, in, zero_cmp, out)
#else
    ! declare vars
    INTEGER :: i, j, k

    out(:) = zero_cmp
    !$OMP PARALLEL DO PRIVATE(j, k)
    ! Note: fortran indexing is off by 1
    DO i = 1, SIZE(in)
        DO j = mat%indptr(i) + 1, mat%indptr(i + 1)
            k = mat%indices(j) + 1
            out(i) = a * mat%values(j) * in(k) + out(i)
        END DO
    END DO
    !$OMP END PARALLEL DO
#endif
END FUNCTION amv_z

! Calculate a*mat*in + out
SUBROUTINE amxpy_d(a, mat, in, out)
    USE const, ONLY: one_cmp
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: in
    ! output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: out
#ifdef MKL
    ! declare vars
    INTEGER :: t
    COMPLEX(KIND=8) :: alpha

    alpha = CMPLX(a, KIND=8)
    !use Sparse BLAS in MKL to calculate the product
    t = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat, &
                        H_descr, in, one_cmp, out)
#else
    ! declare vars
    INTEGER :: i, j, k

    !$OMP PARALLEL DO PRIVATE(j, k)
    ! Note: fortran indexing is off by 1
    DO i = 1, SIZE(in)
        DO j = mat%indptr(i) + 1, mat%indptr(i + 1)
            k = mat%indices(j) + 1
            out(i) = a * mat%values(j) * in(k) + out(i)
        END DO
    END DO
    !$OMP END PARALLEL DO
#endif
END SUBROUTINE amxpy_d

! Calculate a*mat*in + out
SUBROUTINE amxpy_z(a, mat, in, out)
    USE const, ONLY: one_cmp
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN) :: a
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: in
    ! output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: out
#ifdef MKL
    ! declare vars
    INTEGER :: t

    !use Sparse BLAS in MKL to calculate the product
    t = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, a, mat, &
                        H_descr, in, one_cmp, out)
#else
    ! declare vars
    INTEGER :: i, j, k

    !$OMP PARALLEL DO PRIVATE(j, k)
    ! Note: fortran indexing is off by 1
    DO i = 1, SIZE(in)
        DO j = mat%indptr(i) + 1, mat%indptr(i + 1)
            k = mat%indices(j) + 1
            out(i) = a * mat%values(j) * in(k) + out(i)
        END DO
    END DO
    !$OMP END PARALLEL DO
#endif
END SUBROUTINE amxpy_z

! Calculate a*mat*in + b*out
SUBROUTINE amxpby_d(a, mat, in, b, out)
    IMPLICIT NONE
    ! input
    REAL(KIND=8), INTENT(IN) :: a, b
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: in
    ! output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: out
#ifdef MKL
    ! declare vars
    INTEGER :: t
    COMPLEX(KIND=8) :: alpha, beta

    alpha = CMPLX(a, KIND=8)
    beta = CMPLX(b, KIND=8)
    !use Sparse BLAS in MKL to calculate the product
    t = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat, &
                        H_descr, in, beta, out)
#else
    ! declare vars
    INTEGER :: i, j, k

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! WARNING: it should be 'out(i) = b * out(i)' at the begining of the loop
    ! over i, followed by 'out(i) = a * mat%values(j + 1) * in(k + 1) + out(i)'
    ! in the loop over j. Otherwise the results will be wrong, causing 'Fermi'
    ! subroutine to yield diverged results, affecting many calculations, e.g.
    ! AC conductivity.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !$OMP PARALLEL DO PRIVATE(j, k)
    ! Note: fortran indexing is off by 1
    DO i = 1, SIZE(in)
        out(i) = b * out(i)
        DO j = mat%indptr(i) + 1, mat%indptr(i + 1)
            k = mat%indices(j) + 1
            out(i) = a * mat%values(j) * in(k) + out(i)
        END DO
    END DO
    !$OMP END PARALLEL DO
#endif
END SUBROUTINE amxpby_d

! Calculate a*mat*in + out
SUBROUTINE amxpby_z(a, mat, in, b, out)
    IMPLICIT NONE
    ! input
    COMPLEX(KIND=8), INTENT(IN) :: a, b
    TYPE(SPARSE_MATRIX_T), INTENT(IN) :: mat
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: in
    ! output
    COMPLEX(KIND=8), INTENT(INOUT), DIMENSION(:) :: out
#ifdef MKL
    ! declare vars
    INTEGER :: t

    !use Sparse BLAS in MKL to calculate the product
    t = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, a, mat, &
                        H_descr, in, b, out)
#else
    ! declare vars
    INTEGER :: i, j, k

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! WARNING: it should be 'out(i) = b * out(i)' at the begining of the loop
    ! over i, followed by 'out(i) = a * mat%values(j + 1) * in(k + 1) + out(i)'
    ! in the loop over j. Otherwise the results will be wrong, causing 'Fermi'
    ! subroutine to yield diverged results, affecting many calculations, e.g.
    ! AC conductivity.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !$OMP PARALLEL DO PRIVATE(j, k)
    ! Note: fortran indexing is off by 1
    DO i = 1, SIZE(in)
        out(i) = b * out(i)
        DO j = mat%indptr(i) + 1, mat%indptr(i + 1)
            k = mat%indices(j) + 1
            out(i) = a * mat%values(j) * in(k) + out(i)
        END DO
    END DO
    !$OMP END PARALLEL DO
#endif
END SUBROUTINE amxpby_z

END MODULE csr
