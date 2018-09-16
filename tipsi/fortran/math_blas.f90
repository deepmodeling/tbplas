! -------------------------------
! math functions, linking to BLAS
! -------------------------------

MODULE math

	IMPLICIT NONE

	! overload inner_prod functions
	INTERFACE inner_prod
		MODULE PROCEDURE dot_c, dot_r, dot_self_c, dot_self_r
	END INTERFACE inner_prod

	INTERFACE
		PURE FUNCTION ddot(n, x, incx, y, incy)
			INTEGER, INTENT(IN) :: n, incx, incy
			REAL(KIND=8), INTENT(IN), DIMENSION(n) :: x, y
			REAL(KIND=8) :: ddot
		END FUNCTION ddot
		PURE FUNCTION zdotc(n, x, incx, y, incy)
			INTEGER, INTENT(IN) :: n, incx, incy
			COMPLEX(KIND=8), INTENT(IN), DIMENSION(n) :: x, y
			COMPLEX(KIND=8) :: zdotc
		END FUNCTION zdotc
	END INTERFACE

	PRIVATE :: dot_c, dot_r, dot_self_c, dot_self_r, ddot, zdotc

CONTAINS

PURE FUNCTION dot_c(A, B)

	IMPLICIT NONE
	! input
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: A, B
	! output
	COMPLEX(KIND=8) :: dot_c

	!declare vars
	INTEGER :: n

	n = SIZE(A)
	dot_c = zdotc(n, A, 1, B, 1)

END FUNCTION dot_c

PURE FUNCTION dot_r(A, B)

	IMPLICIT NONE
	! input
	REAL(KIND=8), INTENT(IN), DIMENSION(:) :: A, B
	! output
	REAL(KIND=8) :: dot_r

	!declare vars
	INTEGER :: n

	n = SIZE(A)
	dot_r = ddot(n, A, 1, B, 1)

END FUNCTION dot_r

PURE FUNCTION dot_self_c(A)

	IMPLICIT NONE
	! input
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: A
	! output
	REAL(KIND=8) :: dot_self_c

	!declare vars
	INTEGER :: n

	n = SIZE(A)
	dot_self_c = DBLE(zdotc(n, A, 1, A, 1))

END FUNCTION dot_self_c

PURE FUNCTION dot_self_r(A)

	IMPLICIT NONE
	! input
	REAL(KIND=8), INTENT(IN), DIMENSION(:) :: A
	! output
	REAL(KIND=8) :: dot_self_r

	!declare vars
	INTEGER :: n

	n = SIZE(A)
	dot_self_r = ddot(n, A, 1, A, 1)

END FUNCTION dot_self_r

END MODULE math
