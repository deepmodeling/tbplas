! -----------------------------------------
! math functions, using intrinsic functions
! -----------------------------------------

MODULE math

	IMPLICIT NONE

	! inner_prod function
	INTERFACE inner_prod
		MODULE PROCEDURE dot_c, dot_r, dot_self_c, dot_self_r
	END INTERFACE inner_prod

	PRIVATE :: dot_c, dot_r, dot_self_c, dot_self_r

CONTAINS

PURE FUNCTION dot_c(vec1, vec2)

	IMPLICIT NONE
	! input
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: vec1, vec2
	! output
	COMPLEX(KIND=8) :: dot_c

	dot_c = DOT_PRODUCT(vec1, vec2)

END FUNCTION dot_c

PURE FUNCTION dot_r(vec1, vec2)

	IMPLICIT NONE
	! input
	REAL(KIND=8), INTENT(IN), DIMENSION(:) :: vec1, vec2
	! output
	REAL(KIND=8) :: dot_r

	dot_r = DOT_PRODUCT(vec1, vec2)

END FUNCTION dot_r

PURE FUNCTION dot_self_c(vec)

	IMPLICIT NONE
	! input
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(:) :: vec
	! output
	REAL(KIND=8) :: dot_self_c

	dot_self_c = DBLE(DOT_PRODUCT(vec, vec))

END FUNCTION dot_self_c

PURE FUNCTION dot_self_r(vec)

	IMPLICIT NONE
	! input
	REAL(KIND=8), INTENT(IN), DIMENSION(:) :: vec
	! output
	REAL(KIND=8) :: dot_self_r

	dot_self_r = DBLE(DOT_PRODUCT(vec, vec))

END FUNCTION dot_self_r

END MODULE math
