! ---------------------------------------
! FFT functions using Cooley-Tukey method
! ---------------------------------------

MODULE fft

	IMPLICIT NONE

CONTAINS

! Cooley-Tukey FFT in non-recursive form
SUBROUTINE fft1d_inplace(x, sgn)

	USE const
	IMPLICIT NONE
	COMPLEX(KIND=8), DIMENSION(:), INTENT(INOUT) :: x
	INTEGER, INTENT(IN) :: sgn

	INTEGER :: n, i, j, k, ncur, ntmp, itmp
	REAL(KIND=8) :: e
	COMPLEX(KIND=8) :: ctmp
	n = SIZE(x)
	ncur = n
	DO
		ntmp = ncur
		e = 2.0 * pi / ncur
		ncur = ncur / 2
		IF ( ncur < 1 ) EXIT
		DO j = 1, ncur
			DO i = j, n, ntmp
				itmp = i + ncur
				ctmp = x(i) - x(itmp)
				x(i) = x(i) + x(itmp)
				x(itmp) = ctmp * EXP(CMPLX(0.0, sgn*e*(j-1), KIND=8))
			END DO
		END DO
	END DO
	j = 1
	DO i = 1, n - 1
		IF ( i < j ) THEN
			ctmp = x(j)
			x(j) = x(i)
			x(i) = ctmp
		END IF
		k = n/2
		DO WHILE( k < j )
			j = j - k
			k = k / 2
		END DO
		j = j + k
	END DO
	RETURN
END SUBROUTINE fft1d_inplace

END MODULE fft
