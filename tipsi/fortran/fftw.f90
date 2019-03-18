! ----------------------------------
! FFTW wrapper, faster FFT functions
! ----------------------------------

MODULE fft

	IMPLICIT NONE

CONTAINS

! FFTW interface
SUBROUTINE fft1d_inplace(x, sgn)

	IMPLICIT NONE
	INTEGER, INTENT(IN) :: sgn
	INTEGER, PARAMETER :: FFTW_ESTIMATE = 64
	COMPLEX(KIND=8), DIMENSION(:), INTENT(INOUT) :: x
	INTEGER(KIND=8) :: plan

	CALL dfftw_plan_dft_1d(plan, SIZE(x), x, x, sgn, FFTW_ESTIMATE)
	CALL dfftw_execute_dft(plan, x, x)
	CALL dfftw_destroy_plan(plan)
END SUBROUTINE fft1d_inplace

END MODULE fft
