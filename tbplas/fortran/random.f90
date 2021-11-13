! --------------------------------------
! random state generator with given seed
! --------------------------------------

MODULE random

    IMPLICIT NONE

    PRIVATE :: init_seed

CONTAINS

! Initialize random seed
SUBROUTINE init_seed(idum)
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: idum

    ! declare vars
    INTEGER :: i, n
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed

    CALL RANDOM_SEED(size=n)
    ALLOCATE(seed(n))

    ! is there a better way to create a seed array
    ! based on the input integer?
    DO i = 1, n
        seed(i)=INT(MODULO(i * idum * 74231, 104717))
    END DO

    CALL RANDOM_SEED(put=seed)
END SUBROUTINE init_seed

! Make random initial state
SUBROUTINE random_state(wf, n_wf, iseed)
    USE const
    IMPLICIT NONE
    ! input
    INTEGER, INTENT(IN) :: n_wf, iseed
    ! output
    COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_wf) :: wf

    ! declare vars
    INTEGER :: i, iseed0
    REAL(KIND=8) :: f, g, abs_z_sq, sum_wf_sq

    ! make random wf
    iseed0 = iseed * 49741

    CALL init_seed(iseed0)

    sum_wf_sq = 0D0
    ! NOTE: openmp must be disabled for this section. Otherwise results will
    ! be dependent on the number of threads.
    DO i = 1, n_wf
        CALL RANDOM_NUMBER(f)
        CALL RANDOM_NUMBER(g)
        abs_z_sq = -1D0 * LOG(1D0 - f) ! exponential distribution
        wf(i) = SQRT(abs_z_sq) * EXP(CMPLX(0D0, 2*pi*g, KIND=8)) ! random phase
        sum_wf_sq = sum_wf_sq + abs_z_sq
    END DO

    !$OMP PARALLEL DO
    DO i = 1, n_wf
        wf(i) = wf(i) / DSQRT(sum_wf_sq)
    END DO
    !$OMP END PARALLEL DO
END SUBROUTINE random_state

END MODULE random
