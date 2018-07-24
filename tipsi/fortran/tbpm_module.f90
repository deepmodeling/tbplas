! ------------------------------------------
! MODULE with helper functions for tbpm_f2py
! ------------------------------------------

MODULE tbpm_mod

	IMPLICIT NONE
	REAL(8), PARAMETER :: pi = 3.14159265358979323846264338327950D0
	COMPLEX(8), PARAMETER :: img = CMPLX(0.0d0, 1.0d0, kind=8)

CONTAINS

! Scalar product
COMPLEX(8) FUNCTION inner_prod(A, B, N)

	IMPLICIT NONE
	INTEGER, INTENT(in) :: N
	COMPLEX(8), INTENT(in), DIMENSION(N) :: A, B
	inner_prod = dot_PRODUCT(A, B)

END FUNCTION inner_prod

! Cooley-Tukey FFT
SUBROUTINE fft(x, sgn)
	COMPLEX(kind=8), INTENT(inout) :: x(:)
	INTEGER, INTENT(in) :: sgn

	INTEGER :: n, i, j, k, ncur, ntmp, itmp
	REAL(kind=8) :: e
	COMPLEX(kind=8) :: ctmp
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
				x(itmp) = ctmp * EXP(CMPLX(0.0, sgn*e*(j-1), kind=8))
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
END SUBROUTINE fft

! Hamiltonian operator
SUBROUTINE Hamiltonian(wf_in, n_wf, s_indptr, n_indptr, s_indices, &
		n_indices, s_hop, n_hop, wf_out)

	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(in) :: n_wf, n_indptr, n_indices, n_hop
	COMPLEX(8), INTENT(in), DIMENSION(n_wf) :: wf_in
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_wf) :: wf_out

	! declare vars
	INTEGER :: i, j, k, j_start, j_end

	wf_out = 0.0d0

	!$OMP parallel do private (j,k)
	! Nota bene: fortran indexing is off by 1
	DO i = 1, n_wf
		j_start = s_indptr(i)
		j_end = s_indptr(i + 1)
		DO j = j_start, j_end - 1
			k = s_indices(j + 1)
			wf_out(i) = wf_out(i) + s_hop(j + 1)* wf_in(k + 1)
		END DO
	END DO
	!$OMP end parallel do

END SUBROUTINE Hamiltonian

! Apply timestep using Chebyshev decomposition
SUBROUTINE cheb_wf_timestep(wf_t, n_wf, Bes, n_Bes, s_indptr, n_indptr, &
							s_indices, n_indices, s_hop, n_hop, wf_t1)

	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(in) :: n_wf, n_Bes, n_indptr, n_indices, n_hop
	COMPLEX(8), INTENT(in), DIMENSION(n_wf) :: wf_t
	REAL(8), INTENT(in), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop

	! declare vars
	INTEGER :: i, k
	REAL(8) :: sum_wf
	COMPLEX(8), DIMENSION(n_wf), TARGET :: Tcheb0, Tcheb1, Tcheb2
	COMPLEX(8), DIMENSION(:), POINTER :: p0, p1, p2

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_wf) :: wf_t1

	CALL Hamiltonian(wf_t, n_wf, s_indptr, &
			n_indptr, s_indices, n_indices, s_hop, n_hop, Tcheb1)

	!$OMP parallel do
	DO i = 1, n_wf
		Tcheb0(i) = wf_t(i)
		Tcheb1(i) = -img * Tcheb1(i)
		wf_t1(i) = Bes(1) * Tcheb0(i) + 2* Bes(2) * Tcheb1(i)
	END DO
	!$OMP end parallel do

	p0 => Tcheb0
	p1 => Tcheb1
	DO k=3, n_Bes
		p2 => p0
		CALL Hamiltonian(p1, n_wf, s_indptr, n_indptr, s_indices, &
						 n_indices, s_hop, n_hop, Tcheb2)

		!$OMP parallel do
		DO i = 1, n_wf
			p2(i) = p0(i) - 2 * img * Tcheb2(i)
			wf_t1(i) = wf_t1(i) + 2 * Bes(k) * p2(i)
		END DO
		!$OMP end parallel do
		p0 => p1
		p1 => p2
	END DO

END SUBROUTINE cheb_wf_timestep

! get coefficients of current operator
SUBROUTINE current_coefficient(hop, dr, n_hop, cur_coefs)

	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(in) :: n_hop
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: hop
	REAL(8), INTENT(in), DIMENSION(n_hop) :: dr
	COMPLEX(8), INTENT(out), DIMENSION(n_hop) :: cur_coefs

	! declare vars
	INTEGER :: i

	!$OMP parallel do
	DO i = 1, n_hop
		cur_coefs(i) = img * hop(i) * dr(i)
	END DO
	!$OMP end parallel do

END SUBROUTINE current_coefficient

! current operator
SUBROUTINE current(wf_in, n_wf, s_indptr, n_indptr, s_indices, &
		n_indices, cur_coefs, n_cur_coefs, wf_out)

	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(in) :: n_wf, n_indptr, n_indices, n_cur_coefs
	COMPLEX(8), INTENT(in), DIMENSION(n_wf) :: wf_in
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_cur_coefs) :: cur_coefs

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_wf) :: wf_out

	! declare vars
	INTEGER :: i, j, k, j_start, j_end

	wf_out = 0.0d0

	!$OMP parallel do private (j, k)
	! Nota bene: fortran indexing is off by 1
	DO i = 1, n_wf
		j_start = s_indptr(i)
		j_end = s_indptr(i + 1)
		DO j = j_start, j_end - 1
			k = s_indices(j + 1)
			wf_out(i) = wf_out(i) + cur_coefs(j + 1)* wf_in(k + 1)
		END DO
	END DO
	!$OMP end parallel do

END SUBROUTINE current

! The actual Fermi distribution
REAL(8) FUNCTION Fermi_dist(beta,Ef,energy,eps)

	IMPLICIT NONE
	REAL(8) :: eps, beta, Ef, energy, x

	IF (energy >= Ef) THEN
		x = 1. * EXP(beta * (Ef - energy))
		Fermi_dist = x / (1 + x)
	ELSE
		x = 1. * EXP(beta * (energy - Ef))
		Fermi_dist = 1 / (1 + x)
	END IF

	IF (Fermi_dist < eps) THEN
		Fermi_dist = 0
	END IF

END FUNCTION Fermi_dist

! compute Chebyshev expansion coefficients of Fermi operator
SUBROUTINE get_Fermi_cheb_coef(cheb_coef, n_cheb, nr_Fermi, &
							   beta, mu, one_minus_Fermi, eps)

	! declarations
	IMPLICIT NONE
	INTEGER, INTENT(in) :: nr_Fermi
	LOGICAL, INTENT(in) :: one_minus_Fermi ! if true: compute coeffs for
	! one minus Fermi operator
	REAL(8), INTENT(in) :: beta, mu, eps
	COMPLEX(8), DIMENSION(nr_Fermi) :: cheb_coef_complex
	REAL(8), INTENT(out), DIMENSION(nr_Fermi) :: cheb_coef
	INTEGER, INTENT(out) :: n_cheb
	REAL(8) :: r0, compare, x, prec, energy
	INTEGER :: i
	r0 = 2 * pi / nr_Fermi

	IF (one_minus_Fermi) THEN ! compute coeffs for one minus Fermi operator
		DO i = 1, nr_Fermi
			energy = COS((i - 1) * r0)
			cheb_coef_complex(i) = 1. - Fermi_dist(beta,mu,energy,eps)
		END DO
	ELSE ! compute coeffs for Fermi operator
		DO i=1, nr_Fermi
			energy = COS((i - 1) * r0)
			cheb_coef_complex(i) = Fermi_dist(beta,mu,energy,eps)
		END DO
	END IF

	! Fourier transform result
	CALL fft(cheb_coef_complex, -1)

	! Get number of nonzero elements
	prec = -LOG10(eps)
	n_cheb = 0
	cheb_coef_complex = 2. * cheb_coef_complex / nr_Fermi
	cheb_coef_complex(1) = cheb_coef_complex(1) / 2
	compare = LOG10(MAXVAL(ABS(cheb_coef_complex(1:nr_Fermi))))-prec
	cheb_coef = DBLE(cheb_coef_complex)
	DO i = 1, nr_Fermi
		IF((LOG10(ABS(cheb_coef(i)))<compare).AND.&
		   (LOG10(ABS(cheb_coef(i+1)))<compare)) THEN
			n_cheb = i
			EXIT
		END IF
	END DO
	IF (n_cheb==0) THEN
		PRINT *,"WARNING: not enough Fermi operator Cheb. coeficcients"
	END IF

END SUBROUTINE get_Fermi_cheb_coef

! Fermi-Dirac distribution operator
SUBROUTINE Fermi(wf_in, n_wf, cheb_coef, n_cheb, s_indptr, n_indptr, &
				 s_indices, n_indices, s_hop, n_hop, wf_out)

	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(in) :: n_wf, n_cheb, n_indptr, n_indices, n_hop
	COMPLEX(8), INTENT(in), DIMENSION(n_wf) :: wf_in
	REAL(8), INTENT(in), DIMENSION(n_cheb) :: cheb_coef
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop

	! declare vars
	INTEGER :: i, k
	REAL(8) :: sum_wf
	COMPLEX(8), DIMENSION(n_wf), TARGET :: Tcheb0, Tcheb1, Tcheb2
	COMPLEX(8), DIMENSION(:), POINTER :: p0, p1, p2

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_wf) :: wf_out

	CALL Hamiltonian(wf_in, n_wf, s_indptr, n_indptr, s_indices, &
					 n_indices, s_hop, n_hop, Tcheb1)

	!$OMP parallel do
	DO i = 1, n_wf
		Tcheb0(i) = wf_in(i)
		wf_out(i) = cheb_coef(1) * Tcheb0(i) + cheb_coef(2) * Tcheb1(i)
	END DO
	!$OMP end parallel do

	p0 => Tcheb0
	p1 => Tcheb1
	DO k=3, n_cheb
		p2 => p0
		CALL Hamiltonian(p1, n_wf, s_indptr, &
						 n_indptr, s_indices, n_indices, s_hop, n_hop, Tcheb2)

		!$OMP parallel do
		DO i = 1, n_wf
			p2(i) = 2 * Tcheb2(i) - p0(i)
			wf_out(i) = wf_out(i) + cheb_coef(k) * p2(i)
		END DO
		!$OMP end parallel do
		p0 => p1
		p1 => p2
	END DO

END SUBROUTINE fermi

SUBROUTINE density_coef(n_wf, site_x, site_y, site_z, q_point, &
						s_density_q, s_density_min_q)

	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(in) :: n_wf
	REAL(8), INTENT(in), DIMENSION(n_wf) :: site_x, site_y, site_z
	REAL(8), INTENT(in), DIMENSION(3) :: q_point
	COMPLEX(8),INTENT(out),DIMENSION(n_wf)::s_density_q
	COMPLEX(8),INTENT(out),DIMENSION(n_wf)::s_density_min_q

	! declare vars
	INTEGER :: i,j
	REAL(8) :: power

	!$OMP parallel do private (power)
	DO i = 1, n_wf
		power = q_point(1)*site_x(i) + &
				q_point(2)*site_y(i) + &
				q_point(3)*site_z(i)
		s_density_q(i) = COS(power) + img*SIN(power)
		s_density_min_q(i) = COS(power) - img*SIN(power)
	END DO
	!$OMP end parallel do

END SUBROUTINE density_coef

! density operator
SUBROUTINE density(wf_in, n_wf, s_density, wf_out)

	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(in) :: n_wf
	COMPLEX(8), INTENT(in), DIMENSION(n_wf) :: s_density
	COMPLEX(8), INTENT(in), DIMENSION(n_wf) :: wf_in

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_wf) :: wf_out

	! declare vars
	INTEGER :: i

	wf_out = 0.0d0

	!$OMP parallel do
	DO i = 1, n_wf
		wf_out(i) = s_density(i) * wf_in(i)
	END DO
	!$OMP end parallel do

END SUBROUTINE density

! Make random initial state
SUBROUTINE random_state(wf, n_wf, iseed)

	! variables
	IMPLICIT NONE
	INTEGER, INTENT(in) :: n_wf, iseed
	COMPLEX(8), INTENT(out), DIMENSION(n_wf) :: wf
	INTEGER :: i, iseed0
	REAL(8) :: f, g, wf_sum, abs_z_sq
	COMPLEX(8) :: z

	! make random wf
	iseed0=iseed*49741

	f=ranx(iseed0)
	wf_sum = 0
	DO i = 1, n_wf
		f=ranx(0)
		g=ranx(0)
		abs_z_sq = -1.0d0 * LOG(1.0d0 - f) ! dirichlet distribution
		z = DSQRT(abs_z_sq)*EXP(img*2*pi*g) ! give random phase
		wf(i) = z
		wf_sum = wf_sum + abs_z_sq
	END DO
	DO i = 1, n_wf
		wf(i) = wf(i)/DSQRT(wf_sum)
	END DO

CONTAINS

	! random number
	FUNCTION ranx(idum)
	INTEGER :: idum, n
	INTEGER, ALLOCATABLE :: seed(:)
	REAL*8 :: ranx
	IF (idum>0) THEN
		CALL random_seed(size=n)
		ALLOCATE(seed(n))
		! is there a better way to create a seed array
		! based on the input integer?
		DO i=1, n
			seed(i)=INT(MODULO(i * idum * 74231, 104717))
		END DO
		CALL random_seed(put=seed)
	END IF
	CALL random_number(ranx)
	END FUNCTION ranx

END SUBROUTINE random_state

! Haydock recursion method
SUBROUTINE recursion(site_indices, n_siteind, wf_weights, n_wfw, n_depth, &
		s_indptr, n_indptr, s_indices, n_indices, &
		s_hop, n_hop, coefa, coefb)
	IMPLICIT NONE
	! deal with input
	INTEGER, INTENT(IN) :: n_depth, n_indptr, n_indices
	INTEGER, INTENT(IN) :: n_hop, n_siteind, n_wfw
	INTEGER, INTENT(IN), DIMENSION(n_siteind) :: site_indices
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	REAL(KIND=8), INTENT(IN), DIMENSION(n_wfw) :: wf_weights
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop

	! declare variables
	INTEGER :: i, j, n_wf
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: n0, n1, n2
	COMPLEX(KIND=8), DIMENSION(n_siteind) :: wf_temp

	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_depth) :: coefa
	REAL(KIND=8), INTENT(OUT), DIMENSION(n_depth) :: coefb

	n_wf = n_indptr - 1
	! make LDOS state
	n1 = 0D0
	wf_temp = 1D0 / DSQRT(DBLE(n_siteind))
	DO i = 1, n_siteind
		n1(site_indices(i) + 1) = wf_temp(i) * wf_weights(i)
	END DO

	! get a1
	CALL Hamiltonian(n1, n_wf, s_indptr, n_indptr, s_indices, &
					 n_indices, s_hop, n_hop, n2)
	coefa(1) = inner_prod(n1, n2, n_wf)

	!$OMP PARALLEL DO
	DO j = 1, n_wf
		n2(j) = n2(j) - coefa(1) * n1(j)
	END DO
	!$OMP END PARALLEL DO

	coefb(1) = DSQRT(DBLE(inner_prod(n2, n2, n_wf)))

	! recursion
	DO i = 2, n_depth
		!$OMP PARALLEL DO
		DO j = 1, n_wf
			n0(j) = n1(j)
			n1(j) = n2(j) / coefb(i-1)
		END DO
		!$OMP END PARALLEL DO

		CALL Hamiltonian(n1, n_wf, s_indptr, n_indptr, s_indices, &
						 n_indices, s_hop, n_hop, n2)
		coefa(i) = inner_prod(n1, n2, n_wf)

		!$OMP PARALLEL DO
		DO j = 1, n_wf
			n2(j) = n2(j) - coefa(i) * n1(j) - coefb(i-1) * n0(j)
		END DO
		!$OMP END PARALLEL DO

		coefb(i) = DSQRT(DBLE(inner_prod(n2, n2, n_wf)))
	END DO
END SUBROUTINE recursion

! Green's function G00(E) using Haydock recursion method
SUBROUTINE green_function(energy, delta, coefa, coefb, n_depth, g00)
	IMPLICIT NONE
	! deal with input
	INTEGER, INTENT(IN) :: n_depth
	REAL(KIND=8), INTENT(IN) :: energy, delta
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_depth) :: coefa
	REAL(KIND=8), INTENT(IN), DIMENSION(n_depth) :: coefb

	! declare variables
	COMPLEX(KIND=8) :: E_cmplx
	INTEGER :: i

	! output
	COMPLEX(KIND=8), INTENT(OUT) :: g00

	E_cmplx = CMPLX(energy, delta, KIND=8)
	g00 = CMPLX(0D0, 0D0)

	DO i = n_depth, 1, -1
		g00 = 1D0 / (E_cmplx - coefa(i) - coefb(i)**2 * g00)
	END DO

END SUBROUTINE green_function

END MODULE tbpm_mod
