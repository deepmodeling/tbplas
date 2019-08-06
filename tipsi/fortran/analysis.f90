! ------------------------------------------------------
! fortran subroutines for analysis, callable from python
! ------------------------------------------------------

! Get LDOS using Haydock recursion method
SUBROUTINE ldos_haydock(site_indices, n_siteind, wf_weights, n_wfw, delta, &
						E_range, s_indptr, n_indptr, s_indices, n_indices, &
						s_hop, n_hop, H_rescale, seed, n_depth, n_timestep, &
						n_ran_samples, output_filename, energy, ldos)

	USE const
	USE random, ONLY: random_state
	USE csr
	USE propagation, ONLY: Haydock_coef
	USE funcs, ONLY: green_function
	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(IN) :: n_siteind, n_indptr, n_indices, n_hop, seed
	INTEGER, INTENT(IN) :: n_depth, n_timestep, n_wfw, n_ran_samples
	INTEGER, INTENT(IN), DIMENSION(n_siteind) :: site_indices
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	REAL(KIND=8), INTENT(IN) :: E_range, delta, H_rescale
	REAL(KIND=8), INTENT(IN), DIMENSION(n_wfw) :: wf_weights
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
	CHARACTER*(*), INTENT(IN) :: output_filename

	! declare variables
	COMPLEX(KIND=8) :: g00
	INTEGER :: i, n_wf, i_sample
	COMPLEX(KIND=8), DIMENSION(n_siteind) :: wf_temp
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0
	COMPLEX(KIND=8), DIMENSION(n_depth) :: a, coefa
	REAL(KIND=8), DIMENSION(n_depth) :: b, coefb
	TYPE(SPARSE_MATRIX_T) :: H_csr

	! output
	REAL(KIND=8), INTENT(OUT), DIMENSION(-n_timestep:n_timestep) :: energy
	REAL(KIND=8), INTENT(OUT), DIMENSION(-n_timestep:n_timestep) :: ldos

	n_wf = n_indptr - 1
	CALL make_csr_matrix(n_wf, n_depth, s_indptr, s_indices, s_hop, H_csr)
	energy = (/(0.5*i*E_range/n_timestep, i = -n_timestep, n_timestep)/)

	PRINT *, "Getting Haydock coefficients."
	coefa = 0D0
	coefb = 0D0
	DO i_sample = 1, n_ran_samples
		PRINT *, "Sample ", i_sample, " of ", n_ran_samples

		! make LDOS state
		wf0 = 0D0
		CALL random_state(wf_temp, n_siteind, seed*i_sample)
		DO i = 1, n_siteind
			wf0(site_indices(i) + 1) = wf_temp(i) * wf_weights(i)
		END DO

		CALL Haydock_coef(wf0, n_wf, n_depth, H_csr, H_rescale, a, b)
		coefa = coefa + a / n_ran_samples
		coefb = coefb + b / n_ran_samples
	END DO

	PRINT *, "Calculating LDOS with Green's function."
	!$OMP PARALLEL DO SIMD PRIVATE(g00)
	DO i = -n_timestep, n_timestep
		CALL green_function(energy(i), delta, coefa, coefb, n_depth, g00)
		ldos(i) = -1D0 / pi * AIMAG(g00)
	END DO
	!$OMP END PARALLEL DO SIMD

END SUBROUTINE ldos_haydock


!calculates everything after the calculation of the trace
SUBROUTINE cond_from_trace(mu_mn, n_kernel, mu, n_mu, &
						   H_rescale, beta, fermi_precision, prefactor, cond)

	USE const, ONLY: PI
	USE kpm
	USE funcs, ONLY: Fermi_dist
	IMPLICIT NONE

	! input
	INTEGER, INTENT(IN) :: n_kernel, n_mu
	REAL(KIND=8), INTENT(IN) :: H_rescale, beta, fermi_precision, prefactor
	REAL(KIND=8), INTENT(IN), DIMENSION(n_mu) :: mu
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_kernel, n_kernel) :: mu_mn
	!output
	REAL(KIND=8), INTENT(OUT), DIMENSION(n_mu) :: cond

	!declare vars
	INTEGER :: i, j, k, NE_integral
	REAL(KIND=8), DIMENSION(4*n_kernel-1) :: energy
	COMPLEX(KIND=8), DIMENSION(n_kernel, n_kernel) :: Gamma_mn
	REAL(KIND=8), DIMENSION(4*n_kernel-1) :: sum_gamma_mu
	REAL(KIND=8) :: dcx, fd, en, div, dE, sum

	cond = 0D0
	! NE_integral = 2 * n_kernel - 1
	NE_integral = 4 * n_kernel - 1
	dE = PI / (NE_integral+1)

	PRINT*, "Calculating sum"
	DO k = 1, NE_integral
		energy(k) = k * dE
		CALL get_gamma_mn(energy(k), n_kernel, Gamma_mn)

		sum = 0D0
		!$OMP PARALLEL DO PRIVATE(i) REDUCTION(+: sum)
		DO j = 1, n_kernel
			DO i = 1, n_kernel
				sum = sum + DBLE(Gamma_mn(i,j) * mu_mn(i,j))
			END DO
		END DO
		!$OMP END PARALLEL DO
		sum_gamma_mu(k) = sum
	END DO

	PRINT*,"Final integral"
	DO i = 1, n_mu
		dcx = 0D0

		!$OMP PARALLEL DO SIMD PRIVATE(en, div, fd) REDUCTION(+: dcx)
		DO k = 1, NE_integral
			div = DSIN(energy(k))**3
			en = DCOS(energy(k)) * H_rescale
			fd = Fermi_dist(beta, mu(i), en, fermi_precision)
			dcx = dcx + sum_gamma_mu(k) * fd * dE / div
		END DO
		!$OMP END PARALLEL DO SIMD

		cond(i) = dcx * prefactor / H_rescale / H_rescale
	END DO

END SUBROUTINE cond_from_trace
