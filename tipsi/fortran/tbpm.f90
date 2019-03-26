! ----------------------------------------------
! TBPM fortran subroutines, callable from python
! ----------------------------------------------

! Get DOS
SUBROUTINE tbpm_dos(Bes, n_Bes, s_indptr, n_indptr, s_indices, n_indices, &
					s_hop, n_hop, seed, n_timestep, n_ran_samples, &
					output_filename, corr)

	USE math, ONLY: inner_prod
	USE random
	USE csr
	USE propagation, ONLY: cheb_wf_timestep
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_Bes, n_indptr, n_indices, n_hop
	INTEGER, INTENT(IN) :: n_timestep, n_ran_samples, seed
	REAL(KIND=8), INTENT(IN), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
	CHARACTER*(*), INTENT(IN) :: output_filename
	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_timestep) :: corr

	! declare vars
	INTEGER :: i_sample,k, n_wf, n_calls
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf_t
	COMPLEX(KIND=8) :: corrval
	TYPE(SPARSE_MATRIX_T) :: H_csr

	! set some values
	corr = 0D0
	n_wf = n_indptr - 1
	n_calls = n_ran_samples * n_timestep * (n_Bes - 1)
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, s_hop, H_csr)

	OPEN(unit=27,file=output_filename)
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	PRINT *, "Calculating DOS correlation function."

	! Average over (n_ran_samples) samples
	DO i_sample = 1, n_ran_samples

		PRINT *, "Sample ", i_sample, " of ", n_ran_samples
		WRITE(27,*) "Sample =", i_sample

		! make random state
		CALL random_state(wf0, n_wf, seed*i_sample)
		CALL cheb_wf_timestep(wf0, n_wf, Bes, 1D0, H_csr, wf_t)
		corrval = inner_prod(wf0, wf_t)

		WRITE(27,*) 1, REAL(corrval), AIMAG(corrval)
		corr(1) = corr(1) + corrval / n_ran_samples

		! iterate over time, get correlation function
		DO k = 2, n_timestep
			IF (MODULO(k, 64) == 0) THEN
				PRINT *, "    Timestep ", k, " of ", n_timestep
			END IF

			CALL cheb_wf_timestep(wf_t, n_wf, Bes, 1D0, H_csr, wf_t)
			corrval = inner_prod(wf0, wf_t)

			WRITE(27,*) k, REAL(corrval), AIMAG(corrval)
			corr(k) = corr(k) + corrval / n_ran_samples
		END DO
	END DO

	CLOSE(27)

END SUBROUTINE tbpm_dos


! Get LDOS
SUBROUTINE tbpm_ldos(site_indices, n_siteind, wf_weights, n_wfw, &
					 Bes, n_Bes, s_indptr, n_indptr, s_indices, n_indices, &
					 s_hop, n_hop, seed, n_timestep, n_ran_samples, &
					 output_filename, corr)

	USE math, ONLY: inner_prod
	USE random
	USE csr
	USE propagation, ONLY: cheb_wf_timestep
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_Bes, n_indptr, n_indices, n_hop
	INTEGER, INTENT(IN) :: n_timestep, seed, n_siteind, n_wfw
	INTEGER, INTENT(IN) :: n_ran_samples
	INTEGER, INTENT(IN), DIMENSION(n_siteind) :: site_indices
	REAL(KIND=8), INTENT(IN), DIMENSION(n_wfw) :: wf_weights
	REAL(KIND=8), INTENT(IN), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
	CHARACTER*(*), INTENT(IN) :: output_filename
	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_timestep) :: corr

	! declare vars
	INTEGER :: k, i, n_wf, i_sample, n_calls
	COMPLEX(KIND=8), DIMENSION(n_siteind) :: wf_temp
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf_t
	COMPLEX(KIND=8) :: corrval
	TYPE(SPARSE_MATRIX_T) :: H_csr

	! set some values
	corr = 0D0
	n_wf = SIZE(s_indptr) - 1
	n_calls = n_ran_samples * n_timestep * (n_Bes - 1)
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, s_hop, H_csr)

	OPEN(unit=27,file=output_filename)
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	PRINT *, "Calculating LDOS correlation function."

	DO i_sample=1, n_ran_samples

		PRINT *, "Sample ", i_sample, " of ", n_ran_samples
		WRITE(27,*) "Sample =", i_sample

		! make LDOS state
		CALL random_state(wf_temp, n_siteind, seed*i_sample)
		wf0 = 0D0
		DO i = 1, SIZE(site_indices)
			wf0(site_indices(i) + 1) = wf_temp(i) * wf_weights(i)
		END DO
		CALL cheb_wf_timestep(wf0, n_wf, Bes, 1D0, H_csr, wf_t)
		corrval = inner_prod(wf0, wf_t)

		WRITE(27,*) 1, REAL(corrval), AIMAG(corrval)
		corr(1) = corr(1) + corrval / n_ran_samples

		! iterate over time, get correlation function
		DO k = 2, n_timestep
			IF (MODULO(k, 64) == 0) THEN
				PRINT *, "    Timestep ", k, " of ", n_timestep
			END IF

			CALL cheb_wf_timestep(wf_t, n_wf, Bes, 1D0, H_csr, wf_t)
			corrval = inner_prod(wf0, wf_t)

			WRITE(27,*) k, REAL(corrval), AIMAG(corrval)
			corr(k) = corr(k) + corrval / n_ran_samples
		END DO
	END DO

	CLOSE(27)

END SUBROUTINE tbpm_ldos


! Get AC conductivity
SUBROUTINE tbpm_accond(Bes, n_Bes, beta, mu, s_indptr, n_indptr, &
					   s_indices, n_indices, s_hop, n_hop, H_rescale, &
					   s_dx, n_dx, s_dy, n_dy, seed, n_timestep, &
					   n_ran_samples, nr_Fermi, Fermi_precision, &
					   output_filename, corr)

	USE math, ONLY: inner_prod
	USE random
	USE csr
	USE propagation, ONLY: cheb_wf_timestep, Fermi
	USE funcs
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_Bes, n_indptr, n_indices, n_hop, n_dx, n_dy
	INTEGER, INTENT(IN) :: n_timestep, n_ran_samples, seed
	INTEGER, INTENT(IN) :: nr_Fermi
	REAL(KIND=8), INTENT(IN) :: Fermi_precision, H_rescale, beta, mu
	REAL(KIND=8), INTENT(IN), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
	REAL(KIND=8), INTENT(IN), DIMENSION(n_dx) :: s_dx
	REAL(KIND=8), INTENT(IN), DIMENSION(n_dy) :: s_dy
	CHARACTER*(*), INTENT(IN) :: output_filename
	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(4, n_timestep) :: corr
	! corr has 4 elements, respectively: corr_xx, corr_xy, corr_yx, corr_yy

	! declare vars
	INTEGER :: i_sample, k, n_cheb, n_wf, n_calls
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf1
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: psi1_x, psi1_y, psi2
	REAL(KIND=8), DIMENSION(nr_Fermi), TARGET :: coef1, coef2
	COMPLEX(KIND=8), DIMENSION(4) :: corrval
	! cheb coefs for Fermi operator and one minus Fermi operator
	REAL(KIND=8), DIMENSION(:), POINTER :: coef_F, coef_omF
	! coefs for x or y current
	COMPLEX(KIND=8), DIMENSION(n_hop) :: sys_current_x, sys_current_y
	TYPE(SPARSE_MATRIX_T) :: H_csr, cur_csr_x, cur_csr_y

	! set some values
	corr = 0D0
	n_wf = n_indptr - 1

	! prepare output file
	OPEN(unit=27,file=output_filename)
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	! get current coefficients
	n_calls = n_ran_samples * (1 + 2 * n_timestep)
	CALL current_coefficient(s_hop, s_dx, n_hop, H_rescale, sys_current_x)
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, &
						 sys_current_x, cur_csr_x)
	CALL current_coefficient(s_hop, s_dy, n_hop, H_rescale, sys_current_y)
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, &
						 sys_current_y, cur_csr_y)

	! get Fermi cheb coefficients
	CALL get_Fermi_cheb_coef(coef1, n_cheb, nr_Fermi, &
							 beta, mu, .FALSE., Fermi_precision)
	coef_F => coef1(1:n_cheb)

	! get one minus Fermi cheb coefficients
	CALL get_Fermi_cheb_coef(coef2, n_cheb, nr_Fermi, &
							 beta, mu, .TRUE., Fermi_precision)
	coef_omF => coef2(1:n_cheb)

	n_calls = n_ran_samples * 3 * ((n_cheb-1) + (n_timestep-1) * (n_Bes-1))
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, s_hop, H_csr)

	PRINT *, "Calculating AC conductivity correlation function."

	! Average over (n_sample) samples
	DO i_sample=1, n_ran_samples

		PRINT *, "Sample ", i_sample, " of ", n_ran_samples
		WRITE(27,*) "Sample =", i_sample

		! make random state and psi1, psi2
		CALL random_state(wf0, n_wf, seed*i_sample)
		CALL csr_mv(wf0, n_wf, 1D0, cur_csr_x, psi1_x)
		CALL csr_mv(wf0, n_wf, 1D0, cur_csr_y, psi1_y)
		CALL Fermi(psi1_x, n_wf, coef_omF, H_csr, psi1_x)
		CALL Fermi(psi1_y, n_wf, coef_omF, H_csr, psi1_y)
		CALL Fermi(wf0, n_wf, coef_F, H_csr, psi2)

		!get correlation functions in all directions
		CALL csr_mv(psi1_x, n_wf, 1D0, cur_csr_x, wf1)
		corrval(1) = inner_prod(psi2, wf1)
		CALL csr_mv(psi1_x, n_wf, 1D0, cur_csr_y, wf1)
		corrval(2) = inner_prod(psi2, wf1)
		CALL csr_mv(psi1_y, n_wf, 1D0, cur_csr_x, wf1)
		corrval(3) = inner_prod(psi2, wf1)
		CALL csr_mv(psi1_y, n_wf, 1D0, cur_csr_y, wf1)
		corrval(4) = inner_prod(psi2, wf1)

		! write to file
		WRITE(27,"(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3,&
				   ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)") &
			1, &
			REAL(corrval(1)), AIMAG(corrval(1)), &
			REAL(corrval(2)), AIMAG(corrval(2)), &
			REAL(corrval(3)), AIMAG(corrval(3)), &
			REAL(corrval(4)), AIMAG(corrval(4))

		! iterate over time
		DO k = 2, n_timestep
			IF (MODULO(k, 64) == 0) THEN
				PRINT*, "    Timestep ", k, " of ", n_timestep
			END IF

			! calculate time evolution
			CALL cheb_wf_timestep(psi1_x, n_wf, Bes, 1D0, H_csr, psi1_x)
			CALL cheb_wf_timestep(psi1_y, n_wf, Bes, 1D0, H_csr, psi1_y)
			CALL cheb_wf_timestep(psi2, n_wf, Bes, 1D0, H_csr, psi2)

			!get correlation functions in all directions
			CALL csr_mv(psi1_x, n_wf, 1D0, cur_csr_x, wf1)
			corrval(1) = inner_prod(psi2, wf1)
			CALL csr_mv(psi1_x, n_wf, 1D0, cur_csr_y, wf1)
			corrval(2) = inner_prod(psi2, wf1)
			CALL csr_mv(psi1_y, n_wf, 1D0, cur_csr_x, wf1)
			corrval(3) = inner_prod(psi2, wf1)
			CALL csr_mv(psi1_y, n_wf, 1D0, cur_csr_y, wf1)
			corrval(4) = inner_prod(psi2, wf1)

			! write to file
			WRITE(27,"(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3,&
					   ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)") &
				k, &
				REAL(corrval(1)), AIMAG(corrval(1)), &
				REAL(corrval(2)), AIMAG(corrval(2)), &
				REAL(corrval(3)), AIMAG(corrval(3)), &
				REAL(corrval(4)), AIMAG(corrval(4))

			! update output array
			corr(1,k) = corr(1,k) + corrval(1) / n_ran_samples
			corr(2,k) = corr(2,k) + corrval(2) / n_ran_samples
			corr(3,k) = corr(3,k) + corrval(3) / n_ran_samples
			corr(4,k) = corr(4,k) + corrval(4) / n_ran_samples
		END DO
	END DO

	CLOSE(27)

END SUBROUTINE tbpm_accond


! Get dynamical polarization
SUBROUTINE tbpm_dyn_pol(Bes, n_Bes, beta, mu, s_indptr, n_indptr, &
						s_indices, n_indices, s_hop, n_hop, H_rescale, &
						s_dx, n_dx, s_dy, n_dy, s_site_x, n_site_x, &
						s_site_y, n_site_y, s_site_z, n_site_z, seed, &
						n_timestep, n_ran_samples, nr_Fermi, Fermi_precision, &
						q_points, n_q_points, output_filename, corr)

	USE math, ONLY: inner_prod
	USE random
	USE csr
	USE propagation
	USE funcs
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_Bes, n_indptr, n_indices, n_hop, n_dx, n_dy
	INTEGER, INTENT(IN) :: n_timestep, n_ran_samples, seed, n_q_points
	INTEGER, INTENT(IN) :: n_site_x, n_site_y, n_site_z, nr_Fermi
	REAL(KIND=8), INTENT(IN) :: Fermi_precision, H_rescale, beta, mu
	REAL(KIND=8), INTENT(IN), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
	REAL(KIND=8), INTENT(IN), DIMENSION(n_dx) :: s_dx
	REAL(KIND=8), INTENT(IN), DIMENSION(n_dy) :: s_dy
	REAL(KIND=8), INTENT(IN), DIMENSION(n_site_x) :: s_site_x
	REAL(KIND=8), INTENT(IN), DIMENSION(n_site_y) :: s_site_y
	REAL(KIND=8), INTENT(IN), DIMENSION(n_site_z) :: s_site_z
	REAL(KIND=8), INTENT(IN), DIMENSION(n_q_points, 3) :: q_points
	CHARACTER*(*), INTENT(IN) :: output_filename
	! output
	REAL(KIND=8), INTENT(OUT), DIMENSION(n_q_points, n_timestep) :: corr

	! declare vars
	INTEGER :: i_sample, k, n_cheb1, n_cheb2, i_q, n_wf, n_calls
	REAL(KIND=8) :: corrval
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf1, psi1, psi2
	REAL(KIND=8), DIMENSION(nr_Fermi), TARGET :: coef1, coef2
	! cheb coefs for Fermi operator and one minus Fermi operator
	REAL(KIND=8), DIMENSION(:), POINTER :: coef_F, coef_omF
	! coefs for density
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: s_density_q, s_density_min_q
	TYPE(SPARSE_MATRIX_T) :: H_csr

	! set some values
	n_wf = n_indptr - 1
	corr = 0D0

	! get Fermi cheb coefficients
	CALL get_Fermi_cheb_coef(coef1, n_cheb1, nr_Fermi, &
							 beta, mu, .FALSE., Fermi_precision)
	coef_F => coef1(1:n_cheb1)

	! get one minus Fermi cheb coefficients
	CALL get_Fermi_cheb_coef(coef2, n_cheb2, nr_Fermi, &
							 beta, mu, .TRUE., Fermi_precision)
	coef_omF => coef2(1:n_cheb2)

	n_calls = n_ran_samples * (n_cheb1+n_cheb2-1)+2*((n_timestep-1)*(n_Bes-1))
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, s_hop, H_csr)

	OPEN(unit=27,file=output_filename)
	WRITE(27,*) "Number of qpoints =", n_q_points
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	PRINT *, "Calculating dynamical polarization correlation function."

	!loop over n qpoints
	DO i_q = 1, n_q_points
		PRINT *, "q-point ", i_q, " of ", n_q_points
		WRITE(27,"(A9, ES24.14E3,ES24.14E3,ES24.14E3)") "qpoint= ", &
			q_points(i_q,1), q_points(i_q,2), q_points(i_q,3)

		!calculate the coefficients for the density operator
		!exp(i * q dot r)
		CALL density_coef(n_wf, s_site_x, s_site_y, s_site_z, &
						  q_points(i_q,:), s_density_q, s_density_min_q)

		! Average over (n_ran_samples) samples
		DO i_sample=1, n_ran_samples
			PRINT *, " Sample ", i_sample, " of ", n_ran_samples
			WRITE(27,*) "Sample =", i_sample

			! make random state and psi1, psi2
			CALL random_state(wf0, n_wf, seed*i_sample)
			! call density(-q)*wf0, resulting in psi1
			CALL density(wf0, n_wf, s_density_min_q, psi1)
			! call fermi with 1-fermi coefficients for psi1
			CALL Fermi(psi1, n_wf, coef_omF, H_csr, psi1)
			! call fermi with fermi coefficients for psi2
			CALL Fermi(wf0, n_wf, coef_F, H_csr, psi2)
			! call density(q)*psi1, resulting in wf1
			CALL density(psi1, n_wf, s_density_q, wf1)

			! get correlation and store
			corrval = AIMAG(inner_prod(psi2, wf1))
			WRITE(27,*) 1, corrval
			corr(i_q, 1) = corr(i_q, 1) + corrval / n_ran_samples

			! iterate over tau
			DO k = 2, n_timestep
			IF (MODULO(k, 64) == 0) THEN
				PRINT *, "    Timestep ", k, " of ", n_timestep
			END IF

			! call time and density operators
			CALL cheb_wf_timestep(psi1, n_wf, Bes, 1D0, H_csr, psi1)
			CALL cheb_wf_timestep(psi2, n_wf, Bes, 1D0, H_csr, psi2)
			CALL density(psi1, n_wf, s_density_q, wf1)

			! get correlation and store
			corrval = AIMAG(inner_prod(psi2, wf1))
			WRITE(27,*) k, corrval
			corr(i_q, k) = corr(i_q, k) + corrval / n_ran_samples

			END DO
		END DO
	END DO

	CLOSE(27)

END SUBROUTINE tbpm_dyn_pol


! Get DC conductivity
SUBROUTINE tbpm_dccond(Bes, n_Bes, beta, mu, s_indptr, n_indptr, &
					   s_indices, n_indices, s_hop, n_hop, H_rescale, &
					   s_dx, n_dx, s_dy, n_dy, seed, &
					   n_timestep, n_ran_samples, t_step, &
					   energies, n_energies, en_inds, n_en_inds, &
					   output_filename_dos, output_filename_dc, &
					   dos_corr, dc_corr)

	USE const
	USE math, ONLY: inner_prod
	USE random
	USE csr
	USE propagation, ONLY: cheb_wf_timestep
	USE funcs
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_Bes, n_indptr, n_indices, n_hop, n_dx, n_dy
	INTEGER, INTENT(IN) :: n_timestep, n_ran_samples, seed, n_energies
	INTEGER, INTENT(IN) :: n_en_inds
	REAL(KIND=8), INTENT(IN) :: H_rescale, beta, mu, t_step
	REAL(KIND=8), INTENT(IN), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
	REAL(KIND=8), INTENT(IN), DIMENSION(n_dx) :: s_dx
	REAL(KIND=8), INTENT(IN), DIMENSION(n_dy) :: s_dy
	REAL(KIND=8), INTENT(IN), DIMENSION(n_energies) :: energies
	INTEGER, INTENT(IN), DIMENSION(n_en_inds) :: en_inds
	CHARACTER*(*), INTENT(IN) :: output_filename_dos
	CHARACTER*(*), INTENT(IN) :: output_filename_dc
	! output
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_timestep) :: dos_corr
	COMPLEX(KIND=8), INTENT(OUT), DIMENSION(2,n_energies,n_timestep) :: dc_corr
	! elements dc_corr_x and dc_corr_y

	! declare vars
	INTEGER :: i_sample, i, j, k, l, t, n_wf, n_calls
	REAL(KIND=8) :: W, QE_sum, en
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf_t_pos, wf_t_neg, wfE
	COMPLEX(KIND=8), DIMENSION(n_en_inds, n_indptr - 1) :: wf_QE
	COMPLEX(KIND=8), DIMENSION(2, n_indptr - 1) :: wf0_J, wfE_J, wfE_J_t
	! cheb coefs for Fermi operator and one minus Fermi operator
	REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: coef_F, coef_omF
	! coefs for x or y current
	COMPLEX(KIND=8), DIMENSION(n_hop) :: sys_current_x, sys_current_y
	COMPLEX(KIND=8) :: dos_corrval
	COMPLEX(KIND=8), DIMENSION(2) :: dc_corrval
	TYPE(SPARSE_MATRIX_T) :: H_csr, cur_csr_x, cur_csr_y

	! set some values
	n_wf = n_indptr - 1
	dos_corr = 0D0
	dc_corr = 0D0
	n_calls = n_ran_samples*2*(n_timestep+n_en_inds*(n_timestep-1))*(n_Bes-1)
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, s_hop, H_csr)

	! prepare output files
	OPEN(unit=27,file=output_filename_dos)
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	OPEN(unit=28,file=output_filename_dc)
	WRITE(28,*) "Number of samples =", n_ran_samples
	WRITE(28,*) "Number of energies =", n_en_inds
	WRITE(28,*) "Number of timesteps =", n_timestep

	! get current coefficients
	n_calls = n_ran_samples * 4 * n_en_inds
	! CALL current_coefficient(s_hop, s_dx, n_hop, H_rescale, sys_current_x)
	! CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, &
	! 					 sys_current_x, cur_csr_x)
	CALL current_coefficient(s_hop, s_dy, n_hop, H_rescale, sys_current_y)
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, &
						 sys_current_y, cur_csr_y)

	PRINT *, "Calculating DC conductivity correlation function."

	! Average over (n_ran_samples) samples
	DO i_sample=1, n_ran_samples
		PRINT *, "Calculating for sample ", i_sample, " of ", n_ran_samples
		WRITE(27,*) "Sample =", i_sample
		WRITE(28,*) "Sample =", i_sample

		! make random state
		CALL random_state(wf0, n_wf, seed*i_sample)

		! ------------
		! first, get DOS and quasi-eigenstates
		! ------------

		! initial values for wf_t and wf_QE
		DO i = 1, n_wf
			wf_t_pos(i) = wf0(i)
			wf_t_neg(i) = wf0(i)
		END DO
		DO i = 1, n_en_inds
			DO j = 1, n_wf
				wf_QE(i, j) = wf0(j)
			END DO
		END DO

		! Iterate over time, get Fourier transform
		DO k = 1, n_timestep
			IF (MODULO(k,64) == 0) THEN
				PRINT *, "Getting DOS/QE for timestep ", k, " of ", n_timestep
			END IF

			! time evolution
			CALL cheb_wf_timestep(wf_t_pos, n_wf, Bes, 1D0, H_csr, wf_t_pos)
			CALL cheb_wf_timestep(wf_t_neg, n_wf, Bes, -1D0, H_csr, wf_t_neg)

			! get dos correlation
			dos_corrval = inner_prod(wf0, wf_t_pos)
			dos_corr(k) = dos_corr(k) + dos_corrval/n_ran_samples
			WRITE(27,*) k, REAL(dos_corrval), AIMAG(dos_corrval)

			W = 0.5 * (1 + COS(pi * k / n_timestep)) ! Hanning window

			!$OMP PARALLEL DO SIMD PRIVATE(j)
			DO i = 1, n_en_inds

				en = energies(en_inds(i) + 1)

				DO j = 1, n_wf
					wf_QE(i,j) = wf_QE(i,j) + &
								 EXP(img*en*k*t_step)*wf_t_pos(j)*W
					wf_QE(i,j) = wf_QE(i,j) + &
								 EXP(-img*en*k*t_step)*wf_t_neg(j)*W
				END DO
			END DO
			!$OMP END PARALLEL DO SIMD
		END DO

		! Normalise
		DO i = 1, n_en_inds
			QE_sum = 0
			DO j = 1, n_wf
				QE_sum = QE_sum + ABS(wf_QE(i, j))**2
			END DO
			DO j = 1, n_wf
				wf_QE(i, j) = wf_QE(i, j)/DSQRT(QE_sum)
			END DO
		END DO

		! ------------
		! then, get dc conductivity
		! ------------

		! iterate over energies
		DO i = 1, n_en_inds
			! some output
			IF (MODULO(i,8) == 0) THEN
				PRINT *, "Getting DC conductivity for energy: ", &
						i, " of ", n_en_inds
			END IF
			WRITE(28,*) "Energy ", i, en_inds(i), energies(en_inds(i) + 1)

			! get corresponding quasi-eigenstate
			wfE(:) = wf_QE(i,:)/ABS(inner_prod(wf0, wf_QE(i,:)))

			! make psi1, psi2
			CALL csr_mv(wf0, n_wf, 1D0, cur_csr_y, wf0_J(1,:))
			CALL csr_mv(wf0, n_wf, 1D0, cur_csr_y, wf0_J(2,:))
			CALL csr_mv(wfE, n_wf, 1D0, cur_csr_y, wfE_J(1,:))
			CALL csr_mv(wfE, n_wf, 1D0, cur_csr_y, wfE_J(2,:))

			! get correlation values
			dc_corrval(1) = inner_prod(wf0_J(1,:), wfE_J(1,:))
			dc_corrval(2) = inner_prod(wf0_J(2,:), wfE_J(2,:))

			! write to file
			WRITE(28,"(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)")&
				1, &
				REAL(dc_corrval(1)), AIMAG(dc_corrval(1)), &
				REAL(dc_corrval(2)), AIMAG(dc_corrval(2))

			! update correlation functions
			dc_corr(1,i,1)=dc_corr(1,i,1)+dc_corrval(1)/n_ran_samples
			dc_corr(2,i,1)=dc_corr(2,i,1)+dc_corrval(2)/n_ran_samples

			! iterate over time
			DO k = 2, n_timestep
			! NEGATIVE time evolution of QE state
			CALL cheb_wf_timestep(wfE_J(1,:), n_wf, Bes, &
								  -1D0, H_csr, wfE_J(1,:))
			CALL cheb_wf_timestep(wfE_J(2,:), n_wf, Bes, &
								  -1D0, H_csr, wfE_J(2,:))

			! get correlation values
			dc_corrval(1) = inner_prod(wf0_J(1,:), wfE_J(1,:))
			dc_corrval(2) = inner_prod(wf0_J(2,:), wfE_J(2,:))

			! write to file
			WRITE(28,"(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)")&
				k, &
				REAL(dc_corrval(1)), AIMAG(dc_corrval(1)), &
				REAL(dc_corrval(2)), AIMAG(dc_corrval(2))

			! update correlation functions
			dc_corr(1,i,k)=dc_corr(1,i,k)+dc_corrval(1)/n_ran_samples
			dc_corr(2,i,k)=dc_corr(2,i,k)+dc_corrval(2)/n_ran_samples
			END DO
		END DO
	END DO

	CLOSE(27)
	CLOSE(28)

END SUBROUTINE tbpm_dccond


! Get quasi-eigenstates
SUBROUTINE tbpm_eigenstates(Bes, n_Bes, s_indptr, n_indptr, &
							s_indices, n_indices, s_hop, n_hop, &
							seed, n_timestep, n_ran_samples, t_step, &
							energies, n_energies, wf_QE)

	USE const
	USE math, ONLY: inner_prod
	USE random
	USE csr
	USE propagation, ONLY: cheb_wf_timestep
	IMPLICIT NONE
	! input
	INTEGER, INTENT(IN) :: n_Bes, n_indptr, n_indices, n_hop
	INTEGER, INTENT(IN) :: n_timestep, seed, n_energies, n_ran_samples
	REAL(KIND=8), INTENT(IN), DIMENSION(n_Bes) :: Bes
	REAL(KIND=8), INTENT(IN) :: t_step
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
	REAL(KIND=8), INTENT(IN), DIMENSION(n_energies) :: energies
	! output
	REAL(KIND=8), INTENT(OUT), DIMENSION(n_energies, n_indptr - 1) :: wf_QE

	! declare vars
	INTEGER :: i, j, k, l, t, n_wf, i_sample, n_calls
	REAL(KIND=8) :: W, QE_sum
	COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf_t_pos, wf_t_neg
	COMPLEX(KIND=8), DIMENSION(n_energies, n_indptr - 1) :: wfq
	TYPE(SPARSE_MATRIX_T) :: H_csr

	n_wf = n_indptr - 1
	wf_QE = 0D0
	n_calls = n_ran_samples * 2 * n_timestep * (n_Bes - 1)
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, s_hop, H_csr)

	PRINT *, "Calculating quasi-eigenstates."

	! Average over (n_ran_samples) samples
	DO i_sample=1, n_ran_samples

		PRINT *, "  Calculating for sample ", i_sample, " of ", n_ran_samples
		! make random state
        CALL random_state(wf0, n_wf, seed*i_sample)

        ! initial values for wf_t and wf_QE
        DO i = 1, n_wf
            wf_t_pos(i) = wf0(i)
            wf_t_neg(i) = wf0(i)
        END DO
		DO i = 1, n_energies
			DO j = 1, n_wf
				wfq(i, j) = wf0(j)
			END DO
		END DO

		! Iterate over time, get Fourier transform
        DO k = 1, n_timestep

            IF (MODULO(k,64) == 0) THEN
                PRINT *, "    Timestep ", k, " of ", n_timestep
            END IF

            CALL cheb_wf_timestep(wf_t_pos, n_wf, Bes, 1D0, H_csr, wf_t_pos)
            CALL cheb_wf_timestep(wf_t_neg, n_wf, Bes, -1D0, H_csr, wf_t_neg)

            W = 0.5 * (1 + COS(pi * k / n_timestep)) ! Hanning window

            !$OMP PARALLEL DO SIMD PRIVATE (j)
            DO i = 1, n_energies
                DO j = 1, n_wf
                    wfq(i,j) = wfq(i,j)+&
                        	   EXP(img*energies(i)*k*t_step)*wf_t_pos(j)*W
                    wfq(i,j) = wfq(i,j)+&
                        	   EXP(-img*energies(i)*k*t_step)*wf_t_neg(j)*W
                END DO
            END DO
            !$OMP END PARALLEL DO SIMD

		END DO

		! Normalise
		DO i = 1, n_energies
			QE_sum = 0
			DO j = 1, n_wf
				QE_sum = QE_sum + ABS(wfq(i, j))**2
			END DO
			DO j = 1, n_wf
				wfq(i, j) = wfq(i, j)/DSQRT(QE_sum)
			END DO
		END DO

		DO i = 1, n_energies
			DO j = 1, n_wf
				wf_QE(i, j) = wf_QE(i, j) + ABS(wfq(i, j))**2 / n_ran_samples
			END DO
		END DO

	END DO

END SUBROUTINE tbpm_eigenstates


! USE the Kubo-Bastin formula to calculate the Hall conductivity
! based on PRL 114, 116602 (2015)
! This is a version with only XX (iTypeDC==1) or XY (iTypeDC==2)
SUBROUTINE tbpm_kbdc(seed, s_indptr, n_indptr, s_indices, n_indices, &
					 s_hop, n_hop, H_rescale, s_dx, n_dx, s_dy, n_dy, &
					 n_ran_samples, energies, n_energies, beta, prefactor, &
					 n_kernel, iTypeDC, NE_Integral, fermi_precision, &
					 corr_mu_avg)

	USE math, ONLY: inner_prod
	USE random
	USE csr
	USE propagation, ONLY: cheb_wf_timestep
	USE funcs
	USE kpm
    IMPLICIT NONE

    ! deal with input
    INTEGER, INTENT(in) :: iTypeDC, n_energies
    INTEGER, INTENT(in) :: seed, n_ran_samples
    INTEGER, INTENT(in) :: n_kernel, NE_Integral
    INTEGER, INTENT(in) :: n_indptr, n_indices, n_hop, n_dx, n_dy
    REAL(8), INTENT(in) :: prefactor, beta, fermi_precision, H_rescale
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop
	REAL(8), INTENT(in), DIMENSION(n_dx) :: s_dx
	REAL(8), INTENT(in), DIMENSION(n_dy) :: s_dy
    REAL(8), INTENT(in), DIMENSION(n_energies) :: energies
    COMPLEX(8), DIMENSION(n_hop) :: sys_current_x
    COMPLEX(8), DIMENSION(n_hop) :: sys_current_y

    !declare vars
    INTEGER :: i, j, k, i_sample, mu_min, mu_max, NE, n_wf, n_calls

    REAL(8):: WaveFunctionNorm
    COMPLEX(8), DIMENSION(n_indptr - 1) :: wf0,wf1,wf1X,wf1X0,wf1X1,wf_in
    COMPLEX(8):: wf_DimKern(1:(n_indptr - 1),0:n_kernel)
    COMPLEX(8),DIMENSION(n_kernel,n_kernel)::corr_mu

    REAL(8):: a,r0,x,y,energy,mu2
    COMPLEX(8) :: ca,cb,COMPLEXa,COMPLEXb,dcx
    REAL(8),DIMENSION(n_kernel):: KernelFunction,ChebPol
    COMPLEX(8),DIMENSION(n_kernel):: sum_temp
    COMPLEX(8),DIMENSION(n_kernel,n_kernel)::Gamma_mn
    !COMPLEX(8),ALLOCATABLE:: en_integral(:),sum_gamma_mu(:)
	TYPE(SPARSE_MATRIX_T) :: H_csr, cur_csr_x, cur_csr_y

    ! output
    !REAL(8),intent(out),DIMENSION(n_energies) :: cond
    COMPLEX(8),intent(out),DIMENSION(0:n_kernel,0:n_kernel)::corr_mu_avg

    n_wf = n_indptr - 1
	! write(*,*) 'go into subroutine current_coefficient now'
	n_calls = n_ran_samples * (n_kernel + 1)
	CALL current_coefficient(s_hop, s_dx, n_hop, H_rescale, sys_current_x)
	CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, &
						 sys_current_x, cur_csr_x)
	if (iTypeDC==2) then
		n_calls = n_ran_samples * n_kernel
		CALL current_coefficient(s_hop, s_dy, n_hop, H_rescale, sys_current_y)
		CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, &
							 sys_current_y, cur_csr_y)
	end if

	n_calls = n_ran_samples * (2 * n_kernel - 1)
    CALL make_csr_matrix(n_wf, n_calls, s_indptr, s_indices, s_hop, H_csr)


    corr_mu=0.
    corr_mu_avg=0.

    ! iterate over random states
    do i_sample=1, n_ran_samples
        ! get random state
        call random_state(wf_in, n_wf, seed*i_sample)

        do j=1, n_kernel
            if (mod(j,250)==0) then
                PRINT *, "Currently at j = ", j
            end if

            if (j==1) then
                wf_DimKern(:,j)=wf_in
            else
                call csr_mv(wf_DimKern(:,j-1), n_wf, 1D0, H_csr, &
                    		wf_DimKern(:,j))
            end if

            !calculate the chebyshev polynomial and replace wf_ 0, 1
            if (j>2) then
                call get_ChebPol_n_wfthOrder( &
                    n_wf, wf_DimKern(:,j-2), wf_DimKern(:,j-1), &
                    wf_DimKern(:,j))
            end if

            ! if (j==n_kernel .or. j==n_kernel/2 .or. &
            !     j==n_kernel/4 .or. j==n_kernel*3/4 .or. j==1) then
            !     WaveFunctionNorm=abs(inner_prod(wf_DimKern(:,j),&
            !         wf_DimKern(:,j)))
            !     if (WaveFunctionNorm>H_rescale .or. WaveFunctionNorm<0.) then
            !         PRINT *, "WaveFunctionNorm=", WaveFunctionNorm
            !         PRINT *, "Error: hoprescale too small..."
            !         stop
            !     end if
            ! end if

        end do

        if (iTypeDC==1) then
            do j=1, n_kernel
                call csr_mv(wf_DimKern(:,j), n_wf, 1D0, cur_csr_x, &
							wf_DimKern(:,j))
            end do
        else if (iTypeDC==2) then
            do j=1, n_kernel
                call csr_mv(wf_DimKern(:,j), n_wf, 1D0, cur_csr_y, &
							wf_DimKern(:,j))
            end do
        end if

        ! calculate correlation matrix
        do i=1, n_kernel
            if (mod(i,256)==0) then
                PRINT *, "Currently at i = ", i
            end if
            if (i==1) then
                call csr_mv(wf_in, n_wf, 1D0, cur_csr_x, wf1X)
                wf1X0=wf1X
            else if (i==2) then
                call csr_mv(wf1X0, n_wf, 1D0, H_csr, wf1X)
                wf1X1=wf1X
            else
                call csr_mv(wf1X1, n_wf, 1D0, H_csr, wf1X)
                ! calculate the chebyshev polynomial and replace wf_ 0, 1
                call get_ChebPol_wf(n_wf,wf1X0,wf1X1,wf1X)

            end if

        ! calculate the matrix elements of the correction function

            !$OMP PARALLEL DO
            do j=1, n_kernel
                corr_mu(i,j)=inner_prod(wf1X, wf_DimKern(:,j))
            end do
            !$OMP END PARALLEL DO

        end do


        !$OMP PARALLEL DO SIMD
        do j=1, n_kernel
            do i=1, n_kernel
                corr_mu_avg(i,j)=corr_mu_avg(i,j)+corr_mu(i,j)
            end do
        end do
        !$OMP END PARALLEL DO SIMD

    end do

    if (n_ran_samples>1) then
        !$OMP PARALLEL DO SIMD
        do j=1, n_kernel
            do i=1, n_kernel
                corr_mu_avg(i,j)=corr_mu_avg(i,j)/n_ran_samples
            end do
        end do
        !$OMP END PARALLEL DO SIMD
    end if


    !call cond_from_trace(corr_mu_avg, n_kernel, n_kernel, energies, n_energies, &
    !        NE_integral, H_rescale, beta, fermi_precision, prefactor, &
    !        cond)


END SUBROUTINE tbpm_kbdc
