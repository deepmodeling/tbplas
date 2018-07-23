! ------------------------------------------
! TBPM fortran subroutines, callable from python
! ------------------------------------------

! Get DOS
SUBROUTINE tbpm_dos(Bes, n_Bes, s_indptr, n_indptr, s_indices, n_indices, &
	 				s_hop, n_hop, seed, n_timestep, n_ran_samples, &
					output_filename, corr)

	! prepare
	USE tbpm_mod
	IMPLICIT NONE

	! deal with input
	INTEGER, INTENT(in) :: n_Bes, n_indptr, n_indices, n_hop
	INTEGER, INTENT(in) :: n_timestep, n_ran_samples, seed
	REAL(8), INTENT(in), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop
	CHARACTER*(*), INTENT(in) :: output_filename

	! declare vars
	INTEGER :: i_sample,k, i, n_wf
	COMPLEX(8), DIMENSION(n_indptr - 1) :: wf0, wf_t
	COMPLEX(8) :: corrval

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_timestep) :: corr

	! set some values
	corr = 0.0d0
	n_wf = n_indptr - 1

	OPEN(unit=27,file=output_filename)
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	PRINT*, "Calculating DOS correlation function."

	! Average over (n_ran_samples) samples
	DO i_sample = 1, n_ran_samples

		PRINT*, "Sample ", i_sample, " of ", n_ran_samples
		WRITE(27,*) "Sample =", i_sample

		! make random state
		CALL random_state(wf0, n_wf, seed*i_sample)
		CALL cheb_wf_timestep(wf0, n_wf, Bes, n_Bes, &
							  s_indptr, n_indptr, s_indices, n_indices, &
							  s_hop, n_hop, wf_t)
		corrval = inner_prod(wf0, wf_t, n_wf)

		WRITE(27,*) 1, REAL(corrval), AIMAG(corrval)
		corr(1) = corr(1) + corrval / n_ran_samples

		! iterate over time, get correlation function
		DO k = 2, n_timestep
			IF (MODULO(k,64) == 0) THEN
				PRINT*, "Timestep ", k, " of ", n_timestep
			END IF

			CALL cheb_wf_timestep(wf_t, n_wf, Bes, n_Bes, &
								  s_indptr, n_indptr, s_indices, n_indices, &
								  s_hop, n_hop, wf_t)
			corrval = inner_prod(wf0, wf_t, n_wf)

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

	! prepare
	USE tbpm_mod
	IMPLICIT NONE

	! deal with input
	INTEGER, INTENT(in) :: n_Bes, n_indptr, n_indices, n_hop
	INTEGER, INTENT(in) :: n_timestep, seed, n_siteind, n_wfw
	INTEGER, INTENT(in) :: n_ran_samples
	INTEGER, INTENT(in), DIMENSION(n_siteind) :: site_indices
	REAL(8), INTENT(in), DIMENSION(n_wfw) :: wf_weights
	REAL(8), INTENT(in), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop
	CHARACTER*(*), INTENT(in) :: output_filename

	! declare vars
	INTEGER :: k, i, n_wf, i_sample
	COMPLEX(8), DIMENSION(n_siteind) :: wf_temp
	COMPLEX(8), DIMENSION(n_indptr - 1) :: wf0, wf_t
	COMPLEX(8) :: corrval

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_timestep) :: corr

	! set some values
	corr = 0.0d0
	n_wf = n_indptr - 1

	OPEN(unit=27,file=output_filename)
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	PRINT*, "Calculating LDOS correlation function."

	DO i_sample=1, n_ran_samples

		PRINT*, "Sample ", i_sample, " of ", n_ran_samples
		WRITE(27,*) "Sample =", i_sample

		! make LDOS state
		CALL random_state(wf_temp, n_siteind, seed*i_sample)
		wf0 = 0.0d0
		DO i = 1, n_siteind
			wf0(site_indices(i) + 1) = wf_temp(i) * wf_weights(i)
		END DO
		DO i = 1, n_wf
			wf_t(i) = wf0(i)
		END DO

		! iterate over time, get correlation function
		DO k = 1, n_timestep
			IF (MODULO(k,64) == 0) THEN
				PRINT*, "Timestep ", k, " of ", n_timestep
			END IF

			CALL cheb_wf_timestep(wf_t, n_wf, Bes, n_Bes, &
								  s_indptr, n_indptr, s_indices, n_indices, &
								  s_hop, n_hop, wf_t)
			corrval = inner_prod(wf0, wf_t, n_wf)

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

	! prepare
	USE tbpm_mod
	IMPLICIT NONE

	! deal with input
	INTEGER, INTENT(in) :: n_Bes, n_indptr, n_indices, n_hop, n_dx, n_dy
	INTEGER, INTENT(in) :: n_timestep, n_ran_samples, seed
	INTEGER, INTENT(in) :: nr_Fermi
	REAL(8), INTENT(in) :: Fermi_precision, H_rescale, beta, mu
	REAL(8), INTENT(in), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop
	REAL(8), INTENT(in), DIMENSION(n_dx) :: s_dx
	REAL(8), INTENT(in), DIMENSION(n_dy) :: s_dy
	CHARACTER*(*), INTENT(in) :: output_filename

	! declare vars
	INTEGER :: i_sample, k, i, j, n_cheb, n_wf
	COMPLEX(8), DIMENSION(n_indptr - 1) :: wf0,wf1,psi1_x,psi1_y,psi2
	REAL(8), ALLOCATABLE :: coef(:)
	COMPLEX(8), DIMENSION(4) :: corrval
	REAL(8), ALLOCATABLE :: coef_F(:) ! cheb coefs for Fermi operator
	REAL(8), ALLOCATABLE :: coef_omF(:) ! cheb coefs one minus Fermi operator
	COMPLEX(8), DIMENSION(n_hop) :: sys_current_x ! coefs for x current
	COMPLEX(8), DIMENSION(n_hop) :: sys_current_y ! coefs for y current

	! output
	COMPLEX(8), INTENT(out), DIMENSION(4, n_timestep) :: corr
	! corr has 4 elements, respectively: corr_xx, corr_xy, corr_yx, corr_yy

	! set some values
	corr = 0.0d0
	n_wf = n_indptr - 1

	! prepare output file
	OPEN(unit=27,file=output_filename)
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	! get current coefficients
	CALL current_coefficient(s_hop, s_dx, n_hop, sys_current_x)
	CALL current_coefficient(s_hop, s_dy, n_hop, sys_current_y)
	sys_current_x = H_rescale * sys_current_x
	sys_current_y = H_rescale * sys_current_y

	! get Fermi cheb coefficients
	ALLOCATE(coef(nr_Fermi))
	coef = 0.0d0
	CALL get_Fermi_cheb_coef(coef, n_cheb, nr_Fermi,&
							 beta, mu, .FALSE., Fermi_precision)
	ALLOCATE(coef_F(n_cheb))
	DO i=1, n_cheb
		coef_F(i) = coef(i)
	END DO

	! get one minus Fermi cheb coefficients
	coef = 0.0d0
	CALL get_Fermi_cheb_coef(coef, n_cheb,&
			nr_Fermi, beta, mu, .TRUE., Fermi_precision)
	ALLOCATE(coef_omF(n_cheb))
	DO i=1, n_cheb
		coef_omF(i) = coef(i)
	END DO
	DEALLOCATE(coef)

	PRINT*, "Calculating AC conductivity correlation function."

	! Average over (n_sample) samples
	corr = 0.0d0
	DO i_sample=1, n_ran_samples

		PRINT*, "Sample ", i_sample, " of ", n_ran_samples
		WRITE(27,*) "Sample =", i_sample

		! make random state and psi1, psi2
		CALL random_state(wf0, n_wf, seed*i_sample)
		CALL current(wf0, n_wf, s_indptr, n_indptr, s_indices, &
			n_indices, sys_current_x, n_hop, psi1_x)
		CALL current(wf0, n_wf, s_indptr, n_indptr, s_indices, &
			n_indices, sys_current_y, n_hop, psi1_y)
		CALL Fermi(psi1_x, n_wf, coef_omF, n_cheb, &
			s_indptr, n_indptr, s_indices, n_indices, &
			s_hop, n_hop, psi1_x)
		CALL Fermi(psi1_y, n_wf, coef_omF, n_cheb, &
			s_indptr, n_indptr, s_indices, n_indices, &
			s_hop, n_hop, psi1_y)
		CALL Fermi(wf0, n_wf, coef_F, n_cheb, &
			s_indptr, n_indptr, s_indices, n_indices, &
			s_hop, n_hop, psi2)

		!get correlation functions in all directions
		CALL current(psi1_x, n_wf, s_indptr, n_indptr, s_indices, &
			n_indices, sys_current_x, n_hop, wf1)
		corrval(1) = inner_prod(psi2, wf1, n_wf)
		CALL current(psi1_x, n_wf, s_indptr, n_indptr, s_indices, &
			n_indices, sys_current_y, n_hop, wf1)
		corrval(2) = inner_prod(psi2, wf1, n_wf)
		CALL current(psi1_y, n_wf, s_indptr, n_indptr, s_indices, &
			n_indices, sys_current_x, n_hop, wf1)
		corrval(3) = inner_prod(psi2, wf1, n_wf)
		CALL current(psi1_y, n_wf, s_indptr, n_indptr, s_indices, &
			n_indices, sys_current_y, n_hop, wf1)
		corrval(4) = inner_prod(psi2, wf1, n_wf)

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
			IF (MODULO(k,64) == 0) THEN
				PRINT*, "Timestep ", k, " of ", n_timestep
			END IF

			! calculate time evolution
			CALL cheb_wf_timestep(psi1_x, n_wf, Bes, n_Bes, &
					s_indptr, n_indptr, s_indices, n_indices, &
					s_hop, n_hop, psi1_x)
			CALL cheb_wf_timestep(psi1_y, n_wf, Bes, n_Bes, &
					s_indptr, n_indptr, s_indices, n_indices, &
					s_hop, n_hop, psi1_y)
			CALL cheb_wf_timestep(psi2, n_wf, Bes, n_Bes, &
					s_indptr, n_indptr, s_indices, n_indices, &
					s_hop, n_hop, psi2)

			!get correlation functions in all directions
			CALL current(psi1_x, n_wf, s_indptr, n_indptr, s_indices, &
					n_indices, sys_current_x, n_hop, wf1)
			corrval(1) = inner_prod(psi2, wf1, n_wf)
			CALL current(psi1_x, n_wf, s_indptr, n_indptr, s_indices, &
					n_indices, sys_current_y, n_hop, wf1)
			corrval(2) = inner_prod(psi2, wf1, n_wf)
			CALL current(psi1_y, n_wf, s_indptr, n_indptr, s_indices, &
					n_indices, sys_current_x, n_hop, wf1)
			corrval(3) = inner_prod(psi2, wf1, n_wf)
			CALL current(psi1_y, n_wf, s_indptr, n_indptr, s_indices, &
					n_indices, sys_current_y, n_hop, wf1)
			corrval(4) = inner_prod(psi2, wf1, n_wf)

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

	! prepare
	USE tbpm_mod
	IMPLICIT NONE

	! deal with input
	INTEGER, INTENT(in) :: n_Bes, n_indptr, n_indices, n_hop, n_dx, n_dy
	INTEGER, INTENT(in) :: n_timestep, n_ran_samples, seed, n_q_points
	INTEGER, INTENT(in) :: n_site_x, n_site_y, n_site_z, nr_Fermi
	REAL(8), INTENT(in) :: Fermi_precision, H_rescale, beta, mu
	REAL(8), INTENT(in), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop
	REAL(8), INTENT(in), DIMENSION(n_dx) :: s_dx
	REAL(8), INTENT(in), DIMENSION(n_dy) :: s_dy
	REAL(8), INTENT(in), DIMENSION(n_site_x) :: s_site_x
	REAL(8), INTENT(in), DIMENSION(n_site_y) :: s_site_y
	REAL(8), INTENT(in), DIMENSION(n_site_z) :: s_site_z
	REAL(8), INTENT(in), DIMENSION(n_q_points, 3) :: q_points
	CHARACTER*(*), INTENT(in) :: output_filename

	! declare vars
	INTEGER :: i_sample, k, i, j, n_cheb, i_q, n_wf
	REAL(8) :: omega, eps, W, tau, dpi, dpr, corrval
	COMPLEX(8), DIMENSION(n_indptr - 1) :: wf0, wf1, psi1, psi2
	REAL(8), ALLOCATABLE :: coef(:) ! temp variable
	REAL(8), ALLOCATABLE :: coef_F(:) ! cheb coefs for Fermi operator
	REAL(8), ALLOCATABLE :: coef_omF(:) ! cheb coefs one minus Fermi operator
	COMPLEX(8), DIMENSION(n_indptr - 1 ) :: s_density_q, s_density_min_q ! coefs for density

	REAL(8), INTENT(out), DIMENSION(n_q_points, n_timestep) :: corr
	n_wf = n_indptr - 1
	corr = 0.0d0

	! get Fermi cheb coefficients
	ALLOCATE(coef(nr_Fermi))
	coef = 0.0d0
	CALL get_Fermi_cheb_coef(coef,n_cheb,nr_Fermi,&
			beta,mu,.FALSE.,Fermi_precision)
	ALLOCATE(coef_F(n_cheb))
	DO i=1, n_cheb
		coef_F(i) = coef(i)
	END DO

	! get one minus Fermi cheb coefficients
	coef = 0.0d0
	CALL get_Fermi_cheb_coef(coef,n_cheb,&
		nr_Fermi,beta,mu,.TRUE.,Fermi_precision)
	ALLOCATE(coef_omF(n_cheb))
	DO i=1, n_cheb
		coef_omF(i) = coef(i)
	END DO
	DEALLOCATE(coef)

	OPEN(unit=27,file=output_filename)
	WRITE(27,*) "Number of qpoints =", n_q_points
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	PRINT*, "Calculating dynamical polarization correlation function."

	!loop over n qpoints
	DO i_q = 1, n_q_points
		PRINT*, "q-point ", i_q, " of ", n_q_points
		WRITE(27,"(A9, ES24.14E3,ES24.14E3,ES24.14E3)") "qpoint= ", &
			q_points(i_q,1), q_points(i_q,2), q_points(i_q,3)

		!calculate the coefficients for the density operator
		!exp(i* q dot r)
		CALL density_coef(n_wf, s_site_x, s_site_y, s_site_z, &
			q_points(i_q,:), s_density_q, s_density_min_q)

		! Average over (n_ran_samples) samples
		DO i_sample=1, n_ran_samples
			PRINT*, "Sample ", i_sample, " of ", n_ran_samples
			WRITE(27,*) "Sample =", i_sample

			! make random state and psi1, psi2
			CALL random_state(wf0, n_wf, seed*i_sample)
			! call density(-q)*wf0, resulting in psi1
			CALL density(wf0, n_wf, s_density_min_q, psi1)
			! call fermi with 1-fermi coefficients for psi1
			CALL Fermi(psi1, n_wf, coef_omF, n_cheb, &
				s_indptr, n_indptr, s_indices, n_indices, &
				s_hop, n_hop, psi1)
			! call fermi with fermi coefficients for psi2
			CALL Fermi(wf0, n_wf, coef_F, n_cheb, &
				s_indptr, n_indptr, s_indices, n_indices, &
				s_hop, n_hop, psi2)
			! call density(q)*psi1, resulting in wf1
			CALL density(psi1, n_wf, s_density_q, wf1)

			! get correlation and store
			corrval = AIMAG(inner_prod(psi2, wf1, n_wf))
			WRITE(27,*) 1, corrval
			corr(i_q, 1) = corr(i_q, 1) + corrval / n_ran_samples

			! iterate over tau
			DO k = 2, n_timestep
			IF (MODULO(k,64) == 0) THEN
				PRINT*, "Timestep ", k, " of ", n_timestep
			END IF

			! call time and density operators
			CALL cheb_wf_timestep(psi1, n_wf, Bes, n_Bes, &
					s_indptr, n_indptr, s_indices, n_indices, &
					s_hop, n_hop, psi1)
			CALL cheb_wf_timestep(psi2, n_wf, Bes, n_Bes, &
					s_indptr, n_indptr, s_indices, n_indices, &
					s_hop, n_hop, psi2)
			CALL density(psi1, n_wf, s_density_q, wf1)

			! get correlation and store
			corrval = AIMAG(inner_prod(psi2, wf1, n_wf))
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

	! prepare
	USE tbpm_mod
	IMPLICIT NONE

	! deal with input
	INTEGER, INTENT(in) :: n_Bes, n_indptr, n_indices, n_hop, n_dx, n_dy
	INTEGER, INTENT(in) :: n_timestep, n_ran_samples, seed, n_energies
	INTEGER, INTENT(in) :: n_en_inds
	REAL(8), INTENT(in) :: H_rescale, beta, mu, t_step
	REAL(8), INTENT(in), DIMENSION(n_Bes) :: Bes
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop
	REAL(8), INTENT(in), DIMENSION(n_dx) :: s_dx
	REAL(8), INTENT(in), DIMENSION(n_dy) :: s_dy
	REAL(8), INTENT(in), DIMENSION(n_energies) :: energies
	INTEGER, INTENT(in), DIMENSION(n_en_inds) :: en_inds
	CHARACTER*(*), INTENT(in) :: output_filename_dos
	CHARACTER*(*), INTENT(in) :: output_filename_dc

	! declare vars
	INTEGER :: i_sample,i, j, k, l, t, n_wf
	REAL(8) :: W, QE_sum, en
	COMPLEX(8), DIMENSION(n_indptr - 1) :: wf0,wf_t_pos,wf_t_neg,wfE
	COMPLEX(8), DIMENSION(n_en_inds, n_indptr - 1) :: wf_QE
	COMPLEX(8), DIMENSION(2,n_indptr - 1) :: wf0_J,wfE_J,wfE_J_t
	REAL(8), ALLOCATABLE :: coef_F(:) ! cheb coefs for Fermi operator
	REAL(8), ALLOCATABLE :: coef_omF(:) ! cheb coefs one minus Fermi operator
	COMPLEX(8), DIMENSION(n_hop) :: sys_current_x ! coefs for x current
	COMPLEX(8), DIMENSION(n_hop) :: sys_current_y ! coefs for y current
	COMPLEX(8) :: dos_corrval
	COMPLEX(8), DIMENSION(2) :: dc_corrval

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_timestep) :: dos_corr
	COMPLEX(8), INTENT(out), DIMENSION(2, n_energies, n_timestep) :: dc_corr
	! elements dc_corr_x and dc_corr_y

	! set some values
	n_wf = n_indptr - 1
	dos_corr = 0.0d0
	dc_corr = 0.0d0

	! prepare output files
	OPEN(unit=27,file=output_filename_dos)
	WRITE(27,*) "Number of samples =", n_ran_samples
	WRITE(27,*) "Number of timesteps =", n_timestep

	OPEN(unit=28,file=output_filename_dc)
	WRITE(28,*) "Number of samples =", n_ran_samples
	WRITE(28,*) "Number of energies =", n_en_inds
	WRITE(28,*) "Number of timesteps =", n_timestep

	! get current coefficients
	CALL current_coefficient(s_hop, s_dx, n_hop, sys_current_x)
	CALL current_coefficient(s_hop, s_dy, n_hop, sys_current_y)
	sys_current_x = H_rescale * sys_current_x
	sys_current_y = H_rescale * sys_current_y

	PRINT*, "Calculating DC conductivity correlation function."

	! Average over (n_ran_samples) samples
	DO i_sample=1, n_ran_samples
		PRINT*, "Calculating for sample ", i_sample, " of ", n_ran_samples
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
				PRINT*, "Getting DOS/QE for timestep ", k, " of ", n_timestep
			END IF

			! time evolution
			CALL cheb_wf_timestep(wf_t_pos, n_wf, Bes, n_Bes, &
								  s_indptr, n_indptr, s_indices, n_indices, &
								  s_hop, n_hop, wf_t_pos)
			CALL cheb_wf_timestep(wf_t_neg, n_wf, Bes, n_Bes, &
								  s_indptr, n_indptr, s_indices, n_indices, &
								  -s_hop, n_hop, wf_t_neg)

			! get dos correlation
			dos_corrval = inner_prod(wf0, wf_t_pos, n_wf)
			dos_corr(k) = dos_corr(k) + dos_corrval/n_ran_samples
			WRITE(27,*) k, REAL(dos_corrval), AIMAG(dos_corrval)

			W = 0.5*(1+COS(pi*k/n_timestep)) ! Hanning window

			!$OMP PARALLEL DO PRIVATE (j)
			DO i = 1, n_en_inds

				en = energies(en_inds(i))

				DO j = 1, n_wf
					wf_QE(i,j) = wf_QE(i,j) + &
								 EXP(img*en*k*t_step)*wf_t_pos(j)*W
					wf_QE(i,j) = wf_QE(i,j) + &
								 EXP(-img*en*k*t_step)*wf_t_neg(j)*W
				END DO
			END DO
			!$OMP END PARALLEL DO
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
				PRINT*, "Getting DC conductivity for energy: ", &
						i, " of ", n_en_inds
			END IF
			WRITE(28,*) "Energy ", i, en_inds(i), energies(en_inds(i))

			! get corresponding quasi-eigenstate
			wfE(:) = wf_QE(i,:)/ABS(inner_prod(wf0(:),wf_QE(i,:), n_wf))

			! make psi1, psi2
			CALL current(wf0, n_wf, s_indptr, n_indptr, s_indices, n_indices, &
						 sys_current_y, n_hop, wf0_J(1,:))
			CALL current(wf0, n_wf, s_indptr, n_indptr, s_indices, n_indices, &
						 sys_current_y, n_hop, wf0_J(2,:))
			CALL current(wfE, n_wf, s_indptr, n_indptr, s_indices, n_indices, &
						 sys_current_y, n_hop, wfE_J(1,:))
			CALL current(wfE, n_wf, s_indptr, n_indptr, s_indices, n_indices, &
						 sys_current_y, n_hop, wfE_J(2,:))

			! get correlation values
			dc_corrval(1) = inner_prod(wf0_J(1,:),wfE_J(1,:), n_wf)
			dc_corrval(2) = inner_prod(wf0_J(2,:),wfE_J(2,:), n_wf)

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
			CALL cheb_wf_timestep(wfE_J(1,:), n_wf, Bes, n_Bes, &
								  s_indptr, n_indptr, s_indices, n_indices, &
								  -s_hop, n_hop, wfE_J(1,:))
			CALL cheb_wf_timestep(wfE_J(2,:), n_wf, Bes, n_Bes, &
								  s_indptr, n_indptr, s_indices, n_indices, &
								  -s_hop, n_hop, wfE_J(2,:))

			! get correlation values
			dc_corrval(1) = inner_prod(wf0_J(1,:),wfE_J(1,:), n_wf)
			dc_corrval(2) = inner_prod(wf0_J(2,:),wfE_J(2,:), n_wf)

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

	! prepare
	USE tbpm_mod
	IMPLICIT NONE

	! deal with input
	INTEGER, INTENT(in) :: n_Bes, n_indptr, n_indices, n_hop
	INTEGER, INTENT(in) :: n_timestep, seed, n_energies, n_ran_samples
	REAL(8), INTENT(in), DIMENSION(n_Bes) :: Bes
	REAL(8), INTENT(in) :: t_step
	INTEGER, INTENT(in), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(in), DIMENSION(n_indices) :: s_indices
	COMPLEX(8), INTENT(in), DIMENSION(n_hop) :: s_hop
	REAL(8), INTENT(in), DIMENSION(n_energies) :: energies

	! declare vars
	INTEGER :: i, j, k, l, t, n_wf, i_sample
	REAL(8) :: W, QE_sum
	COMPLEX(8), DIMENSION(n_indptr - 1) :: wf0, wf_t_pos, wf_t_neg
	COMPLEX(8), DIMENSION(n_energies,n_indptr-1) :: wfq

	! output
	COMPLEX(8), INTENT(out), DIMENSION(n_energies, n_indptr - 1) :: wf_QE
	n_wf = n_indptr - 1

	wf_QE = 0D0

	PRINT*, "Calculating quasi-eigenstates."

	! Average over (n_ran_samples) samples
	DO i_sample=1, n_ran_samples

		PRINT*, "  Calculating for sample ", i_sample, " of ", n_ran_samples
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
                PRINT*, "    Timestep ", k, " of ", n_timestep
            END IF

            CALL cheb_wf_timestep(wf_t_pos, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, wf_t_pos)
            CALL cheb_wf_timestep(wf_t_neg, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                -s_hop, n_hop, wf_t_neg)

            W = 0.5*(1+cos(pi*k/n_timestep)) ! Hanning window

            !$OMP PARALLEL DO PRIVATE (j)
            DO i = 1, n_energies
                DO j = 1, n_wf
                    wfq(i,j) = wfq(i,j)+&
                        	   exp(img*energies(i)*k*t_step)*wf_t_pos(j)*W
                    wfq(i,j) = wfq(i,j)+&
                        	   exp(-img*energies(i)*k*t_step)*wf_t_neg(j)*W
                END DO
            END DO
            !$OMP END PARALLEL DO

		END DO

		! Normalise
		DO i = 1, n_energies
			QE_sum = 0
			DO j = 1, n_wf
				QE_sum = QE_sum + ABS(wf_QE(i, j))**2
			END DO
			DO j = 1, n_wf
				wf_QE(i, j) = wf_QE(i, j)/DSQRT(QE_sum)
			END DO
		END DO

		wf_QE(:,:) = wf_QE(:,:) + wfq(:,:) / n_ran_samples

	END DO

END SUBROUTINE tbpm_eigenstates

SUBROUTINE ldos_haydock(site_indices, n_siteind, wf_weights, n_wfw, delta, &
						E_range, s_indptr, n_indptr, s_indices, n_indices, &
						s_hop, n_hop, n_depth, n_timestep, &
						output_filename, energy, ldos)
	USE tbpm_mod
	! deal with input
	IMPLICIT NONE
	INTEGER, INTENT(IN) :: n_siteind, n_indptr, n_indices, n_hop
	INTEGER, INTENT(IN) :: n_depth, n_timestep, n_wfw
	INTEGER, INTENT(IN), DIMENSION(n_siteind) :: site_indices
	INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
	INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
	REAL(KIND=8), INTENT(IN) :: E_range, delta
	REAL(KIND=8), INTENT(IN), DIMENSION(n_wfw) :: wf_weights
	COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
	CHARACTER*(*), INTENT(IN) :: output_filename

	! declare variables
	COMPLEX(KIND=8) :: g00
	INTEGER :: i
	COMPLEX(KIND=8), DIMENSION(n_depth) :: coefa
	REAL(KIND=8), DIMENSION(n_depth) :: coefb

	! output
	REAL(KIND=8), INTENT(OUT), DIMENSION(-n_timestep:n_timestep) :: energy
	REAL(KIND=8), INTENT(OUT), DIMENSION(-n_timestep:n_timestep) :: ldos

	energy = (/(0.5*i*E_range/n_timestep, i = -n_timestep, n_timestep)/)
	CALL recursion(site_indices, n_siteind, wf_weights, n_wfw, n_depth, &
				   s_indptr, n_indptr, s_indices, n_indices, &
				   s_hop, n_hop, coefa, coefb)

	DO i = -n_timestep, n_timestep
		CALL green_function(energy(i), delta, coefa, coefb, n_depth, g00)
		ldos(i) = -1D0 / pi * AIMAG(g00)
	END DO
END SUBROUTINE ldos_haydock
