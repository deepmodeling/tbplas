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
    USE propagation
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
    INTEGER :: i_sample, t, n_wf
    COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf_t
    COMPLEX(KIND=8) :: corrval
    TYPE(SPARSE_MATRIX_T) :: H_csr

    ! set some values
    corr = 0D0
    n_wf = n_indptr - 1
    H_csr = make_csr_matrix(s_indptr, s_indices, s_hop)

    OPEN(unit=27, file=output_filename)
    WRITE(27, *) "Number of samples =", n_ran_samples
    WRITE(27, *) "Number of timesteps =", n_timestep

    PRINT *, "Calculating DOS correlation function."

    ! Average over (n_ran_samples) samples
    DO i_sample = 1, n_ran_samples
        PRINT *, "Sample ", i_sample, " of ", n_ran_samples
        WRITE(27, *) "Sample =", i_sample

        ! make random state
        CALL random_state(wf0, n_wf, seed*i_sample)
        wf_t = cheb_wf_timestep_fwd(H_csr, Bes, wf0)
        corrval = inner_prod(wf0, wf_t)

        WRITE(27, *) 1, REAL(corrval), AIMAG(corrval)
        corr(1) = corr(1) + corrval / n_ran_samples

        ! iterate over time, get correlation function
        DO t = 2, n_timestep
            IF (MODULO(t, 64) == 0) THEN
                PRINT *, "    Timestep ", t, " of ", n_timestep
            END IF

            wf_t = cheb_wf_timestep_fwd(H_csr, Bes, wf_t)
            corrval = inner_prod(wf0, wf_t)

            WRITE(27, *) t, REAL(corrval), AIMAG(corrval)
            corr(t) = corr(t) + corrval / n_ran_samples
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
    USE propagation
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
    INTEGER :: k, i, n_wf, i_sample
    COMPLEX(KIND=8), DIMENSION(n_siteind) :: wf_temp
    COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf_t
    COMPLEX(KIND=8) :: corrval
    TYPE(SPARSE_MATRIX_T) :: H_csr

    ! set some values
    corr = 0D0
    n_wf = n_indptr - 1
    H_csr = make_csr_matrix(s_indptr, s_indices, s_hop)

    OPEN(unit=27, file=output_filename)
    WRITE(27, *) "Number of samples =", n_ran_samples
    WRITE(27, *) "Number of timesteps =", n_timestep

    PRINT *, "Calculating LDOS correlation function."

    DO i_sample = 1, n_ran_samples
        PRINT *, "Sample ", i_sample, " of ", n_ran_samples
        WRITE(27, *) "Sample =", i_sample

        ! make LDOS state
        CALL random_state(wf_temp, n_siteind, seed*i_sample)
        wf0 = 0D0
        DO i = 1, SIZE(site_indices)
            wf0(site_indices(i) + 1) = wf_temp(i) * wf_weights(i)
        END DO
        wf_t = cheb_wf_timestep_fwd(H_csr, Bes, wf0)
        corrval = inner_prod(wf0, wf_t)

        WRITE(27, *) 1, REAL(corrval), AIMAG(corrval)
        corr(1) = corr(1) + corrval / n_ran_samples

        ! iterate over time, get correlation function
        DO k = 2, n_timestep
            IF (MODULO(k, 64) == 0) THEN
                PRINT *, "    Timestep ", k, " of ", n_timestep
            END IF

            wf_t = cheb_wf_timestep_fwd(H_csr, Bes, wf_t)
            corrval = inner_prod(wf0, wf_t)

            WRITE(27, *) k, REAL(corrval), AIMAG(corrval)
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
    USE propagation
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
    INTEGER :: i_sample, t, n_cheb, n_wf
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
    OPEN(unit=27, file=output_filename)
    WRITE(27, *) "Number of samples =", n_ran_samples
    WRITE(27, *) "Number of timesteps =", n_timestep

    ! get current coefficients
    CALL current_coefficient(s_hop, s_dx, n_hop, H_rescale, sys_current_x)
    cur_csr_x = make_csr_matrix(s_indptr, s_indices, sys_current_x)
    CALL current_coefficient(s_hop, s_dy, n_hop, H_rescale, sys_current_y)
    cur_csr_y = make_csr_matrix(s_indptr, s_indices, sys_current_y)

    ! get Fermi cheb coefficients
    CALL get_Fermi_cheb_coef(coef1, n_cheb, nr_Fermi, beta, &
                             mu, .FALSE., Fermi_precision)
    coef_F => coef1(1:n_cheb)
    ! get one minus Fermi cheb coefficients
    CALL get_Fermi_cheb_coef(coef2, n_cheb, nr_Fermi, beta, &
                             mu, .TRUE., Fermi_precision)
    coef_omF => coef2(1:n_cheb)

    H_csr = make_csr_matrix(s_indptr, s_indices, s_hop)

    PRINT *, "Calculating AC conductivity correlation function."

    ! Average over (n_sample) samples
    DO i_sample = 1, n_ran_samples

        PRINT *, "Sample ", i_sample, " of ", n_ran_samples
        WRITE(27, *) "Sample =", i_sample

        ! make random state and psi1, psi2
        CALL random_state(wf0, n_wf, seed*i_sample)
        psi1_x = cur_csr_x * wf0
        psi1_y = cur_csr_y * wf0
        psi1_x = Fermi(H_csr, coef_omF, psi1_x)
        psi1_y = Fermi(H_csr, coef_omF, psi1_y)
        psi2 = Fermi(H_csr, coef_F, wf0)

        !get correlation functions in all directions
        wf1 = cur_csr_x * psi1_x
        corrval(1) = inner_prod(psi2, wf1)
        wf1 = cur_csr_y * psi1_x
        corrval(2) = inner_prod(psi2, wf1)
        wf1 = cur_csr_x * psi1_y
        corrval(3) = inner_prod(psi2, wf1)
        wf1 = cur_csr_y * psi1_y
        corrval(4) = inner_prod(psi2, wf1)

        ! write to file
        WRITE(27, "(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3, &
                  & ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)") &
            1, &
            REAL(corrval(1)), AIMAG(corrval(1)), &
            REAL(corrval(2)), AIMAG(corrval(2)), &
            REAL(corrval(3)), AIMAG(corrval(3)), &
            REAL(corrval(4)), AIMAG(corrval(4))

        ! iterate over time
        DO t = 2, n_timestep
            IF (MODULO(t, 64) == 0) THEN
                PRINT *, "    Timestep ", t, " of ", n_timestep
            END IF

            ! calculate time evolution
            psi1_x = cheb_wf_timestep_fwd(H_csr, Bes, psi1_x)
            psi1_y = cheb_wf_timestep_fwd(H_csr, Bes, psi1_y)
            psi2 = cheb_wf_timestep_fwd(H_csr, Bes, psi2)

            !get correlation functions in all directions
            wf1 = cur_csr_x * psi1_x
            corrval(1) = inner_prod(psi2, wf1)
            wf1 = cur_csr_y * psi1_x
            corrval(2) = inner_prod(psi2, wf1)
            wf1 = cur_csr_x * psi1_y
            corrval(3) = inner_prod(psi2, wf1)
            wf1 = cur_csr_y * psi1_y
            corrval(4) = inner_prod(psi2, wf1)

            ! write to file
            WRITE(27, "(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3, &
                      & ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)") &
                t, &
                REAL(corrval(1)), AIMAG(corrval(1)), &
                REAL(corrval(2)), AIMAG(corrval(2)), &
                REAL(corrval(3)), AIMAG(corrval(3)), &
                REAL(corrval(4)), AIMAG(corrval(4))

            ! update output array
            corr(1, t) = corr(1, t) + corrval(1) / n_ran_samples
            corr(2, t) = corr(2, t) + corrval(2) / n_ran_samples
            corr(3, t) = corr(3, t) + corrval(3) / n_ran_samples
            corr(4, t) = corr(4, t) + corrval(4) / n_ran_samples
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
    USE math
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
    INTEGER :: i_sample, t, n_cheb1, n_cheb2, i_q, n_wf
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
    CALL get_Fermi_cheb_coef(coef1, n_cheb1, nr_Fermi, beta, &
                             mu, .FALSE., Fermi_precision)
    coef_F => coef1(1:n_cheb1)
    ! get one minus Fermi cheb coefficients
    CALL get_Fermi_cheb_coef(coef2, n_cheb2, nr_Fermi, beta, &
                             mu, .TRUE., Fermi_precision)
    coef_omF => coef2(1:n_cheb2)

    H_csr = make_csr_matrix(s_indptr, s_indices, s_hop)

    OPEN(unit=27, file=output_filename)
    WRITE(27, *) "Number of qpoints =", n_q_points
    WRITE(27, *) "Number of samples =", n_ran_samples
    WRITE(27, *) "Number of timesteps =", n_timestep

    PRINT *, "Calculating dynamical polarization correlation function."

    !loop over n qpoints
    DO i_q = 1, n_q_points
        PRINT *, "q-point ", i_q, " of ", n_q_points
        WRITE(27, "(A9, ES24.14E3,ES24.14E3,ES24.14E3)") "qpoint= ", &
            q_points(i_q,1), q_points(i_q,2), q_points(i_q,3)

        !calculate the coefficients for the density operator
        !exp(i * q dot r)
        CALL density_coef(n_wf, s_site_x, s_site_y, s_site_z, &
                          q_points(i_q,:), s_density_q, s_density_min_q)

        ! Average over (n_ran_samples) samples
        DO i_sample = 1, n_ran_samples
            PRINT *, " Sample ", i_sample, " of ", n_ran_samples
            WRITE(27, *) "Sample =", i_sample

            ! make random state and psi1, psi2
            CALL random_state(wf0, n_wf, seed*i_sample)
            ! call density(-q)*wf0, resulting in psi1
            psi1 = s_density_min_q .pMul. wf0
            ! call fermi with 1-fermi coefficients for psi1
            psi1 = Fermi(H_csr, coef_omF, psi1)
            ! call fermi with fermi coefficients for psi2
            psi2 = Fermi(H_csr, coef_F, wf0)
            ! call density(q)*psi1, resulting in wf1
            wf1 = s_density_q .pMul. psi1

            ! get correlation and store
            corrval = AIMAG(inner_prod(psi2, wf1))
            WRITE(27, *) 1, corrval
            corr(i_q, 1) = corr(i_q, 1) + corrval / n_ran_samples

            ! iterate over tau
            DO t = 2, n_timestep
            IF (MODULO(t, 64) == 0) THEN
                PRINT *, "    Timestep ", t, " of ", n_timestep
            END IF

            ! call time and density operators
            psi1 = cheb_wf_timestep_fwd(H_csr, Bes, psi1)
            psi2 = cheb_wf_timestep_fwd(H_csr, Bes, psi2)
            wf1 = s_density_q .pMul. psi1

            ! get correlation and store
            corrval = AIMAG(inner_prod(psi2, wf1))
            WRITE(27, *) t, corrval
            corr(i_q, t) = corr(i_q, t) + corrval / n_ran_samples

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
    USE math
    USE random
    USE csr
    USE propagation
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
    INTEGER :: i_sample, i, j, t, n_wf
    REAL(KIND=8) :: W, QE_sum
    COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf_t_pos, wf_t_neg, wfE
    COMPLEX(KIND=8), DIMENSION(n_en_inds, n_indptr - 1) :: wf_QE
    COMPLEX(KIND=8), DIMENSION(2, n_indptr - 1) :: wf0_J, wfE_J
    ! coefs for current
    COMPLEX(KIND=8), DIMENSION(n_hop) :: sys_current_y
    COMPLEX(KIND=8) :: dos_corrval, P
    COMPLEX(KIND=8), DIMENSION(2) :: dc_corrval
    TYPE(SPARSE_MATRIX_T) :: H_csr, cur_csr_y

    ! set some values
    n_wf = n_indptr - 1
    dos_corr = 0D0
    dc_corr = 0D0
    H_csr = make_csr_matrix(s_indptr, s_indices, s_hop)

    ! prepare output files
    OPEN(unit=27,file=output_filename_dos)
    WRITE(27,*) "Number of samples =", n_ran_samples
    WRITE(27,*) "Number of timesteps =", n_timestep

    OPEN(unit=28,file=output_filename_dc)
    WRITE(28,*) "Number of samples =", n_ran_samples
    WRITE(28,*) "Number of energies =", n_en_inds
    WRITE(28,*) "Number of timesteps =", n_timestep

    ! get current coefficients
    CALL current_coefficient(s_hop, s_dy, n_hop, H_rescale, sys_current_y)
    cur_csr_y = make_csr_matrix(s_indptr, s_indices, sys_current_y)

    PRINT *, "Calculating DC conductivity correlation function."

    ! Average over (n_ran_samples) samples
    DO i_sample = 1, n_ran_samples
        PRINT *, "Calculating for sample ", i_sample, " of ", n_ran_samples
        WRITE(27, *) "Sample =", i_sample
        WRITE(28, *) "Sample =", i_sample

        ! make random state
        CALL random_state(wf0, n_wf, seed*i_sample)

        ! ------------
        ! first, get DOS and quasi-eigenstates
        ! ------------

        ! initial values for wf_t and wf_QE
        wf_t_pos = copy(wf0)
        wf_t_neg = copy(wf0)
        DO i = 1, n_en_inds
            wf_QE(i, :) = copy(wf0)
        END DO

        ! Iterate over time, get Fourier transform
        DO t = 1, n_timestep
            IF (MODULO(t, 64) == 0) THEN
                PRINT *, "Getting DOS/QE for timestep ", t, " of ", n_timestep
            END IF

            ! time evolution
            wf_t_pos = cheb_wf_timestep_fwd(H_csr, Bes, wf_t_pos)
            wf_t_neg = cheb_wf_timestep_inv(H_csr, Bes, wf_t_neg)

            ! get dos correlation
            dos_corrval = inner_prod(wf0, wf_t_pos)
            dos_corr(t) = dos_corr(t) + dos_corrval/n_ran_samples
            WRITE(27, *) t, REAL(dos_corrval), AIMAG(dos_corrval)

            W = 0.5 * (1 + COS(pi * t / n_timestep)) ! Hanning window

            !$OMP PARALLEL DO SIMD PRIVATE(P, j)
            DO i = 1, n_en_inds
                P = EXP(img * energies(en_inds(i) + 1) * t * t_step)
                DO j = 1, n_wf
                    wf_QE(i, j) = wf_QE(i, j) + P * wf_t_pos(j) * W
                    wf_QE(i, j) = wf_QE(i, j) + CONJG(P) * wf_t_neg(j) * W
                END DO
            END DO
            !$OMP END PARALLEL DO SIMD
        END DO

        ! Normalise
        DO i = 1, n_en_inds
            QE_sum = norm(wf_QE(i, :))
            CALL self_div(wf_QE(i, :), QE_sum)
        END DO

        ! ------------
        ! then, get dc conductivity
        ! ------------

        ! iterate over energies
        DO i = 1, n_en_inds
            ! some output
            IF (MODULO(i, 8) == 0) THEN
                PRINT *, "Getting DC conductivity for energy: ", &
                        i, " of ", n_en_inds
            END IF
            WRITE(28, *) "Energy ", i, en_inds(i), energies(en_inds(i) + 1)

            ! get corresponding quasi-eigenstate
            wfE(:) = wf_QE(i, :) .pDiv. ABS(inner_prod(wf0, wf_QE(i, :)))

            ! make psi1, psi2
            wf0_J(1, :) = cur_csr_y * wf0
            wf0_J(2, :) = copy(wf0_J(1, :))
            wfE_J(1, :) = cur_csr_y * wfE
            wfE_J(2, :) = copy(wfE_J(1, :))

            ! get correlation values
            dc_corrval(1) = inner_prod(wf0_J(1, :), wfE_J(1, :))
            dc_corrval(2) = inner_prod(wf0_J(2, :), wfE_J(2, :))

            ! write to file
            WRITE(28, "(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)") &
                1, &
                REAL(dc_corrval(1)), AIMAG(dc_corrval(1)), &
                REAL(dc_corrval(2)), AIMAG(dc_corrval(2))

            ! update correlation functions
            dc_corr(1, i, 1) = dc_corr(1, i, 1) + dc_corrval(1)/n_ran_samples
            dc_corr(2, i, 1) = dc_corr(2, i, 1) + dc_corrval(2)/n_ran_samples

            ! iterate over time
            DO t = 2, n_timestep
                ! NEGATIVE time evolution of QE state
                wfE_J(1, :) = cheb_wf_timestep_inv(H_csr, Bes, wfE_J(1, :))
                wfE_J(2, :) = cheb_wf_timestep_inv(H_csr, Bes, wfE_J(2, :))

                ! get correlation values
                dc_corrval(1) = inner_prod(wf0_J(1, :), wfE_J(1, :))
                dc_corrval(2) = inner_prod(wf0_J(2, :), wfE_J(2, :))

                ! write to file
                WRITE(28, "(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)") &
                    t, &
                    REAL(dc_corrval(1)), AIMAG(dc_corrval(1)), &
                    REAL(dc_corrval(2)), AIMAG(dc_corrval(2))

                ! update correlation functions
                dc_corr(1,i,t) = dc_corr(1,i,t) + dc_corrval(1)/n_ran_samples
                dc_corr(2,i,t) = dc_corr(2,i,t) + dc_corrval(2)/n_ran_samples
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
    USE math
    USE random
    USE csr
    USE propagation
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
    INTEGER :: i, j, k, n_wf, i_sample
    REAL(KIND=8) :: W, QE_sum
    COMPLEX(KIND=8) :: P
    COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf0, wf_t_pos, wf_t_neg
    COMPLEX(KIND=8), DIMENSION(n_energies, n_indptr - 1) :: wfq
    TYPE(SPARSE_MATRIX_T) :: H_csr

    n_wf = n_indptr - 1
    wf_QE = 0D0
    H_csr = make_csr_matrix(s_indptr, s_indices, s_hop)

    PRINT *, "Calculating quasi-eigenstates."

    ! Average over (n_ran_samples) samples
    DO i_sample = 1, n_ran_samples

        PRINT *, "  Calculating for sample ", i_sample, " of ", n_ran_samples
        ! make random state
        CALL random_state(wf0, n_wf, seed*i_sample)

        ! initial values for wf_t and wf_QE
        wf_t_pos = copy(wf0)
        wf_t_neg = copy(wf0)
        DO i = 1, n_energies
            wfq(i, :) = copy(wf0)
        END DO

        ! Iterate over time, get Fourier transform
        DO k = 1, n_timestep
            IF (MODULO(k, 64) == 0) THEN
                PRINT *, "    Timestep ", k, " of ", n_timestep
            END IF

            wf_t_pos = cheb_wf_timestep_fwd(H_csr, Bes, wf_t_pos)
            wf_t_neg = cheb_wf_timestep_inv(H_csr, Bes, wf_t_neg)

            W = 0.5 * (1 + COS(pi * k / n_timestep)) ! Hanning window

            !$OMP PARALLEL DO PRIVATE (P, j)
            DO i = 1, n_energies
                P = EXP(img * energies(i) * k * t_step)
                DO j = 1, n_wf
                    wfq(i, j) = wfq(i, j)+ P * wf_t_pos(j) * W
                    wfq(i, j) = wfq(i, j)+ CONJG(P) * wf_t_neg(j) * W
                END DO
            END DO
            !$OMP END PARALLEL DO
        END DO

        ! Normalise
        DO i = 1, n_energies
            QE_sum = norm(wfq(i, :))
            CALL self_div(wfq(i, :), QE_sum)
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
                     n_ran_samples, n_kernel, iTypeDC, corr_mu_avg)
    USE math
    USE random
    USE csr
    USE propagation
    USE funcs
    USE kpm
    IMPLICIT NONE

    ! deal with input
    INTEGER, INTENT(IN) :: iTypeDC, seed, n_ran_samples, n_kernel
    INTEGER, INTENT(IN) :: n_indptr, n_indices, n_hop, n_dx, n_dy
    REAL(KIND=8), INTENT(IN) :: H_rescale
    INTEGER, INTENT(IN), DIMENSION(n_indptr) :: s_indptr
    INTEGER, INTENT(IN), DIMENSION(n_indices) :: s_indices
    COMPLEX(KIND=8), INTENT(IN), DIMENSION(n_hop) :: s_hop
    REAL(KIND=8), INTENT(IN), DIMENSION(n_dx) :: s_dx
    REAL(KIND=8), INTENT(IN), DIMENSION(n_dy) :: s_dy
    ! output
    COMPLEX(KIND=8), INTENT(OUT), DIMENSION(n_kernel, n_kernel) :: corr_mu_avg

    !declare vars
    INTEGER :: i, j, i_sample, n_wf
    REAL(KIND=8), DIMENSION(n_kernel) :: g
    COMPLEX(KIND=8), DIMENSION(n_indptr - 1) :: wf_in
    COMPLEX(KIND=8), DIMENSION(n_hop) :: sys_current_x
    COMPLEX(KIND=8), DIMENSION(n_hop) :: sys_current_y
    COMPLEX(KIND=8), DIMENSION(n_indptr - 1), TARGET :: wf0, wf1
    COMPLEX(KIND=8), DIMENSION(:), POINTER :: p0, p1, p2
    COMPLEX(KIND=8), DIMENSION(n_indptr-1, n_kernel) :: wf_DimKern
    COMPLEX(KIND=8), DIMENSION(n_kernel, n_kernel) :: corr_mu
    TYPE(SPARSE_MATRIX_T) :: H_csr, cur_csr_x, cur_csr_y

    n_wf = n_indptr - 1
    ! write(*,*) 'go into subroutine current_coefficient now'
    CALL current_coefficient(s_hop, s_dx, n_hop, H_rescale, sys_current_x)
    cur_csr_x = make_csr_matrix(s_indptr, s_indices, sys_current_x)
    IF (iTypeDC == 2) THEN
        CALL current_coefficient(s_hop, s_dy, n_hop, H_rescale, sys_current_y)
        cur_csr_y = make_csr_matrix(s_indptr, s_indices, sys_current_y)
    END IF

    H_csr = make_csr_matrix(s_indptr, s_indices, s_hop)

    corr_mu_avg = 0D0

    CALL jackson_kernel(g, n_kernel)

    ! iterate over random states
    DO i_sample = 1, n_ran_samples
        PRINT*, "  Calculating for sample ", i_sample, " of ", n_ran_samples
        ! get random state
        CALL random_state(wf_in, n_wf, seed*i_sample)

        wf_DimKern(:, 1) = copy(wf_in)
        wf_DimKern(:, 2) = H_csr * wf_DimKern(:, 1)

        DO j = 3, n_kernel
            IF (MOD(j, 256) == 0) THEN
                PRINT *, "    Currently at j = ", j
            END IF

            wf_DimKern(:, j) = H_csr * wf_DimKern(:, j-1)
            CALL axpby(-1D0, wf_DimKern(:, j-2), 2D0, wf_DimKern(:, j))
        END DO

        ! for xx direction
        IF(iTypeDC == 1) THEN
            DO j = 1, n_kernel
                wf0 = copy(wf_DimKern(:, j))
                wf_DimKern(:, j) = cur_csr_x * wf0
            END DO
        ! for xy direction
        ELSE IF(iTypeDC == 2) THEN
            DO j = 1, n_kernel
                wf0 = copy(wf_DimKern(:, j))
                wf_DimKern(:, j) = cur_csr_y * wf0
            END DO
        ELSE
            PRINT*, "Error: wrong direction!"
            STOP
        END IF

        ! calculate correlation
        ! i = 1
        wf0 = cur_csr_x * wf_in
        DO j = 1, n_kernel
            corr_mu(1, j) = inner_prod(wf0, wf_DimKern(:, j))
        END DO

        ! i = 2
        wf1 = H_csr * wf0
        DO j = 1, n_kernel
            corr_mu(2, j) = inner_prod(wf1, wf_DimKern(:, j))
        END DO

        p0 => wf0
        p1 => wf1
        DO i = 3, n_kernel
            IF (MOD(i, 256) == 0) THEN
                PRINT *, "    Currently at i = ", i
            END IF

            p2 => p0
            CALL amxpby(2D0, H_csr, p1, -1D0, p0) ! p2 = 2 * H_csr * p1 - p0

            DO j = 1, n_kernel
                corr_mu(i, j) = inner_prod(p2, wf_DimKern(:, j))
            END DO
            p0 => p1
            p1 => p2
        END DO

        ! add Jackson kernel and get avg
        !$OMP PARALLEL DO
        DO j = 1, n_kernel
            corr_mu_avg(:, j) = corr_mu_avg(:, j) &
                                + g(:)*g(j)*corr_mu(:, j) / n_ran_samples
        END DO
        !$OMP END PARALLEL DO
    END DO
END SUBROUTINE tbpm_kbdc
