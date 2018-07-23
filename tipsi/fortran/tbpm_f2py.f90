! ------------------------------------------
! TBPM fortran subroutines, callable from python
! ------------------------------------------

! Get DOS
subroutine tbpm_dos(Bes, n_Bes, &
    s_indptr, n_indptr, s_indices, n_indices, s_hop, n_hop, &
    seed, n_timestep, n_ran_samples, output_filename, corr)
    
    ! prepare
    use tbpm_mod
    implicit none
    
    ! deal with input
    integer, intent(in) :: n_Bes, n_indptr, n_indices, n_hop
    integer, intent(in) :: n_timestep, n_ran_samples, seed
    real(8), intent(in), dimension(n_Bes) :: Bes
    integer(8), intent(in), dimension(n_indptr) :: s_indptr
    integer(8), intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop
    character*(*), intent(in) :: output_filename    

    ! declare vars
    integer :: i_sample,  k, i, n_wf
    complex(8), dimension(n_indptr - 1) :: wf0, wf_t
    complex(8) :: corrval
 
    ! output
    complex(8), intent(out), dimension(n_timestep) :: corr
    
    ! set some values
    corr = 0.0d0
    n_wf = n_indptr - 1
    
    open(unit=27,file=output_filename)
    write(27,*) "Number of samples =", n_ran_samples
    write(27,*) "Number of timesteps =", n_timestep

    print*, "Calculating DOS correlation function."
    
    ! Average over (n_ran_samples) samples
    do i_sample = 1, n_ran_samples
        
        print*, "  Sample ", i_sample, " of ", n_ran_samples
        write(27,*) "Sample =", i_sample

        ! make random state
        call random_state(wf0, n_wf, seed*i_sample)

        do i = 1, n_wf
            wf_t(i) = wf0(i)
        end do
        
        ! iterate over time, get correlation function
        do k = 1, n_timestep
        
            if (MODULO(k,64) == 0) then
                print*, "    Timestep ", k, " of ", n_timestep
            end if
            
            call cheb_wf_timestep(wf_t, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, wf_t)
            corrval = inner_prod(wf0, wf_t, n_wf)
            
            write(27,*) k, real(corrval), aimag(corrval)
            
            corr(k) = corr(k) + corrval / n_ran_samples
            
        end do
        
    end do
    
    close(27)
    
end subroutine tbpm_dos

! Get LDOS
subroutine tbpm_ldos(site_indices, n_siteind, wf_weights, n_wfw, &
    Bes, n_Bes, s_indptr, n_indptr, s_indices, n_indices, &
    s_hop, n_hop, seed, n_timestep, n_ran_samples, &
    output_filename, corr)
    
    ! prepare
    use tbpm_mod
    implicit none
    
    ! deal with input
    integer, intent(in) :: n_Bes, n_indptr, n_indices, n_hop
    integer, intent(in) :: n_timestep, seed, n_siteind, n_wfw
    integer, intent(in) :: n_ran_samples
    integer, intent(in), dimension(n_siteind) :: site_indices
    real(8), intent(in), dimension(n_wfw) :: wf_weights
    real(8), intent(in), dimension(n_Bes) :: Bes
    integer(8), intent(in), dimension(n_indptr) :: s_indptr
    integer(8), intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop
    character*(*), intent(in) :: output_filename    

    ! declare vars
    integer :: k, i, n_wf, i_sample
    complex(8), dimension(n_siteind) :: wf_temp
    complex(8), dimension(n_indptr - 1) :: wf0, wf_t
    complex(8) :: corrval
 
    ! output
    complex(8), intent(out), dimension(n_timestep) :: corr
    
    ! set some values
    corr = 0.0d0
    n_wf = n_indptr - 1
    
    open(unit=27,file=output_filename)
    write(27,*) "Number of samples =", n_ran_samples
    write(27,*) "Number of timesteps =", n_timestep

    print*, "Calculating LDOS correlation function."
    
    do i_sample=1, n_ran_samples
        
        print*, "  Sample ", i_sample, " of ", n_ran_samples
        write(27,*) "Sample =", i_sample

        ! make LDOS state
        call random_state(wf_temp, n_siteind, seed*i_sample)
        wf0 = 0.0d0
        do i = 1, n_siteind
            wf0(site_indices(i) + 1) = wf_temp(i) * wf_weights(i)
        end do
        do i = 1, n_wf
            wf_t(i) = wf0(i)
        end do

        ! iterate over time, get correlation function
        do k = 1, n_timestep

            if (MODULO(k,64) == 0) then
                print*, "    Timestep ", k, " of ", n_timestep
            end if

            call cheb_wf_timestep(wf_t, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, wf_t)
            corrval = inner_prod(wf0, wf_t, n_wf)

            write(27,*) k, real(corrval), aimag(corrval)

            corr(k) = corr(k) + corrval / n_ran_samples

        end do
        
    end do
    
    close(27)
    
end subroutine tbpm_ldos

! Get AC conductivity
subroutine tbpm_accond(Bes, n_bes, beta, mu, &
    s_indptr, n_indptr, s_indices, n_indices, s_hop, n_hop, H_rescale, &
    s_dx, n_dx, s_dy, n_dy, seed, n_timestep, n_ran_samples, &
    nr_Fermi, Fermi_precision, output_filename, corr)
    
    ! prepare
    use tbpm_mod
    implicit none
    
    ! deal with input
    integer, intent(in) :: n_Bes, n_indptr, n_indices, n_hop, n_dx, n_dy
    integer, intent(in) :: n_timestep, n_ran_samples, seed
    integer, intent(in) :: nr_Fermi
    real(8), intent(in) :: Fermi_precision, H_rescale, beta, mu
    real(8), intent(in), dimension(n_Bes) :: Bes
    integer(8), intent(in), dimension(n_indptr) :: s_indptr
    integer(8), intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop
    real(8), intent(in), dimension(n_dx) :: s_dx
    real(8), intent(in), dimension(n_dy) :: s_dy
    character*(*), intent(in) :: output_filename  
    
    ! declare vars
    integer :: i_sample, k, i, j, n_cheb, n_wf
    complex(8), dimension(n_indptr - 1) :: wf0,wf1,psi1_x,psi1_y,psi2
    real(8), allocatable :: coef(:)
    complex(8), dimension(4) :: corrval
    real(8), allocatable :: coef_F(:) ! cheb coefs for Fermi operator
    real(8), allocatable :: coef_omF(:) ! cheb coefs one minus Fermi operator
    complex(8), dimension(n_hop) :: sys_current_x ! coefs for x current
    complex(8), dimension(n_hop) :: sys_current_y ! coefs for y current
    
    ! output
    complex(8), intent(out), dimension(4, n_timestep) :: corr
    ! corr has 4 elements, respectively: corr_xx, corr_xy, corr_yx, corr_yy
    
    ! set some values
    corr = 0.0d0
    n_wf = n_indptr - 1
    
    ! prepare output file
    open(unit=27,file=output_filename)
    write(27,*) "Number of samples =", n_ran_samples
    write(27,*) "Number of timesteps =", n_timestep
    
    ! get current coefficients
    call current_coefficient(s_hop, s_dx, n_hop, sys_current_x)
    call current_coefficient(s_hop, s_dy, n_hop, sys_current_y)
    sys_current_x = H_rescale * sys_current_x
    sys_current_y = H_rescale * sys_current_y

    ! get Fermi cheb coefficients
    allocate(coef(nr_Fermi))
    coef = 0.0d0
    call get_Fermi_cheb_coef(coef, n_cheb, nr_Fermi,&
        beta, mu, .FALSE., Fermi_precision)
    allocate(coef_F(n_cheb))
    do i=1, n_cheb
        coef_F(i) = coef(i)
    end do
    
    ! get one minus Fermi cheb coefficients
    coef = 0.0d0
    call get_Fermi_cheb_coef(coef, n_cheb,&
        nr_Fermi, beta, mu, .TRUE., Fermi_precision)
    allocate(coef_omF(n_cheb))
    do i=1, n_cheb
        coef_omF(i) = coef(i)
    end do
    deallocate(coef)
        
    print*, "Calculating AC conductivity correlation function."
                
    ! Average over (n_sample) samples
    corr = 0.0d0
    do i_sample=1, n_ran_samples
        
        print*, "  Sample ", i_sample, " of ", n_ran_samples
        write(27,*) "Sample =", i_sample

        ! make random state and psi1, psi2
        call random_state(wf0, n_wf, seed*i_sample)
        call current(wf0, n_wf, s_indptr, n_indptr, s_indices, &
            n_indices, sys_current_x, n_hop, psi1_x)
        call current(wf0, n_wf, s_indptr, n_indptr, s_indices, &
            n_indices, sys_current_y, n_hop, psi1_y)
        call Fermi(psi1_x, n_wf, coef_omF, n_cheb, &
            s_indptr, n_indptr, s_indices, n_indices, &
            s_hop, n_hop, psi1_x)
        call Fermi(psi1_y, n_wf, coef_omF, n_cheb, &
            s_indptr, n_indptr, s_indices, n_indices, &
            s_hop, n_hop, psi1_y)
        call Fermi(wf0, n_wf, coef_F, n_cheb, &
            s_indptr, n_indptr, s_indices, n_indices, &
            s_hop, n_hop, psi2)

        !get correlation functions in all directions
        call current(psi1_x, n_wf, s_indptr, n_indptr, s_indices, &
            n_indices, sys_current_x, n_hop, wf1)
        corrval(1) = inner_prod(psi2, wf1, n_wf)
        call current(psi1_x, n_wf, s_indptr, n_indptr, s_indices, &
            n_indices, sys_current_y, n_hop, wf1)
        corrval(2) = inner_prod(psi2, wf1, n_wf)
        call current(psi1_y, n_wf, s_indptr, n_indptr, s_indices, &
            n_indices, sys_current_x, n_hop, wf1)
        corrval(3) = inner_prod(psi2, wf1, n_wf)
        call current(psi1_y, n_wf, s_indptr, n_indptr, s_indices, &
            n_indices, sys_current_y, n_hop, wf1)
        corrval(4) = inner_prod(psi2, wf1, n_wf)

        ! write to file
        write(27,"(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3,&
                   ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)")&
             1, &
             real(corrval(1)), aimag(corrval(1)), &
             real(corrval(2)), aimag(corrval(2)), &
             real(corrval(3)), aimag(corrval(3)), &
             real(corrval(4)), aimag(corrval(4))
                 
        ! iterate over time
        do k = 2, n_timestep
 
            if (MODULO(k,64) == 0) then
                print*, "    Timestep ", k, " of ", n_timestep
            end if
            
            ! calculate time evolution
            call cheb_wf_timestep(psi1_x, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, psi1_x)
            call cheb_wf_timestep(psi1_y, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, psi1_y)
            call cheb_wf_timestep(psi2, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, psi2)

            !get correlation functions in all directions
            call current(psi1_x, n_wf, s_indptr, n_indptr, s_indices, &
                n_indices, sys_current_x, n_hop, wf1)
            corrval(1) = inner_prod(psi2, wf1, n_wf)
            call current(psi1_x, n_wf, s_indptr, n_indptr, s_indices, &
                n_indices, sys_current_y, n_hop, wf1)
            corrval(2) = inner_prod(psi2, wf1, n_wf)
            call current(psi1_y, n_wf, s_indptr, n_indptr, s_indices, &
                n_indices, sys_current_x, n_hop, wf1)
            corrval(3) = inner_prod(psi2, wf1, n_wf)
            call current(psi1_y, n_wf, s_indptr, n_indptr, s_indices, &
                n_indices, sys_current_y, n_hop, wf1)
            corrval(4) = inner_prod(psi2, wf1, n_wf)

            ! write to file
            write(27,"(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3,&
                       ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)")&
                 k, &
                 real(corrval(1)), aimag(corrval(1)), &
                 real(corrval(2)), aimag(corrval(2)), &
                 real(corrval(3)), aimag(corrval(3)), &
                 real(corrval(4)), aimag(corrval(4))
            
            ! update output array
            corr(1,k) = corr(1,k) + corrval(1) / n_ran_samples
            corr(2,k) = corr(2,k) + corrval(2) / n_ran_samples
            corr(3,k) = corr(3,k) + corrval(3) / n_ran_samples
            corr(4,k) = corr(4,k) + corrval(4) / n_ran_samples
            
        end do
    
    end do
    
    close(27)
    
end subroutine tbpm_accond

! Get dynamical polarization
subroutine tbpm_dyn_pol(Bes, n_bes, beta, mu, &
    s_indptr, n_indptr, s_indices, n_indices, s_hop, n_hop, H_rescale, &
    s_dx, n_dx, s_dy, n_dy, s_site_x, n_site_x, s_site_y, n_site_y, &
    s_site_z, n_site_z, seed, n_timestep, n_ran_samples, nr_Fermi, & 
    Fermi_precision, q_points, n_q_points, output_filename, corr)
    
    ! prepare
    use tbpm_mod
    implicit none
    
    ! deal with input
    integer, intent(in) :: n_Bes, n_indptr, n_indices, n_hop, n_dx, n_dy
    integer, intent(in) :: n_timestep, n_ran_samples, seed, n_q_points
    integer, intent(in) :: n_site_x, n_site_y, n_site_z, nr_Fermi
    real(8), intent(in) :: Fermi_precision, H_rescale, beta, mu
    real(8), intent(in), dimension(n_Bes) :: Bes
    integer(8), intent(in), dimension(n_indptr) :: s_indptr
    integer(8), intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop
    real(8), intent(in), dimension(n_dx) :: s_dx
    real(8), intent(in), dimension(n_dy) :: s_dy
    real(8), intent(in), dimension(n_site_x) :: s_site_x
    real(8), intent(in), dimension(n_site_y) :: s_site_y
    real(8), intent(in), dimension(n_site_z) :: s_site_z
    real(8), intent(in), dimension(n_q_points, 3) :: q_points
    character*(*), intent(in) :: output_filename  
    
    ! declare vars
    integer :: i_sample, k, i, j, n_cheb, i_q, n_wf
    real(8) :: omega, eps, W, tau, dpi, dpr, corrval
    complex(8), dimension(n_indptr - 1) :: wf0, wf1, psi1, psi2
    real(8), allocatable :: coef(:) ! temp variable
    real(8), allocatable :: coef_F(:) ! cheb coefs for Fermi operator
    real(8), allocatable :: coef_omF(:) ! cheb coefs one minus Fermi operator
    complex(8), dimension(n_indptr - 1 ) :: s_density_q, s_density_min_q ! coefs for density

    real(8), intent(out), dimension(n_q_points, n_timestep) :: corr
    n_wf = n_indptr - 1 
    corr = 0.0d0
    
    ! get Fermi cheb coefficients
    allocate(coef(nr_Fermi))
    coef = 0.0d0
    call get_Fermi_cheb_coef(coef,n_cheb,nr_Fermi,&
        beta,mu,.FALSE.,Fermi_precision)
    allocate(coef_F(n_cheb))
    do i=1, n_cheb
        coef_F(i) = coef(i)
    end do
    
    ! get one minus Fermi cheb coefficients
    coef = 0.0d0
    call get_Fermi_cheb_coef(coef,n_cheb,&
        nr_Fermi,beta,mu,.TRUE.,Fermi_precision)
    allocate(coef_omF(n_cheb))
    do i=1, n_cheb
        coef_omF(i) = coef(i)
    end do
    deallocate(coef)
    
    open(unit=27,file=output_filename)
    write(27,*) "Number of qpoints =", n_q_points
    write(27,*) "Number of samples =", n_ran_samples
    write(27,*) "Number of timesteps =", n_timestep
    
    print*, "Calculating dynamical polarization correlation function."
                
    !loop over n qpoints
    do i_q = 1, n_q_points
        print*, "  q-point ", i_q, " of ", n_q_points
        write(27,"(A9, ES24.14E3,ES24.14E3,ES24.14E3)") "qpoint  = ", &
             q_points(i_q,1), q_points(i_q,2), q_points(i_q,3)
        
        !calculate the coefficients for the density operator
        !exp(i* q dot r)
        call density_coef(n_wf, s_site_x, s_site_y, s_site_z, &
            q_points(i_q,:), s_density_q, s_density_min_q)

        ! Average over (n_ran_samples) samples
        do i_sample=1, n_ran_samples
            print*, "    Sample ", i_sample, " of ", n_ran_samples
            write(27,*) "Sample =", i_sample
            
            ! make random state and psi1, psi2
            call random_state(wf0, n_wf, seed*i_sample)
            ! call density(-q)*wf0, resulting in psi1
            call density(wf0, n_wf, s_density_min_q, psi1)
            ! call fermi with 1-fermi coefficients for psi1
            call Fermi(psi1, n_wf, coef_omF, n_cheb, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, psi1)
            ! call fermi with fermi coefficients for psi2
            call Fermi(wf0, n_wf, coef_F, n_cheb, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, psi2)
            ! call density(q)*psi1, resulting in wf1
            call density(psi1, n_wf, s_density_q, wf1)
                
            ! get correlation and store
            corrval = aimag(inner_prod(psi2, wf1, n_wf))
            write(27,*) 1, corrval
            corr(i_q, 1) = corr(i_q, 1) + corrval / n_ran_samples
            
            ! iterate over tau
            do k = 2, n_timestep
            
                if (MODULO(k,64) == 0) then
                    print*, "      Timestep ", k, " of ", n_timestep
                end if
            
                ! call time and density operators
                call cheb_wf_timestep(psi1, n_wf, Bes, n_Bes, &
                    s_indptr, n_indptr, s_indices, n_indices, &
                    s_hop, n_hop, psi1)
                call cheb_wf_timestep(psi2, n_wf, Bes, n_Bes, &
                    s_indptr, n_indptr, s_indices, n_indices, &
                    s_hop, n_hop, psi2)
                call density(psi1, n_wf, s_density_q, wf1)
                    
                ! get correlation and store
                corrval = aimag(inner_prod(psi2, wf1, n_wf))
                write(27,*) k, corrval
                corr(i_q, k) = corr(i_q, k) + corrval / n_ran_samples
                
            end do
            
        end do
    end do
    
    close(27)
    
end subroutine tbpm_dyn_pol
    
! Get DC conductivity
subroutine tbpm_dccond(Bes, n_bes, beta, mu, &
    s_indptr, n_indptr, s_indices, n_indices, s_hop, n_hop, H_rescale, &
    s_dx, n_dx, s_dy, n_dy, seed, n_timestep, n_ran_samples, &
    t_step, energies, n_energies, en_inds, n_en_inds, &
    output_filename_dos, output_filename_dc, dos_corr, dc_corr)
    
    ! prepare
    use tbpm_mod
    implicit none
    
    ! deal with input
    integer, intent(in) :: n_bes, n_indptr, n_indices, n_hop, n_dx, n_dy
    integer, intent(in) :: n_timestep, n_ran_samples, seed, n_energies
    integer, intent(in) :: n_en_inds
    real(8), intent(in) :: H_rescale, beta, mu, t_step
    real(8), intent(in), dimension(n_Bes) :: Bes
    integer(8), intent(in), dimension(n_indptr) :: s_indptr
    integer(8), intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop
    real(8), intent(in), dimension(n_dx) :: s_dx
    real(8), intent(in), dimension(n_dy) :: s_dy
    real(8), intent(in), dimension(n_energies) :: energies
    integer(8), intent(in), dimension(n_en_inds) :: en_inds
    character*(*), intent(in) :: output_filename_dos
    character*(*), intent(in) :: output_filename_dc
    
    ! declare vars
    integer :: i_sample,  i, j, k, l, t, n_wf
    real(8) :: W, QE_sum, en
    complex(8), dimension(n_indptr - 1) :: wf0,wf_t_pos,wf_t_neg,wfE
    complex(8), dimension(n_en_inds, n_indptr - 1) :: wf_QE
    complex(8), dimension(2,n_indptr - 1) :: wf0_J,wfE_J,wfE_J_t
    real(8), allocatable :: coef_F(:) ! cheb coefs for Fermi operator
    real(8), allocatable :: coef_omF(:) ! cheb coefs one minus Fermi operator
    complex(8), dimension(n_hop) :: sys_current_x ! coefs for x current
    complex(8), dimension(n_hop) :: sys_current_y ! coefs for y current
    complex(8) :: dos_corrval
    complex(8), dimension(2) :: dc_corrval
    
    ! output
    complex(8), intent(out), dimension(n_timestep) :: dos_corr
    complex(8), intent(out), dimension(2, n_energies, n_timestep) :: dc_corr
    ! elements dc_corr_x and dc_corr_y
    
    ! set some values
    n_wf = n_indptr - 1
    dos_corr = 0.0d0
    dc_corr = 0.0d0
    
    ! prepare output files
    open(unit=27,file=output_filename_dos)
    write(27,*) "Number of samples =", n_ran_samples
    write(27,*) "Number of timesteps =", n_timestep
    
    open(unit=28,file=output_filename_dc)
    write(28,*) "Number of samples =", n_ran_samples
    write(28,*) "Number of energies =", n_en_inds
    write(28,*) "Number of timesteps =", n_timestep
    
    ! get current coefficientsconfig.generic['nr_random_samples']
    call current_coefficient(s_hop, s_dx, n_hop, sys_current_x)
    call current_coefficient(s_hop, s_dy, n_hop, sys_current_y)
    sys_current_x = H_rescale * sys_current_x
    sys_current_y = H_rescale * sys_current_y
    
    print*, "Calculating DC conductivity correlation function."
        
    ! Average over (n_ran_samples) samples
    do i_sample=1, n_ran_samples
        
        print*, "  Calculating for sample ", i_sample, " of ", n_ran_samples
        write(27,*) "Sample =", i_sample
        write(28,*) "Sample =", i_sample
  
        ! make random state
        call random_state(wf0, n_wf, seed*i_sample)
        
        ! ------------
        ! first, get DOS and quasi-eigenstates
        ! ------------

        ! initial values for wf_t and wf_QE
        do i = 1, n_wf
            wf_t_pos(i) = wf0(i)
            wf_t_neg(i) = wf0(i)
        end do
        do i = 1, n_en_inds
            do j = 1, n_wf
                wf_QE(i, j) = wf0(j)
            end do
        end do

        ! Iterate over time, get Fourier transform
        do k = 1, n_timestep

            if (MODULO(k,64) == 0) then
                print*, "    Getting DOS/QE for timestep ", k, " of ", n_timestep
            end if

            ! time evolution
            call cheb_wf_timestep(wf_t_pos, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, wf_t_pos)
            call cheb_wf_timestep(wf_t_neg, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                -s_hop, n_hop, wf_t_neg)
            
            ! get dos correlation
            dos_corrval = inner_prod(wf0, wf_t_pos, n_wf)
            dos_corr(k) = dos_corr(k) + dos_corrval/n_ran_samples
            write(27,*) k, real(dos_corrval), aimag(dos_corrval)

            W = 0.5*(1+cos(pi*k/n_timestep)) ! Hanning window

            !$OMP parallel do private (j)
            do i = 1, n_en_inds
            
                en = energies(en_inds(i))
                
                do j = 1, n_wf
                    wf_QE(i,j) = wf_QE(i,j)+&
                        exp(img*en*k*t_step)*wf_t_pos(j)*W
                    wf_QE(i,j) = wf_QE(i,j)+&
                        exp(-img*en*k*t_step)*wf_t_neg(j)*W
                end do
            end do
            !$OMP end parallel do

        end do

        ! Normalise
        do i = 1, n_en_inds
            QE_sum = 0
            do j = 1, n_wf
                QE_sum = QE_sum + abs(wf_QE(i, j))**2
            end do
            do j = 1, n_wf
                wf_QE(i, j) = wf_QE(i, j)/sqrt(QE_sum)
            end do
        end do
        
        ! ------------
        ! then, get dc conductivity
        ! ------------
        
        ! iterate over energies
        do i = 1, n_en_inds
        
            ! some output
            if (MODULO(i,8) == 0) then
                print*, "    Getting DC conductivity for energy: ", i, " of ", n_en_inds
            end if
            write(28,*) "Energy ", i, en_inds(i), energies(en_inds(i))
        
            ! get corresponding quasi-eigenstate
            wfE(:) = wf_QE(i,:)/abs(inner_prod(wf0(:),wf_QE(i,:), n_wf))
        
            ! make psi1, psi2
            call current(wf0, n_wf, s_indptr, n_indptr, s_indices, &
                n_indices, sys_current_y, n_hop, wf0_J(1,:))
            call current(wf0, n_wf, s_indptr, n_indptr, s_indices, &
                n_indices, sys_current_y, n_hop, wf0_J(2,:))
            call current(wfE, n_wf, s_indptr, n_indptr, s_indices, &
                n_indices, sys_current_y, n_hop, wfE_J(1,:))
            call current(wfE, n_wf, s_indptr, n_indptr, s_indices, &
                n_indices, sys_current_y, n_hop, wfE_J(2,:))
            
            ! get correlation values
            dc_corrval(1) = inner_prod(wf0_J(1,:),wfE_J(1,:), n_wf)
            dc_corrval(2) = inner_prod(wf0_J(2,:),wfE_J(2,:), n_wf)

            ! write to file
            write(28,"(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)")&
                        1, &
                        real(dc_corrval(1)), aimag(dc_corrval(1)), &
                        real(dc_corrval(2)), aimag(dc_corrval(2))

            ! update correlation functions
            dc_corr(1,i,1)=dc_corr(1,i,1)+dc_corrval(1)/n_ran_samples
            dc_corr(2,i,1)=dc_corr(2,i,1)+dc_corrval(2)/n_ran_samples
                
            ! iterate over time
            do k = 2, n_timestep
                ! NEGATIVE time evolution of QE state
                call cheb_wf_timestep(wfE_J(1,:), n_wf, Bes, n_Bes, &
                    s_indptr, n_indptr, s_indices, n_indices, &
                    -s_hop, n_hop, wfE_J(1,:))
                call cheb_wf_timestep(wfE_J(2,:), n_wf, Bes, n_Bes, &
                    s_indptr, n_indptr, s_indices, n_indices, &
                    -s_hop, n_hop, wfE_J(2,:))
                
                ! get correlation values
                dc_corrval(1) = inner_prod(wf0_J(1,:),wfE_J(1,:), n_wf)
                dc_corrval(2) = inner_prod(wf0_J(2,:),wfE_J(2,:), n_wf)
                  
                ! write to file
                write(28,"(I7,ES24.14E3,ES24.14E3,ES24.14E3,ES24.14E3)")&
                            k, &
                            real(dc_corrval(1)), aimag(dc_corrval(1)), &
                            real(dc_corrval(2)), aimag(dc_corrval(2))
                        
                ! update correlation functions
                dc_corr(1,i,k)=dc_corr(1,i,k)+dc_corrval(1)/n_ran_samples
                dc_corr(2,i,k)=dc_corr(2,i,k)+dc_corrval(2)/n_ran_samples
            end do
            
        end do
    
    end do
    
    close(27)
    close(28)

end subroutine tbpm_dccond
    
! Get quasi-eigenstates
subroutine tbpm_eigenstates(Bes, n_Bes, &
    s_indptr, n_indptr, s_indices, n_indices, s_hop, n_hop, &
    seed, n_timestep, n_ran_samples, t_step, energies, n_energies, &
    wf_QE)
    
    ! prepare
    use tbpm_mod
    implicit none
    
    ! deal with input
    integer, intent(in) :: n_Bes, n_indptr, n_indices, n_hop
    integer, intent(in) :: n_timestep, seed, n_energies, n_ran_samples
    real(8), intent(in), dimension(n_Bes) :: Bes
    real(8), intent(in) :: t_step
    integer(8), intent(in), dimension(n_indptr) :: s_indptr
    integer(8), intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop
    real(8), intent(in), dimension(n_energies) :: energies
    
    ! declare vars
    integer :: i, j, k, l, t, n_wf, i_sample
    real(8) :: W, QE_sum
    complex(8), dimension(n_indptr - 1) :: wf0,wf_t_pos,wf_t_neg
    complex(8), dimension(n_energies,n_indptr-1) :: wfq
    
    ! output
    complex(8),intent(out),dimension(n_energies,n_indptr-1)::wf_QE
    n_wf = n_indptr - 1

    wf_QE = 0.0d0
    
    print*, "Calculating quasi-eigenstates."
    
    ! Average over (n_ran_samples) samples
    do i_sample=1, n_ran_samples
    
        print*, "  Calculating for sample ", i_sample, " of ", n_ran_samples
        
        ! make random state
        call random_state(wf0, n_wf, seed*i_sample)

        ! initial values for wf_t and wf_QE
        do i = 1, n_wf
            wf_t_pos(i) = wf0(i)
            wf_t_neg(i) = wf0(i)
        end do
        do i = 1, n_energies
            do j = 1, n_wf
                wfq(i, j) = wf0(j)
            end do
        end do

        ! Iterate over time, get Fourier transform
        do k = 1, n_timestep

            if (MODULO(k,64) == 0) then
                print*, "    Timestep ", k, " of ", n_timestep
            end if

            call cheb_wf_timestep(wf_t_pos, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                s_hop, n_hop, wf_t_pos)
            call cheb_wf_timestep(wf_t_neg, n_wf, Bes, n_Bes, &
                s_indptr, n_indptr, s_indices, n_indices, &
                -s_hop, n_hop, wf_t_neg)

            W = 0.5*(1+cos(pi*k/n_timestep)) ! Hanning window

            !$OMP parallel do private (j)
            do i = 1, n_energies
                do j = 1, n_wf
                    wfq(i,j) = wfq(i,j)+&
                        exp(img*energies(i)*k*t_step)*wf_t_pos(j)*W
                    wfq(i,j) = wfq(i,j)+&
                        exp(-img*energies(i)*k*t_step)*wf_t_neg(j)*W
                end do
            end do
            !$OMP end parallel do

        end do

        ! Normalise
        do i = 1, n_energies
            QE_sum = 0
            do j = 1, n_wf
                QE_sum = QE_sum + abs(wfq(i, j))**2
            end do
            do j = 1, n_wf
                wfq(i, j) = wfq(i, j)/sqrt(QE_sum)
            end do
        end do

        wf_QE(:,:) = wf_QE(:,:) + wfq(:,:) / n_ran_samples

    end do

end subroutine tbpm_eigenstates
