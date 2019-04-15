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


! !read in mu_mn from file and recalculate the conductivity for different
! !n_energies, beta, etc. Basically a wrapper for cond_from_trace( )
! SUBROUTINE cond_from_file(mu_mn_file, energies, n_energies, &
!     NE_integral, beta, fermi_precision, prefactor, &
!     cond)
!
!     USE tbpm_mod
!     IMPLICIT NONE
!
!     CHARACTER*(*), INTENT(in) :: mu_mn_file
!
!     INTEGER, INTENT(in) :: NE_integral, n_energies
!     REAL(8), INTENT(in) :: beta, fermi_precision, prefactor
!     REAL(8), INTENT(in), DIMENSION(n_energies) :: energies
!
!
!     REAL(8), intent(out), DIMENSION(n_energies) :: cond
!     CHARACTER(len=1024):: line,word
!
!     COMPLEX(8), ALLOCATABLE :: mu_mn(:,:)
!     REAL(8) :: H_rescale
!     INTEGER :: n_kernel, ios, ns,n_ran_samples
!     INTEGER :: i,j,k
! !~     REAL(8), ALLOCATABLE:: x(:),y(:)
!     REAL(8) :: x,y,a
!     COMPLEX(8), ALLOCATABLE :: z(:)
!     ns=1
!
! !~     allocate( x(1:ns),y(1:ns))
!
!     cond(:) = 0.d0
!
!     !the next part does not work so well
!
!     PRINT*,"Reading File"
!
!     open(unit=15,file=mu_mn_file,status='OLD',iostat=ios)
!     if(ios.ne.0) stop 'Failed to open fort.15'
!
!     ios=0
!     read(15,'(a)',iostat=ios) line
!     if(ios==0) then
!         line=adjustl(line)
!         if(index(line(1:1),"!").eq.0.and.index(line(1:1),"#").eq.0) then ! skip comment
!         word='H_rescale='
!         if(index(line,trim(word)).ne.0) then
!         read(line(index(line,trim(word))+len_trim(word):),'(a)',iostat=ios) word
!         if(ios.ne.0) then
!         stop ' I/O error during input: (H_rescale=)'
!         endif
!     word=adjustl(word)
!     read(word,*,iostat=ios) H_rescale
!         endif
!         endif
!         endif
!
!     read(15,'(a)',iostat=ios) line
!         if(ios==0) then
!         line=adjustl(line)
!         if(index(line(1:1),"!").eq.0.and.index(line(1:1),"#").eq.0) then ! skip comment
!         word='N_kernel='
!         if(index(line,trim(word)).ne.0) then
!         read(line(index(line,trim(word))+len_trim(word):),'(a)',iostat=ios) word
!         if(ios.ne.0) then
!         stop ' I/O error during input: (N_kernel=)'
!         endif
!     word=adjustl(word)
!     read(word,*,iostat=ios) n_kernel
!         endif
!         endif
!         endif
!
!     read(15,'(a)',iostat=ios) line
!         if(ios==0) then
!         line=adjustl(line)
!         if(index(line(1:1),"!").eq.0.and.index(line(1:1),"#").eq.0) then ! skip comment
!         word='n_ran_samples='
!         if(index(line,trim(word)).ne.0) then
!         read(line(index(line,trim(word))+len_trim(word):),'(a)',iostat=ios) word
!         if(ios.ne.0) then
!         stop ' I/O error during input: (n_ran_samples=)'
!         endif
!     word=adjustl(word)
!     read(word,*,iostat=ios) n_ran_samples
!         endif
!         endif
!         endif
!
!     PRINT*,n_kernel
!     PRINT*,H_rescale
!
!     n_kernel = n_kernel - 1 !fix to get indices right
!     ALLOCATE(mu_mn(0:n_kernel-1,0:n_kernel-1))
!     ALLOCATE(z(0:n_kernel-1))
!
!     do j=0, n_kernel-1
!         read(15,*,iostat=ios) z
! !~         PRINT*,z
!         mu_mn(j,:) = z
!     end do
!
! !~     do i=0, n_kernel-1
! !~         do j=0, n_kernel-1
! !~             PRINT*, mu_mn(i,j)
! !~         end do
! !~     end do
!
!     call cond_from_trace(mu_mn, n_kernel, n_kernel, energies, n_energies, &
!             NE_integral, H_rescale, beta, fermi_precision, prefactor, &
!             cond)
!
! END SUBROUTINE cond_from_file

!calculates everything after the calculation of the trace
SUBROUTINE cond_from_trace(mu_mn, n_kernel, energies, n_energies, &
    NE_integral, H_rescale, beta, fermi_precision, prefactor, &
    cond)

    USE kpm
	USE funcs, ONLY: Fermi_dist
    IMPLICIT NONE

    INTEGER, INTENT(in) :: n_kernel, NE_integral, n_energies
    REAL(8), INTENT(in) :: H_rescale, beta, fermi_precision, prefactor
    REAL(8), INTENT(in), DIMENSION(n_energies) :: energies
    COMPLEX(8), INTENT(in), DIMENSION(0:n_kernel-1,0:n_kernel-1) :: mu_mn
    COMPLEX(8), DIMENSION(0:n_kernel-1,0:n_kernel-1) :: mu_mn_total

    REAL(8), intent(out), DIMENSION(n_energies) :: cond
    REAL(8) :: g_m(0:n_kernel-1), cheb_x(0:n_kernel-1)
	REAL(8) :: energy_integral(-NE_integral:NE_integral)
    COMPLEX(8) :: Gamma_mn(0:n_kernel-1,0:n_kernel-1)

    COMPLEX(8) :: sum_gamma_mu(-NE_integral:NE_integral)
    COMPLEX(8) :: dcx

    REAL(8) :: E_step,energy,cur_mu,a,fd
    INTEGER :: i,j,k, NE

    cond(:) = 0.d0

    PRINT*,"Calculating Jackson Kernel"
    call jackson_kernel(g_m,n_kernel)

    PRINT*,"!recalculate mu to include kernels"
    !recalculate mu to include kernels
    do i=0, n_kernel-1
        do j=0, n_kernel-1

            mu_mn_total(i,j) = g_m(i)*g_m(j)*mu_mn(i,j)
            !PRINT*, i, j, mu_mn(i,j), mu_mn_total(i,j)
!~             PRINT*, i, j, g_m(i), g_m(j)
            if (i==0) then
                mu_mn_total(i,j) = mu_mn_total(i,j)/2.0
!~                 PRINT*,'test1', mu_mn_total(i,j)
            end if
            if (j==0) then
                mu_mn_total(i,j) = mu_mn_total(i,j)/2.0
!~                 PRINT*,'test2,', mu_mn_total(i,j)
            end if
        end do
    end do

!~     open (21,file='mu_mn_new_fort.out')
!~     do j =0, n_kernel-1
!~         do i =0, n_kernel-1
!~             write(21,*) i,j, mu_mn_total(i,j)
!~
!~         end do
!~     end do
!~     close(21)

    !calculate all the energies for which we have to take the integral
    E_step = 1.d0/NE_integral
    PRINT*,NE_integral,E_step
    NE = NE_integral-1


    do k=-NE_integral,NE_integral
        energy_integral(k) = k*E_step
!~         PRINT*, k,energy_integral(k)
!~         PRINT*,k
    end do

    sum_gamma_mu = 0.d0

    PRINT*,"Calculate sum"
!~     open (17,file='cheb_mega_new.out')
    !let's calculate everything for different energies (between -1 and 1):
    do k=-NE,NE
        if (MODULO(k,256) == 0) then
            PRINT*, (k + NE + 1) / 2
        end if
        call chebyshev_polynomial(energy_integral(k),n_kernel,cheb_x)



!~         PRINT*,cheb_x
        call calculate_gamma_mn(Gamma_mn,energy_integral(k),n_kernel,cheb_x)
!~
!~         do i=0, n_kernel-1
!~             write(17,*) i, k, energy_integral(k), Gamma_mn(i,n_kernel)
!~         end do

        do j=0, n_kernel-1
            do i=0, n_kernel-1
                sum_gamma_mu(k) = sum_gamma_mu(k) + Gamma_mn(i,j)*mu_mn_total(i,j)
            end do
        end do


    end do
!~     close(17)

!~     PRINT*,energies(0),energies(1),energies(n_energies)
!~     PRINT*,n_energies
    PRINT*,"Final integral"
    do i=1,n_energies
        cur_mu = energies(i)
!~         PRINT*,cur_mu
        dcx=0.d0
        do k=-NE,NE
!~             PRINT*, k, energy_integral(k)
            energy=energy_integral(k)
            a = 1.d0-energy*energy
            fd = Fermi_dist(beta,cur_mu,energy*H_rescale,fermi_precision)
!~             fd = Fermi_dist(beta,cur_mu/H_rescale,energy,fermi_precision)
            a = fd/a/a
            dcx = dcx + dble(sum_gamma_mu(k))*a
        end do

!~         cond(i) = dcx*prefactor*E_step*E_step/H_rescale/H_rescale
        cond(i) = dcx*prefactor*E_step/H_rescale/H_rescale
    end do

    PRINT*, prefactor, E_step, H_rescale

CONTAINS

	!!! SUBROUTINE to calculate Gamma_mn
	SUBROUTINE calculate_gamma_mn(Gamma_mn,x,n_kernel,ChebyshevPolynomial)

	    implicit none
	    INTEGER:: i,j,k,n_kernel
	    REAL(kind=8):: x,acosx,a,b,c,ChebyshevPolynomial(0:n_kernel-1)
	    COMPLEX(kind=8):: Gamma_mn(0:n_kernel-1,0:n_kernel-1),ca,cb,cc,cd

	    acosx=acos(x)
	    c=dsqrt(1.-x*x)

	!$OMP parallel do simd private(ca,cb,cc,cd,i)
	    do j=0, n_kernel-1
	        ca=dcmplx(cos(j*acosx),sin(j*acosx))
	        cc=dcmplx(x,-j*c)

	        do i=0, n_kernel-1
	            cb=dcmplx(cos(i*acosx),-sin(i*acosx))
	            cd=dcmplx(x,i*c)

	            Gamma_mn(i,j)=ca*cc*ChebyshevPolynomial(i)+ &
							  cb*cd*ChebyshevPolynomial(j)
	!~             write(*,*) Gamma_mn(i,j)
	        end do
	    end do
	!$OMP end parallel do simd

	END SUBROUTINE calculate_gamma_mn


END SUBROUTINE cond_from_trace
