! ------------------------------------------
! MODULE with helper functions for tbpm_f2py
! ------------------------------------------

module tbpm_mod

    implicit none
    real(8), parameter :: pi=3.141592653589793238460d0
    complex(8), parameter :: img = cmplx(0.0d0, 1.0d0)

contains

! Scalar product
complex(8) function inner_prod(A, B, N)

    implicit none
    integer, intent(in) :: N
    complex(8), intent(in), dimension(N) :: A, B
    complex(8), external :: zdotc

    ! use BLAS to calculate the inner product
    inner_prod = zdotc(N, A, 1, B, 1)

end function inner_prod

! FFTW interface
subroutine fft(x, n_x, sgn)

    implicit none
    integer, intent(in) :: sgn, n_x
    complex(8), dimension(n_x), intent(inout) :: x
    integer(8) :: plan

    call dfftw_plan_dft_1d(plan, n_x, x, x, sgn, 64)
    call dfftw_execute_dft(plan, x, x)
    call dfftw_destroy_plan(plan)

end subroutine fft

! Hamiltonian operator
subroutine Hamiltonian(wf_in, n_wf, s_indptr, &
    n_indptr, s_indices, n_indices, s_hop, n_hop, wf_out)

    ! deal with input
    implicit none
    integer, intent(in) :: n_wf, n_indptr, n_indices, n_hop
    complex(8), intent(in), dimension(n_wf) :: wf_in
    integer, intent(in), dimension(n_indptr) :: s_indptr
    integer, intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop

    ! output
    complex(8), intent(out), dimension(n_wf) :: wf_out

    !use Sparse BLAS in MKL to calculate the production
    call mkl_cspblas_zcsrgemv('N', n_wf, s_hop, s_indptr, s_indices, &
        wf_in, wf_out)

end subroutine Hamiltonian

! Apply timestep using Chebyshev decomposition
subroutine cheb_wf_timestep(wf_t, n_wf, Bes, n_Bes, &
    s_indptr, n_indptr, s_indices, n_indices, s_hop, n_hop, wf_t1)

    ! deal with input
    implicit none
    integer, intent(in) :: n_wf, n_Bes, n_indptr, n_indices, n_hop
    complex(8), intent(in), dimension(n_wf), target :: wf_t
    real(8), intent(in), dimension(n_Bes) :: Bes
    integer, intent(in), dimension(n_indptr) :: s_indptr
    integer, intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop

    ! declare vars
    integer :: i, k
    complex(8), dimension(n_wf), target :: Tcheb0, Tcheb1, Tcheb2
    complex(8), dimension(:), pointer :: p0, p1, p2

    ! output
    complex(8), intent(out), dimension(n_wf) :: wf_t1

    call Hamiltonian(wf_t, n_wf, s_indptr, &
        n_indptr, s_indices, n_indices, s_hop, n_hop, Tcheb1)

    !$OMP parallel do
    do i = 1, n_wf
        Tcheb0(i) = wf_t(i)
        Tcheb1(i) = -img * Tcheb1(i)
        wf_t1(i) = Bes(1) * wf_t(i) + 2  * Bes(2) * Tcheb1(i)
    end do
    !$OMP end parallel do

    p0 => Tcheb0
    p1 => Tcheb1
    do k=3, n_Bes
        p2 => p0
        call Hamiltonian(p1, n_wf, s_indptr, &
            n_indptr, s_indices, n_indices, s_hop, n_hop, Tcheb2)

        !$OMP parallel do
        do i = 1, n_wf
            p2(i) = p0(i) - 2 * img * Tcheb2(i)
            wf_t1(i) = wf_t1(i) + 2 * Bes(k) * p2(i)
        end do
        !$OMP end parallel do
        p0 => p1
        p1 => p2
    end do

end subroutine cheb_wf_timestep

! get coefficients of current operator
subroutine current_coefficient(hop, dr, n_hop, cur_coefs)

    ! deal with input
    implicit none
    integer, intent(in) :: n_hop
    complex(8), intent(in), dimension(n_hop) :: hop
    real(8), intent(in), dimension(n_hop) :: dr
    complex(8), intent(out), dimension(n_hop) :: cur_coefs

    ! declare vars
    integer :: i

    !$OMP parallel do
    do i = 1, n_hop
        cur_coefs(i) = img * hop(i) * dr(i)
    end do
    !$OMP end parallel do

end subroutine current_coefficient

! current operator
subroutine current(wf_in, n_wf, s_indptr, n_indptr, s_indices, &
    n_indices, cur_coefs, n_cur_coefs, wf_out)

    ! deal with input
    implicit none
    integer, intent(in) :: n_wf, n_indptr, n_indices, n_cur_coefs
    complex(8), intent(in), dimension(n_wf) :: wf_in
    integer, intent(in), dimension(n_indptr) :: s_indptr
    integer, intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_cur_coefs) :: cur_coefs

    ! output
    complex(8), intent(out), dimension(n_wf) :: wf_out

    !use Sparse BLAS in MKL to calculate the production
    call mkl_cspblas_zcsrgemv('N', n_wf, cur_coefs, s_indptr, s_indices, &
        wf_in, wf_out)

end subroutine current

! The actual Fermi distribution
real(8) function Fermi_dist(beta,Ef,energy,eps)

    implicit none
    real(8) :: eps, beta, Ef, energy, x

    if (energy >= Ef) then
        x = exp(beta * (Ef - energy))
        Fermi_dist = x / (1 + x)
    else
        x = exp(beta * (energy - Ef))
        Fermi_dist = 1 / (1 + x)
    end if

    if (Fermi_dist < eps) then
        Fermi_dist = 0
    end if

end function

! compute Chebyshev expansion coefficients of Fermi operator
subroutine get_Fermi_cheb_coef(cheb_coef, n_cheb, &
    nr_Fermi, beta, mu, one_minus_Fermi, eps)

    ! declarations
    implicit none
    integer, intent(in) :: nr_Fermi
    logical, intent(in) :: one_minus_Fermi ! if true: compute coeffs for
                                           ! one minus Fermi operator
    real(8), intent(in) :: beta, mu, eps
    complex(8), dimension(nr_Fermi) :: cheb_coef_complex
    real(8), intent(out), dimension(nr_Fermi) :: cheb_coef
    integer, intent(out) :: n_cheb
    real(8) :: r0, compare, x, prec, energy
    integer :: i
    r0 = 2 * pi / nr_Fermi

    if (one_minus_Fermi) then ! compute coeffs for one minus Fermi operator
        do i = 1, nr_Fermi
            energy = cos((i - 1) * r0)
            cheb_coef_complex(i) = 1. - Fermi_dist(beta,mu,energy,eps)
        end do
    else ! compute coeffs for Fermi operator
        do i=1, nr_Fermi
            energy = cos((i - 1) * r0)
            cheb_coef_complex(i) = Fermi_dist(beta,mu,energy,eps)
        end do
    end if

    ! Fourier transform result
    call fft(cheb_coef_complex, nr_Fermi, -1)

    ! Get number of nonzero elements
    prec = -log10(eps)
    n_cheb = 0
    cheb_coef_complex = 2. * cheb_coef_complex / nr_Fermi
    cheb_coef_complex(1) = cheb_coef_complex(1) / 2
    compare = log10(maxval(abs(cheb_coef_complex(1:nr_Fermi))))-prec
    cheb_coef = real(cheb_coef_complex)
    do i = 1, nr_Fermi
        if((log10(abs(cheb_coef(i)))<compare).and.&
            (log10(abs(cheb_coef(i+1)))<compare)) then
            n_cheb = i
            exit
        end if
    end do
    if (n_cheb==0) then
        print *,"WARNING: not enough Fermi operator Cheb. coeficcients"
    end if

end subroutine get_Fermi_cheb_coef

! Fermi-Dirac distribution operator
subroutine Fermi(wf_in, n_wf, cheb_coef, n_cheb, &
    s_indptr, n_indptr, s_indices, n_indices, s_hop, n_hop, wf_out)

    ! deal with input
    implicit none
    integer, intent(in) :: n_wf, n_cheb, n_indptr, n_indices, n_hop
    complex(8), intent(in), dimension(n_wf) :: wf_in
    real(8), intent(in), dimension(n_cheb) :: cheb_coef
    integer, intent(in), dimension(n_indptr) :: s_indptr
    integer, intent(in), dimension(n_indices) :: s_indices
    complex(8), intent(in), dimension(n_hop) :: s_hop

    ! declare vars
    integer :: i, j, k, l
    real(8) :: sum_wf
    complex(8), dimension(n_wf), target :: Tcheb0, Tcheb1, Tcheb2
    complex(8), dimension(:), pointer :: p0, p1, p2

    ! output
    complex(8), intent(out), dimension(n_wf) :: wf_out

    call Hamiltonian(wf_in, n_wf, s_indptr, &
        n_indptr, s_indices, n_indices, s_hop, n_hop, Tcheb1)

    !$OMP parallel do
    do i = 1, n_wf
        Tcheb0(i) = wf_in(i)
        wf_out(i) = cheb_coef(1) * Tcheb0(i) + cheb_coef(2) * Tcheb1(i)
    end do
    !$OMP end parallel do

    p0 => Tcheb0
    p1 => Tcheb1
    do k=3, n_cheb
        p2 => p0
        call Hamiltonian(Tcheb1, n_wf, s_indptr, &
            n_indptr, s_indices, n_indices, s_hop, n_hop, Tcheb2)

        !$OMP parallel do
        do i = 1, n_wf
            p2(i) = 2 * Tcheb2(i) - p0(i)
            wf_out(i) = wf_out(i) + cheb_coef(k) * p2(i)
        end do
        !$OMP end parallel do
        p0 => p1
        p1 => p2
    end do

end subroutine fermi

subroutine density_coef(n_wf, site_x, site_y, &
    site_z, q_point, s_density_q, s_density_min_q)

    ! deal with input
    implicit none
    integer, intent(in) :: n_wf
    real(8), intent(in), dimension(n_wf) :: site_x, site_y, site_z
    real(8), intent(in), dimension(3) :: q_point
    complex(8),intent(out),dimension(n_wf)::s_density_q
    complex(8),intent(out),dimension(n_wf)::s_density_min_q

    ! declare vars
    integer :: i,j
    real(8) :: power

    !$OMP parallel do private (power)
    do i = 1, n_wf
        power = q_point(1)*site_x(i) + &
                q_point(2)*site_y(i) + &
                q_point(3)*site_z(i)
        s_density_q(i) = cos(power) + img*sin(power)
        s_density_min_q(i) = cos(power) - img*sin(power)
    end do
    !$OMP end parallel do

end subroutine density_coef

! density operator
subroutine density(wf_in, n_wf, s_density, wf_out)

    ! deal with input
    implicit none
    integer, intent(in) :: n_wf
    complex(8), intent(in), dimension(n_wf) :: s_density
    complex(8), intent(in), dimension(n_wf) :: wf_in

    ! output
    complex(8), intent(out), dimension(n_wf) :: wf_out

    ! declare vars
    integer :: i

    !$OMP parallel do
    do i = 1, n_wf
        wf_out(i) = s_density(i) * wf_in(i)
    end do
    !$OMP end parallel do

end subroutine density

! Make random initial state
subroutine random_state(wf, n_wf, iseed)

    ! variables
    implicit none
    integer, intent(in) :: n_wf, iseed
    complex(8), intent(out), dimension(n_wf) :: wf
    integer :: i, iseed0
    real(8) :: f, g, wf_sum, abs_z_sq
    complex(8) :: z

    ! make random wf
    iseed0=iseed*49741

    f=ranx(iseed0)
    wf_sum = 0
    do i = 1, n_wf
        f=ranx(0)
        g=ranx(0)
        abs_z_sq = -1.0d0 * log(1.0d0 - f) ! dirichlet distribution
        z = dsqrt(abs_z_sq)*exp(img*2*pi*g) ! give random phase
        wf(i) = z
        wf_sum = wf_sum + abs_z_sq
    end do
    do i = 1, n_wf
        wf(i) = wf(i)/dsqrt(wf_sum)
    end do

    contains

    ! random number
    function ranx(idum)
        integer :: idum, n
        integer, allocatable :: seed(:)
        real*8 :: ranx
        if (idum>0) then
            call random_seed(size=n)
            allocate(seed(n))
            ! is there a better way to create a seed array
            ! based on the input integer?
            do i=1, n
                seed(i)=int(modulo(i * idum * 74231, 104717))
            end do
            call random_seed(put=seed)
        end if
        call random_number(ranx)
    end function ranx

end subroutine random_state

end module tbpm_mod
