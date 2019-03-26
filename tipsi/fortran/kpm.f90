MODULE kpm
	IMPLICIT NONE

CONTAINS

! Get array of Jackson kernel
SUBROUTINE jackson_kernel(KernelFunction, n_kernel)
    USE const
    IMPLICIT NONE
    INTEGER:: k, n_kernel
    REAL(KIND=8):: a, b, q, KernelFunction(0 : n_kernel-1)

    KernelFunction(0) = 1D0
    q = PI / n_kernel
    b = DCOS(q) / DSIN(q)

	!$OMP PARALLEL DO SIMD PRIVATE(a)
    DO k = 1, n_kernel - 1
        a = (n_kernel - k) * dcos(q*k) + b * dsin(q*k)
        a = a / n_kernel
        KernelFunction(k) = a
    END DO
	!$OMP END PARALLEL DO SIMD

END SUBROUTINE jackson_kernel


SUBROUTINE chebyshev_polynomial(x, n_kernel,ChebyshevPolynomial)

    implicit none
    INTEGER:: i,j,k,n_kernel
    REAL(kind=8):: x,ChebyshevPolynomial(0:n_kernel)

    ChebyshevPolynomial(0)=1.d0
    ChebyshevPolynomial(1)=x

    do k=2,n_kernel
	    ChebyshevPolynomial(k)=2.d0*x*ChebyshevPolynomial(k-1)-ChebyshevPolynomial(k-2)
    end do

END SUBROUTINE chebyshev_polynomial


! SUBROUTINE to calculate the Chebyshev Polynomial T_m(H)
! using the recurrence relation, here H is a Hamiltonian matrix
SUBROUTINE get_ChebPol_wf(n_wf,wf0,wf1,wf2)

    INTEGER :: i,j, n_wf
    COMPLEX(8):: wf0(n_wf),wf1(n_wf),wf2(n_wf)

!$OMP parallel do
    do i=1,n_wf
        wf2(i)=2*wf2(i)-wf0(i)
        wf0(i)=wf1(i)
        wf1(i)=wf2(i)
    end do
!$OMP end parallel do

END SUBROUTINE get_ChebPol_wf


! SUBROUTINE to calculate the Chebyshev Polynomial T_m(H)
! using the recurrence relation, here H is a Hamiltonian matrix
SUBROUTINE get_ChebPol_n_wfthOrder(n_wf,wf0,wf1,wf2)

    INTEGER :: i,j,n_wf
    COMPLEX(8):: wf0(n_wf),wf1(n_wf),wf2(n_wf)

!$OMP parallel do
    do i=1,n_wf
        wf2(i)=2*wf2(i)-wf0(i)
    end do
!$OMP end parallel do

END SUBROUTINE get_ChebPol_n_wfthOrder

END MODULE kpm
