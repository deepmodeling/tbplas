! --------------------------------------------------
! Lindhard fortran subroutines, callable from python
! --------------------------------------------------

! calculate dynamic polarizability for regular q-point on k-mesh using
! Lindhard function
subroutine dyn_pol_q(eng, num_orb, num_kpt, wfn, kq_map, &
                     beta, mu, omegas, num_omega, delta, &
                     i_q, q_point, orb_pos, &
                     dyn_pol, num_qpt)
    implicit none

    ! input and output
    real(kind=8), intent(in) :: eng(num_orb, num_kpt)
    integer, intent(in) :: num_orb, num_kpt
    complex(kind=8), intent(in) :: wfn(num_orb, num_orb, num_kpt)
    integer, intent(in) :: kq_map(num_kpt)
    real(kind=8), intent(in) :: beta, mu
    real(kind=8), intent(in) :: omegas(num_omega)
    real(kind=8), intent(in) :: delta
    integer, intent(in) :: num_omega, i_q
    real(kind=8), intent(in) :: q_point(3)
    real(kind=8), intent(in) :: orb_pos(3, num_orb)
    complex(kind=8), intent(inout) :: dyn_pol(num_omega, num_qpt)
    integer, intent(in) :: num_qpt

    ! local variables
    integer :: i_w, i_k, i_kpq, jj, ll, ib
    real(kind=8) :: k_dot_r
    complex(kind=8) :: phase(num_orb)
    real(kind=8) :: omega, f, f_q
    complex(kind=8) :: prod, dp_sum
    real(kind=8) :: delta_eng(num_orb, num_orb, num_kpt)
    complex(kind=8) :: prod_df(num_orb, num_orb, num_kpt)
    complex(kind=8) :: cdelta

    ! build reusable arrays
    !$OMP PARALLEL DO PRIVATE(k_dot_r)
    do ib = 1, num_orb
        k_dot_r = q_point(1) * orb_pos(1, ib) &
                + q_point(2) * orb_pos(2, ib) &
                + q_point(3) * orb_pos(3, ib)
        phase(ib) = dcmplx(cos(k_dot_r), sin(k_dot_r))
    end do
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(i_kpq, jj, f, ll, f_q, prod, ib)
    do i_k = 1, num_kpt
        i_kpq = kq_map(i_k)
        do jj = 1, num_orb
            f = 1.0 / (1.0 + exp(beta * (eng(jj, i_k) - mu)))
            do ll = 1, num_orb
                delta_eng(ll, jj, i_k) = eng(jj, i_k) - eng(ll, i_kpq)
                f_q = 1.0 / (1.0 + exp(beta * (eng(ll, i_kpq) - mu)))
                prod = 0.0
                do ib = 1, num_orb
                    prod = prod + conjg(wfn(ib, ll, i_kpq)) * wfn(ib, jj, i_k) * phase(ib)
                end do
                prod_df(ll, jj, i_k) = prod * conjg(prod) * (f - f_q)
            end do
        end do
    end do
    !$OMP END PARALLEL DO

    ! calculate dyn_pol
    cdelta = dcmplx(0.0D0, delta)
    !$OMP PARALLEL DO PRIVATE(omega, dp_sum, i_k, jj, ll)
    do i_w = 1, num_omega
        omega = omegas(i_w)
        dp_sum = dcmplx(0.0D0, 0.0D0)
        do i_k = 1, num_kpt
            do jj = 1, num_orb
                do ll = 1, num_orb
                dp_sum = dp_sum + prod_df(ll, jj, i_k) &
                       / (delta_eng(ll, jj, i_k) + omega + cdelta)
                end do
            end do
        end do
        dyn_pol(i_w, i_q) = dp_sum
    end do
    !$OMP END PARALLEL DO
end subroutine dyn_pol_q


! calculate dynamic polarizability for arbitrary q-point using Lindhard function
subroutine dyn_pol_q_arb(eng, num_orb, num_kpt, wfn, eng_kq, wfn_kq, &
                         beta, mu, omegas, num_omega, delta, &
                         i_q, q_point, orb_pos, &
                         dyn_pol, num_qpt)
    implicit none

    ! input and output
    real(kind=8), intent(in) :: eng(num_orb, num_kpt)
    integer, intent(in) :: num_orb, num_kpt
    complex(kind=8), intent(in) :: wfn(num_orb, num_orb, num_kpt)
    real(kind=8), intent(in) :: eng_kq(num_orb, num_kpt)
    complex(kind=8), intent(in) :: wfn_kq(num_orb, num_orb, num_kpt)
    real(kind=8), intent(in) :: beta, mu
    real(kind=8), intent(in) :: omegas(num_omega)
    real(kind=8), intent(in) :: delta
    integer, intent(in) :: num_omega, i_q
    real(kind=8), intent(in) :: q_point(3)
    real(kind=8), intent(in) :: orb_pos(3, num_orb)
    complex(kind=8), intent(inout) :: dyn_pol(num_omega, num_qpt)
    integer, intent(in) :: num_qpt

    ! local variables
    integer :: i_w, i_k, jj, ll, ib
    real(kind=8) :: k_dot_r
    complex(kind=8) :: phase(num_orb)
    real(kind=8) :: omega, f, f_q
    complex(kind=8) :: prod, dp_sum
    real(kind=8) :: delta_eng(num_orb, num_orb, num_kpt)
    complex(kind=8) :: prod_df(num_orb, num_orb, num_kpt)
    complex(kind=8) :: cdelta

    ! build reusable arrays
    !$OMP PARALLEL DO PRIVATE(k_dot_r)
    do ib = 1, num_orb
        k_dot_r = q_point(1) * orb_pos(1, ib) &
                + q_point(2) * orb_pos(2, ib) &
                + q_point(3) * orb_pos(3, ib)
        phase(ib) = dcmplx(cos(k_dot_r), sin(k_dot_r))
    end do
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(jj, f, ll, f_q, prod, ib)
    do i_k = 1, num_kpt
        do jj = 1, num_orb
            f = 1.0 / (1.0 + exp(beta * (eng(jj, i_k) - mu)))
            do ll = 1, num_orb
                delta_eng(ll, jj, i_k) = eng(jj, i_k) - eng_kq(ll, i_k)
                f_q = 1.0 / (1.0 + exp(beta * (eng_kq(ll, i_k) - mu)))
                prod = 0.0
                do ib = 1, num_orb
                    prod = prod + conjg(wfn_kq(ib, ll, i_k)) * wfn(ib, jj, i_k) * phase(ib)
                end do
                prod_df(ll, jj, i_k) = prod * conjg(prod) * (f - f_q)
            end do
        end do
    end do
    !$OMP END PARALLEL DO

    ! calculate dyn_pol
    cdelta = dcmplx(0.0D0, delta)
    !$OMP PARALLEL DO PRIVATE(omega, dp_sum, i_k, jj, ll)
    do i_w = 1, num_omega
        omega = omegas(i_w)
        dp_sum = dcmplx(0.0D0, 0.0D0)
        do i_k = 1, num_kpt
            do jj = 1, num_orb
                do ll = 1, num_orb
                dp_sum = dp_sum + prod_df(ll, jj, i_k) &
                       / (delta_eng(ll, jj, i_k) + omega + cdelta)
                end do
            end do
        end do
        dyn_pol(i_w, i_q) = dp_sum
    end do
    !$OMP END PARALLEL DO
end subroutine dyn_pol_q_arb


! calculate full ac conductivity using Kubo-Greenwood formula
subroutine ac_cond_kg(eng, num_orb, num_kpt, wfn, hop_ind, num_hop, hop_eng, &
                      hop_dr, kmesh, beta, mu, omegas, num_omega, delta, &
                      comp, ac_cond)
    implicit none

    ! input and output
    real(kind=8), intent(in) :: eng(num_orb, num_kpt)
    integer, intent(in) :: num_orb, num_kpt
    complex(kind=8), intent(in) :: wfn(num_orb, num_orb, num_kpt)
    integer, intent(in) :: hop_ind(2, num_hop)
    integer, intent(in) :: num_hop
    complex(kind=8), intent(in) :: hop_eng(num_hop)
    real(kind=8), intent(in) :: hop_dr(3, num_hop)
    real(kind=8), intent(in) :: kmesh(3, num_kpt)
    real(kind=8), intent(in) :: beta, mu
    real(kind=8), intent(in) :: omegas(num_omega)
    integer, intent(in) :: num_omega
    real(kind=8), intent(in) :: delta
    integer, intent(in) :: comp(2)
    complex(kind=8), intent(inout) :: ac_cond(num_omega)

    ! local variables
    integer :: i_w, i_k, i_h, mm, nn, ib1, ib2
    real(kind=8) :: k_dot_r
    complex(kind=8) :: phase
    complex(kind=8) :: vmat1(num_orb, num_orb), vmat2(num_orb, num_orb)
    real(kind=8) :: omega, eng_m, eng_n, f_m, f_n
    complex(kind=8) :: prod1, prod2, ac_sum
    real(kind=8) :: delta_eng(num_orb, num_orb, num_kpt)
    complex(kind=8) :: prod_df(num_orb, num_orb, num_kpt)
    complex(kind=8), parameter :: p1j = (0.0D0, 1.0D0)
    complex(kind=8) :: cdelta

    ! initialize working arrays
    vmat1 = 0.0
    vmat2 = 0.0
    delta_eng = 0.0
    prod_df = 0.0

    ! build reusable arrays
    do i_k = 1, num_kpt
        ! build vmat in Bloch basis via Fourier transform
        ! NOTE: this must be done in serial mode bdecause different threads
        ! may have the same ib1 and ib2.
        ! NOTE: vmat should be initialzed as exactly zero. vmat = vmat * 0.0
        ! is incorrect under GCC.
        vmat1 = 0.0
        vmat2 = 0.0
        do i_h = 1, num_hop
            k_dot_r = kmesh(1, i_k) * hop_dr(1, i_h) &
                    + kmesh(2, i_k) * hop_dr(2, i_h) &
                    + kmesh(3, i_k) * hop_dr(3, i_h)
            phase = dcmplx(cos(k_dot_r), sin(k_dot_r)) * hop_eng(i_h)
            ib1 = hop_ind(1, i_h)
            ib2 = hop_ind(2, i_h)
            vmat1(ib1, ib2) = vmat1(ib1, ib2) + p1j * phase * hop_dr(comp(1), i_h)
            vmat1(ib2, ib1) = vmat1(ib2, ib1) - p1j * conjg(phase) * hop_dr(comp(1), i_h)
            vmat2(ib1, ib2) = vmat2(ib1, ib2) + p1j * phase *  hop_dr(comp(2), i_h)
            vmat2(ib2, ib1) = vmat2(ib2, ib1) - p1j * conjg(phase) * hop_dr(comp(2), i_h)
        end do

        ! build delta_eng and prod_df
        !$OMP PARALLEL DO PRIVATE(eng_m, f_m, nn, eng_n, f_n, prod1, prod2, ib1, ib2)
        do mm = 1, num_orb
            eng_m = eng(mm, i_k)
            f_m = 1.0 / (1.0 + exp(beta * (eng_m - mu)))
            do nn = 1, num_orb
                eng_n = eng(nn, i_k)
                delta_eng(nn, mm, i_k) = eng_m - eng_n
                f_n = 1.0 / (1.0 + exp(beta * (eng_n - mu)))
                prod1 = 0.0
                prod2 = 0.0
                do ib1 = 1, num_orb
                    do ib2 = 1, num_orb
                        prod1 = prod1 + conjg(wfn(ib1, nn, i_k)) * vmat1(ib1, ib2) * wfn(ib2, mm, i_k)
                        prod2 = prod2 + conjg(wfn(ib1, mm, i_k)) * vmat2(ib1, ib2) * wfn(ib2, nn, i_k)
                    end do
                end do
                if (abs(eng_m - eng_n) >= 1.0D-7) then
                    prod_df(nn, mm, i_k) = prod1 * prod2 * (f_m - f_n) / (eng_m - eng_n)
                ! else
                !     prod_df(nn, mm, i_k) = prod1 * prod2 * -beta * f_n * (1 - f_n)
                end if
            end do
        end do
        !$OMP END PARALLEL DO
    end do

    ! calculate ac_cond
    cdelta = dcmplx(0.0D0, delta)
    !$OMP PARALLEL DO PRIVATE(omega, ac_sum, i_k, mm, nn)
    do i_w = 1, num_omega
        omega = omegas(i_w)
        ac_sum = dcmplx(0.0D0, 0.0D0)
        do i_k = 1, num_kpt
            do mm = 1, num_orb
                do nn = 1, num_orb
                    ac_sum = ac_sum + prod_df(nn, mm, i_k) &
                           / (delta_eng(nn, mm, i_k) - omega - cdelta)
                end do
            end do
        end do
        ac_cond(i_w) = ac_sum
    end do
    !$OMP END PARALLEL DO
end subroutine ac_cond_kg
