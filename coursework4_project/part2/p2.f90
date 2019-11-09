!Tudor Trita Trita
!M3C 2018
!CID 01199397

module bmodel
    implicit none
    integer :: bm_kmax=1000000 !max number of iterations
    real(kind=8) :: bm_tol=0.00000001d0 !convergence criterion
    real(kind=8), allocatable, dimension(:) :: deltac !|max change in C| each iteration
    real(kind=8) :: bm_g=1.d0,bm_kbc=1.d0 !death rate, r=1 boundary parameter
    real(kind=8) :: bm_s0=1.d0,bm_r0=2.d0,bm_t0=1.5d0 !source parameters

contains
!-----------------------------------------------------
!Solve 2-d contaminant spread problem with Jacobi iteration
subroutine simulate_jacobi(n,C)
    integer, intent(in) :: n
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    integer :: i1,j1,k1
    real(kind=8) :: pi,del,del2f
    real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,Cnew,fac,fac2,facp,facm

    if (allocated(deltac)) then
      deallocate(deltac)
    end if
    allocate(deltac(bm_kmax))
    pi = acos(-1.d0)
    !grid--------------
    del = pi/dble(n+1)
    del2f = 0.25d0*(del**2)

    do i1=0,n+1
        r(i1,:) = 1.d0+i1*del
    end do
    do j1=0,n+1
        t(:,j1) = j1*del
    end do
    !-------------------

    !Update equation parameters------
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*k^2 + 2 + 2/r^2)
    facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
    facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
    fac2 = fac*rinv2
    !-----------------
    !set initial condition/boundary conditions
    C = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi
    !set source function, Sdel2 = S*del^2*fac
    Sdel2 = bm_s0*exp(-20.d0*((r-bm_r0)**2+(t-bm_t0)**2))*(del**2)*fac

    !Jacobi iteration
    do k1=1,bm_kmax
        Cnew(1:n,1:n) = Sdel2(1:n,1:n) + C(2:n+1,1:n)*facp(1:n,1:n) + C(0:n-1,1:n)*facm(1:n,1:n) + &
                                         (C(1:n,0:n-1) + C(1:n,2:n+1))*fac2(1:n,1:n)
        deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error
        C(1:n,1:n)=Cnew(1:n,1:n)    !update variable
        if (deltac(k1)<bm_tol) exit !check convergence criterion
        if (mod(k1,1000)==0) print *, k1,deltac(k1)
    end do

    print *, 'k,error=',k1,deltaC(min(k1,bm_kmax))
end subroutine simulate_jacobi
!-----------------------------------------------------

!Solve 2-d contaminant spread problem with over-step iteration method
subroutine simulate(n,C)
    integer, intent(in) :: n
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    integer :: i1,j1,k1,l1,m1
    real(kind=8) :: pi,del
    real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,Cnew,fac,fac2,facp,facm

    if (allocated(deltac)) then
      deallocate(deltac)
    end if
    allocate(deltac(bm_kmax))

    pi = acos(-1.d0)

    ! Set numerical parameters
    del = pi/dble(n+1)

    do i1=0,n+1
        r(i1,:) = 1.d0+dble(i1)*del
    end do

    do j1=0,n+1
        t(:,j1) = dble(j1)*del
    end do
    !--------------------

    ! Factors used in update equation------
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*g^2 + 2 + 2/r^2)
    facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
    facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
    fac2 = fac*rinv2
    !-----------------

    !set initial condition/boundary conditions
    C = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi
    Cnew = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi
    !--------------------
    !set source function, Sdel2 = S*del^2*fac
    Sdel2 = bm_s0*exp(-20.d0*((r-bm_r0)**2+(t-bm_t0)**2))*(del**2)*fac
    !--------------------

    !OSI Iteration
    do k1=1,bm_kmax
        do l1=0,n-1
            do m1=0,n-1
              Cnew(l1+1, m1+1) = 0.5d0*(3.d0*(Sdel2(l1+1,m1+1) + &
                                C(l1+2,m1+1)*facp(l1+1,m1+1) + &
                                Cnew(l1,m1+1)*facm(l1+1,m1+1) + (Cnew(l1+1,m1) + &
                                C(l1+1,m1+2))*fac2(l1+1,m1+1)) - C(l1+1,m1+1))
            end do
        end do
        deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error
        C(1:n,1:n)=Cnew(1:n,1:n)    !update variable
        if (mod(k1,1000)==0) print *, k1,deltac(k1)
        if (deltac(k1)<bm_tol) then
            print *, "Converged, k = ", k1, "dc_max = ", deltac(k1)
            exit !check convergence criterion
        end if
    end do

    print *, 'k,error=',k1,deltaC(min(k1,bm_kmax))
end subroutine simulate

subroutine simulate_q4(k, n, thetastar, C)
    integer, intent(in) :: n, k(4)
    real(kind=8), dimension(0:n+1,0:n+1), intent(in) :: thetastar
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    integer :: i1,j1,k1,l1,m1, kmax_2
    real(kind=8) :: pi,del
    real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,Cnew,fac,fac2,facp,facm
    real(kind=8) :: s0_2,r0_2, k_bc !source parameters: original: 1, 2, 1.5 - bm_t0

    kmax_2 = k(4)

    if (allocated(deltac)) then
      deallocate(deltac)
    end if
    allocate(deltac(kmax_2))

    pi = acos(-1.d0)
    s0_2=k(1)
    r0_2=k(2)
    k_bc=k(3)


    ! Set numerical parameters
    del = pi/dble(n+1)

    do i1=0,n+1
        r(i1,:) = 1.d0+dble(i1)*del
    end do
    do j1=0,n+1
        t(:,j1) = dble(j1)*del
    end do

    ! Factors used in update equation------
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*g^2 + 2 + 2/r^2)
    facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
    facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
    fac2 = fac*rinv2

    !New initial condition and boundary condition
    C = (exp(-10*(t-thetastar)**2)*sin(k_bc*t)**2)*(pi+1.d0-r)/pi
    Cnew = (exp(-10*(t-thetastar)**2)*sin(k_bc*t)**2)*(pi+1.d0-r)/pi

    !set source function, Sdel2 = S*del^2*fac
    Sdel2 = s0_2*exp(-20.d0*((r-r0_2)**2+(t-bm_t0)**2))*(del**2)*fac

    !OSI Iteration
    do k1=1,kmax_2
        do l1=0,n-1
            do m1=0,n-1
              Cnew(l1+1, m1+1) = 0.5d0*(3.d0*(Sdel2(l1+1,m1+1) + &
                                C(l1+2,m1+1)*facp(l1+1,m1+1) + &
                                Cnew(l1,m1+1)*facm(l1+1,m1+1) + (Cnew(l1+1,m1) + &
                                C(l1+1,m1+2))*fac2(l1+1,m1+1)) - C(l1+1,m1+1))
            end do
        end do
        deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error
        C(1:n,1:n)=Cnew(1:n,1:n)    !update variable
        if (mod(k1,1000)==0) print *, k1,deltac(k1)
        if (deltac(k1)<bm_tol) then
            print *, "Converged, k = ", k1, "dc_max = ", deltac(k1)
            exit !check convergence criterion
        end if
    end do

    print *, 'k,error=',k1,deltaC(min(k1,kmax_2))
end subroutine simulate_q4

end module bmodel
