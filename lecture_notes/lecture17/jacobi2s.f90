module jacobi
    implicit none
    integer :: kmax !max number of iterations
    real(kind=8) :: tol !convergence criterion
    real(kind=8), allocatable, dimension(:) :: deltat
    save
contains
!-----------------------------------------------------
!Solve 2-d heat equation w,ith Jacobi iteration
subroutine jacobi2d(n,Ubc,Dbc,Lbc,Rbc,S0,T)
    !input  n: number of grid points, x=linspace(0,1,n+2)
    !       a,b: boundary onditions
    !       S0: Source amplitude, S = S0 sin(pi*x)
    !output T: final temperature distribution
    !       deltaT(k): max(|T^k - T^k-1|)
    !parameters set in calling program   tol: convergence criterion
    !                                    kmax: maximum number of iterations
    integer, intent(in) :: n
    real(kind=8), intent(in) :: Ubc,Dbc,Lbc,Rbc,S0 !b.c.'s, source amplitude
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: T
    integer :: i1,j1,k1
    real(kind=8) :: pi,del,del2f !grid spacing
    real(kind=8), dimension(0:n+1,0:n+1) :: x,y,S,Tnew

    if (allocated(deltaT)) then
      deallocate(deltaT)
    end if

    allocate(deltaT(kmax))
    pi = acos(-1.d0)

    !grid--------------
    del = 1.d0/dble(n+1)
    del2f = 0.25d0*(del**2)


    do i1=0,n+1
        x(:,i1) = i1*del
    end do

    do j1=0,n+1
        y(j1,:) = j1*del
    end do
    !-------------------

    !set initial condition
    T = (Rbc-Lbc)*x + Lbc + (Ubc-Dbc)*y + Dbc

    !set boundary conditions
    T(0,:) = Dbc
    T(n+1,:) =  Ubc
    T(:,0) = Lbc
    T(:,n+1) = Rbc

    !set source function
    S = S0*sin(pi*x)*sin(pi*y)

    do k1=1,kmax
        Tnew(1:n,1:n) = S(1:n,1:n)*del2f + 0.25d0*(T(2:n+1,1:n) + T(0:n-1,1:n) + T(1:n,0:n-1) + T(1:n,2:n+1)) !Jacobi iteration
        deltaT(k1) = maxval(abs(Tnew(1:n,1:n)-T(1:n,1:n))) !compute relative error
        T(1:n,1:n)=Tnew(1:n,1:n)    !update variable
        if (deltaT(k1)<tol) exit !check convergence criterion
    end do

    print *, 'k,error=',k1,deltaT(min(k1,kmax))

end subroutine jacobi2d
!-----------------------------------------------------
!Solve 1-d heat equation w,ith Jacobi iteration
subroutine jacobi1d(n,a,b,S0,T)
    !input  n: number of grid points, x=linspace(0,1,n+2)
    !       a,b: boundary onditions
    !       S0: Source amplitude, S = S0 sin(pi*x)
    !output T: final temperature distribution
    !       deltaT(k): max(|T^k - T^k-1|)
    !parameters set in calling program   tol: convergence criterion
    !                                    kmax: maximum number of iterations
    integer, intent(in) :: n
	  integer :: i1,k1
	  real(kind=8), intent(in) :: a,b,S0 !b.c.'s, source amplitude
    real(kind=8) :: pi,dx,dx2f !grid spacing
	  real(kind=8), dimension(0:n+1) :: x,S,Tnew !grid, source, Temperatures
    real(kind=8), dimension(0:n+1), intent(out) :: T

    if (allocated(deltaT)) then
      deallocate(deltaT)
    end if
    

    allocate(deltaT(kmax))
    pi = acos(-1.d0)

    !grid--------------
    dx = 1.d0/dble(n+1)
    dx2f = 0.5*(dx**2)

    do i1=0,n+1
        x(i1) = i1*dx
    end do
    !-------------------

    !set initial condition
    T = (b-a)*x + a

    !set source function
    S = S0*sin(pi*x)

    do k1=1,kmax
        Tnew(1:n) = (S(1:n)*dx2f + 0.5d0*(T(0:n-1) + T(2:n+1))) !Jacobi iteration
        deltaT(k1) = maxval(abs(Tnew(1:n)-T(1:n))) !compute relative error
        T(1:n)=Tnew(1:n)    !update variable
        if (deltaT(k1)<tol) exit !check convergence criterion
    end do

    print *, 'k,error=',k1,deltaT(min(k1,kmax))

end subroutine jacobi1d
!-----------------------------------------------------

end module jacobi
