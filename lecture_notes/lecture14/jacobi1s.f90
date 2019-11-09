module jacobi
    implicit none
    integer :: kmax
    real(kind=8) :: tol
    real(kind=8), allocatable, dimension(:) :: deltat
    save
contains

!Solve 1-d heat equation w,ith Jacobi iteration
subroutine jacobi1(n,a,b,S0,T)
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

    if (allocated(deltaT) .neqv. .true.) allocate(deltaT(kmax))
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

end subroutine jacobi1
end module jacobi
