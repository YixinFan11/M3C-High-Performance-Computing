!module for using Jacobi iteration to solve steady-state heat equation
!should be compiled with: f2py --f90flags='-fopenmp' -lgomp -c jacobi1s_omp.f90 -m j1
!for use with jacobi1_omp.py
module jacobi
    implicit none
    integer :: kmax,numThreads
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
    use omp_lib
    integer, intent(in) :: n
	integer :: i1,k1
	real(kind=8), intent(in) :: a,b,S0 !b.c.'s, source amplitude
    real(kind=8) :: dx,dx2f !grid spacing
	real(kind=8), dimension(0:n+1) :: x,S,Tnew !grid, source, Temperatures
    real(kind=8), dimension(0:n+1), intent(out) :: T
    real(kind=8) :: pi,dmax

    !$ call omp_set_num_threads(numThreads)

    if (allocated(deltaT) .neqv. .true.) allocate(deltaT(kmax))
    pi = acos(-1.d0)

    !grid--------------
    dx = 1.d0/dble(n+1)
    dx2f = 0.5*(dx**2)

    !$omp parallel do
    do i1=0,n+1
        x(i1) = i1*dx
        T(i1) = (b-a)*x(i1) + a !set initial condition
        S(i1) = S0*sin(pi*x(i1)) !set source function
    end do
    !$omp end parallel do

    !-------------------

    do k1=1,kmax
        dmax=0.d0
        !$omp parallel do reduction(max:dmax)
        do i1=1,n
            Tnew(i1) = S(i1)*dx2f + 0.5d0*(T(i1-1) + T(i1+1))
            dmax = max(dmax,abs(Tnew(i1)-T(i1)))
        end do
        !$omp end parallel do
        deltaT(k1) = dmax
        if (deltaT(k1)<tol) exit !check convergence criterion

        !$omp parallel do 
        do i1=1,n
            T(i1) = Tnew(i1)
        end do
        !$omp end parallel do

    end do

    print *, 'k,error=',k1,deltaT(min(k1,kmax))

end subroutine jacobi1
end module jacobi
