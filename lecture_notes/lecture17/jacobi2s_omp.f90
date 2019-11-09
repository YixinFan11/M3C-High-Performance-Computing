!Solve 1d or 2d steady heat eqn with Jacobi iteration (parallelized with OpenMP)
!1. Compile with f2py -c jacobi2s_omp.f90 -lgomp --f90flags="-fopenmp -freal-4-real-8" -m jf90_omp
!2. Import into python with: from jf90_omp_sp import jacobi_omp as jac
!3. Set jac.kmax, jac.numthreads, jac.deltat and jac.tol
!4. call jac.jacobi2d_omp or jac_jacobi1d_omp with appropriate input

module jacobi_omp
    use omp_lib
    implicit none
    integer :: kmax,numthreads
    real(kind=8) :: tol
    real(kind=8), allocatable, dimension(:) :: deltat
    save
contains
!-----------------------------------------------------
!Solve 2-d heat equation w,ith Jacobi iteration
subroutine jacobi2d_omp(n,Ubc,Dbc,Lbc,Rbc,S0,T)
    !input  n: number of grid points, x=linspace(0,1,n+2)
    !       a,b: boundary onditions
    !       S0: Source amplitude, S = S0 sin(pi*x)
    !output T: final temperature distribution
    !       deltaT(k): max(|T^k - T^k-1|)
    !parameters set in calling program   tol: convergence criterion
    !                                    kmax: maximum number of iterations
    !                                    numthreads: for OpenMP
    integer, intent(in) :: n
    real(kind=8), intent(in) :: Ubc,Dbc,Lbc,Rbc,S0 !b.c.'s, source amplitude
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: T
    integer :: i1,j1,k1
    real(kind=8) :: pi,del,del2f,dmax !grid spacing
    real(kind=8), dimension(0:n+1,0:n+1) :: x,y,S,Tnew

    if (allocated(deltaT)) then
      deallocate(deltaT)
    end if

    allocate(deltaT(kmax))
    pi = acos(-1.d0)

    !grid--------------
    del = 1.0/dble(n+1)
    del2f = 0.25*(del**2)

    !$OMP parallel do
    do i1=0,n+1
        x(:,i1) = i1*del
        y(i1,:) = i1*del
    end do
    !$OMP end parallel do
    !-------------------

    !set initial condition

    !set source function
    !$OMP parallel do
    do j1=0,n+1
        T(:,j1) = (Rbc-Lbc)*x(:,j1) + Lbc + (Ubc-Dbc)*y(:,j1) + Dbc

        S(:,j1) = S0*sin(pi*x(:,j1))*sin(pi*y(:,j1))
    end do
    !$OMP end parallel do

    !set boundary conditions
    !$OMP parallel do
    do j1=0,n+1
        T(0,j1) = Dbc
        T(n+1,j1) = Ubc
        T(j1,0) = Lbc
        T(j1,n+1) = Rbc
    end do
    !$OMP end parallel do

    do k1=1,kmax
        dmax = 0.d0
        !$OMP parallel do reduction(max:dmax)
        do j1=1,n
            Tnew(1:n,j1) = S(1:n,j1)*del2f + 0.25*(T(2:n+1,j1) + T(0:n-1,j1) + T(1:n,j1-1) + T(1:n,j1+1))!Jacobi iteration
            dmax = max(dmax,maxval(abs(Tnew(1:n,j1)-T(1:n,j1))))
        end do
        !$OMP end parallel do
        deltaT(k1) = dmax
        if (deltaT(k1)<tol) exit !check convergence criterion
        !$OMP parallel do
        do j1=1,n
            T(1:n,j1) = Tnew(1:n,j1)
        end do
        !$OMP end parallel do
    end do

    print *, 'k,error=',k1,deltaT(min(k1,kmax))

end subroutine jacobi2d_omp
!-----------------------------------------------------
!Solve 1-d heat equation w,ith Jacobi iteration
subroutine jacobi1d_omp(n,a,b,S0,T)
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

    if (allocated(deltaT)) then
      deallocate(deltaT)
    end if
    
    allocate(deltaT(kmax))
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

end subroutine jacobi1d_omp
!-----------------------------------------------------

end module jacobi_omp
