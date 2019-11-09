!midpoint_omp.f90 with timing info
!Compute the integral of 4/(1+x^2) from 0 to 1 using the midpoint rule
!Computes timing information; parallelized with OpenMP
!Input: number of intervals, N
!Output (to screen): number of intervals, estimated value of integral, and error
!To compile this code: gfortran -fopenmp -o midpoint.exe midpoint_time_omp.f90
!To run: ./midpoint.exe

program midpoint_time_omp
    use omp_lib
    implicit none
    logical, parameter :: debug = .false.
    integer :: i1,j1,N
    double precision :: dx, sum_i, xm, tsum, f, a, error, pi
    integer :: NumThreads,threadID,NT
    integer(kind=8) :: ta,tb,clock_rate

    pi = acos(-1.d0)


    !read data from data.in
    open(unit=10, file='data.in')
        read(10,*) N
!$        read(10,*) NT
    close(10)

!$    call omp_set_num_threads(NT)

    dx = 1.d0/dble(N) !interval size

    call system_clock(ta)
    do j1=1,10
        tsum = 0.d0
        !loop over intervals computing each interval's contribution to integral
        !$OMP parallel do private(xm,f),reduction(+:tsum)
        do i1 = 1,N
            xm = dx*(dble(i1)-0.5d0) !midpoint of interval i1
            call integrand(xm,f)
            tsum = tsum + dx*f !add contribution from interval to total integral

        if (debug) then
            NumThreads = omp_get_num_threads()
            ThreadID = omp_get_thread_num()
        end if

    end do
    !$OMP end parallel do
    end do
    call system_clock(tb,clock_rate)


    error = abs(tsum - pi)
    print *, 'N=', N
    print *, 'sum=', tsum
    print *, 'error=', error
    if (debug) print *, 'NumThreads=',NumThreads
    print *, 'wall time=', float(tb-ta)/float(clock_rate)
end program midpoint_time_omp
!--------------------------------------------------------------------

!----------------------------------
!subroutine integrand
!   compute integrand, 4.0/(1+a^2)
!----------------------------------

subroutine integrand(a,f)
    implicit none
    double precision, intent(in) :: a
    double precision, intent(out) :: f
    f = 4.d0/(1.d0 + a*a)
end subroutine integrand
