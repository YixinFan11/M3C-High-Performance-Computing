!midpoint.f90 with timing commands
!Compute the integral of 4/(1+x^2) from 0 to 1 using the midpoint rule
!Input: number of intervals, N
!Output (to screen): number of intervals, estimated value of integral, and error
!To compile this code: gfortran -o midpoint_time.exe midpoint_time.f90
!To run: ./midpoint.exe

program midpoint_time
	implicit none
    integer :: i1,N
    real(kind=8) :: pi
    real(kind=8) :: dx, sum_i, xm, sum, f, a, error
!timing variables
    real(kind=8) :: cpu_t1,cpu_t2,clock_time
    integer(kind=8) :: clock_t1,clock_t2,clock_rate


    pi = acos(-1.d0)
    sum = 0.d0 !initialize integral


    !read data from data.in
	open(unit=10, file='data.in')
        read(10,*) N
	close(10)

    dx = 1.d0/dble(N) !interval size

    call system_clock(clock_t1)
    call cpu_time(cpu_t1)
    !loop over intervals computing each interval's contribution to integral
    do i1 = 1,N
        xm = dx*(dble(i1)-0.5d0) !midpoint of interval i1
        call integrand(xm,f)
        sum_i = dx*f
        sum = sum + sum_i !add contribution from interval to total integral
    end do

    call cpu_time(cpu_t2)
    print *, 'elapsed cpu time (seconds) =',cpu_t2-cpu_t1

    call system_clock(clock_t2,clock_rate)
    print *, 'elapsed wall time (seconds)= ',float(clock_t2-clock_t1)/float(clock_rate)

    error = abs(sum - pi)
    print *, 'N=', N
    print *, 'sum=', sum
    print *, 'error=', error

end program midpoint_time
!--------------------------------------------------------------------

!----------------------------------
!subroutine integrand
!   compute integrand, 4.0/(1+a^2)
!----------------------------------

subroutine integrand(a,f)
    implicit none
    real(kind=8), intent(in) :: a
    real(kind=8), intent(out) :: f
    f = 4.d0/(1.d0 + a*a)
end subroutine integrand


