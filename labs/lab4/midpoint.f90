!midpoint.f90
!Compute the integral of 4/(1+x^2) from 0 to 1 using the midpoint rule
!Input: number of intervals, N
!Output (to screen): number of intervals, estimated value of integral, and error
!To compile this code: gfortran -o midpoint.exe midpoint.f90
!To run: ./midpoint.exe

program midpoint
	implicit none
    integer :: i1,N
    real(kind=8) :: pi
    real(kind=8) :: dx, sum_i, xm, S, f, a, error


    S = 0.d0 !initialize integral


    !read data from data.in
	open(unit=10, file='data.in')
        read(10,*) N
	close(10)

    dx = 1.d0/dble(N) !interval size

    !loop over intervals computing each interval's contribution to integral
    do i1 = 1,N
        xm = dx*(dble(i1)-0.5d0) !midpoint of interval i1
        call integrand(xm,f)
        sum_i = dx*f
        S = S + sum_i !add contribution from interval to total integral
    end do

    pi = asin(1.d0)*2.d0
    error = abs(S - pi)
    print *, 'N=', N
    print *, 'sum=', S
    print *, 'error=', error

end program midpoint
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


