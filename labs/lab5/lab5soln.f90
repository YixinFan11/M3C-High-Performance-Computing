!Lab 5 solution code
!Compute the integral of 4/(1+x^2) from 0 to 1 using the midpoint or simpson rule
!Input: number of intervals, N
!Output (to screen): number of intervals, estimated value of integral, and error
!To compile this code: gfortran -o quad.exe lab5soln.f90
!To run: ./quad.exe
!For f2py: f2py -c lab5soln.f90 -m l5
!which generates l5.so


module quad
	implicit none
	integer :: quad_n !number of intervals
	real(kind=8) :: quad_sum !computed integral
contains
	!----------------------------------
	!subroutine midpoint
	! Use midpoint rule to compute
	! integral of, 4.0/(1+x^2)
	! from x=0 to x=1
	!----------------------------------
	subroutine midpoint()
	    implicit none
	    integer :: i1
	    real(kind=8) :: dx, sum_i, xm, f, a, error

	     quad_sum = 0.d0 !initialize integral
	     dx = 1.d0/dble(quad_n) !interval size

	    !loop over intervals computing each interval's contribution to integral
	    do i1 = 1,quad_n
	        xm = dx*(dble(i1)-0.5d0) !midpoint of interval i1
	        call integrand(xm,f)
	        sum_i = dx*f
	        quad_sum = quad_sum + sum_i !add contribution from interval to total integral
	    end do

	end subroutine midpoint
	!--------------------------------------------------------------------

	!----------------------------------
	!subroutine simpson
	! Use Simpson's rule to compute
	! integral of, 4.0/(1+x^2)
	! from x=0 to x=1
	!----------------------------------
	subroutine simpson()
	    implicit none
	    integer :: i1
	    real(kind=8) :: dx, sum_i, xm, f, a, error

	     quad_sum = 0.d0 !initialize integral
	     dx = 1.d0/dble(quad_n) !interval size

	    !loop over intervals computing each interval's contribution to integral
	    do i1 = 1,quad_n+1
	        xm = dx*(dble(i1)-1.d0)
	        call integrand(xm,f)
					if (mod(i1,2)==0) f=f*2.d0
					if ((i1==1) .or. (i1==quad_n+1)) f=f/2.d0 !adjust endpoints
					sum_i = dx*f
	        quad_sum = quad_sum + sum_i !add contribution from interval to total integral
	    end do
			quad_sum = quad_sum*(2.d0/3.d0)
	end subroutine simpson

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

	!--------------------------------------------------------------------
end module quad



program lab5
	use quad
	implicit none
    integer :: quad_type
    real(kind=8) :: pi, error

    !read data from data.in
		open(unit=10, file='data.in')
	  	read(10,*) quad_n
	    read(10,*) quad_type
		close(10)

    !compute integral
    if (quad_type==1) then
        call midpoint()
    elseif (quad_type==2) then
        call trapezoid()
		else
				call simpson()
    end if

    pi = acos(-1.d0)
    error = abs(quad_sum - pi)
    print *, 'N=', quad_n
    print *, 'integral=', quad_sum
    print *, 'error=', error


end program lab5

!--------------------------------------------------------------------
