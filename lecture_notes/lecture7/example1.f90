! This code illustrates the basic structure of Fortran 90 code
! It reads an integer, N, from an input file and computes sin(x) for 
! x = 1,2,..., N
! To compile code:  gfortran -o example1.exe example1.f90
! Then, run executable: ./example1.exe

!1. Header:
program example1

	!2. Variable declarations:
	implicit none !means all variables in code must be declared
	integer :: i1,j1,N
	real(kind=8) :: var1, var2
	real(kind=8), dimension(10) :: array1

	!3. basic code: input, loops, if-statements, subroutine calls

	!read data from data.in
	open(unit=10, file='data.in')
        read(10,*) N
	close(10)

	!check that N is smaller than size of array1:
	if (N <= size(array1)) then
		!compute sin(x) where x = 1,2,3,...,N
		
		do i1 = 1,N !loop from 1 to N
	                var1 = dble(i1) !convert integer to double-precision real number
			array1(i1) = sin(var1)
		end do

		!print 1st N elements of array
		print *, 'array1=',array1(1:N)
	else
		print *, 'N must be smaller than', size(array1)
        STOP
	end if

!4. end program
end program example1



	


	 
