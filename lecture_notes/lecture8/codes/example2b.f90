! This code illustrates the basic structure of Fortran 90 code
! It reads an integer, N, from an input file and computes sin(x) for 
! x = 1,2,..., N
! It is a modified version of example2; here, allocatable arrays
! are used.
! To compile code:  gfortran -o example2b.exe example2b.f90
! Then, run executable: ./example2b.exe
!----------------------------------------------------------------------
!1. Header:
program example2b

	!2. Variable declarations:
	implicit none !means all variables in code must be declared
	integer :: i1,j1,N
	real(kind=8) :: var1, var2
	real(kind=8), allocatable, dimension(:) :: array1



	!3. basic code: input, loops, if-statements, subroutine calls

	!read data from data.in
	open(unit=10, file='data.in')
        read(10,*) N
	close(10)

    allocate(array1(N))
	
        !compute sin(x) where x = 1,2,3,...,N
		 call calculations(N,array1)

		!print 1st N elements of array
		print *, 'array1=',array1(1:N)

    deallocate(array1)
!4. end program
end program example2b
!--------------------------------------------------------------------

!-----------------------
!subroutine calculations
!
!-----------------------
subroutine calculations(N,array)
    implicit none
    integer, intent(in) :: N
    real(kind=8), dimension(N), intent(out) :: array
    integer :: i1
    real(kind=8) :: var1

    do i1 = 1,N !loop from 1 to N
            var1 = dble(i1) !convert integer to real number
			array(i1) = sin(var1)
    end do
end subroutine calculations

!--------------------------------------------------------------------


	 
