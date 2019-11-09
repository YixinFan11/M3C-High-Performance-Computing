! This codes illustrates array operations in Fortran
!----------------------------------------------------------------------
program array1

	!Variable declarations:
	implicit none
	integer :: i1,j1,N
    integer, parameter :: c = 2 !variables declared as parameters cannot be changed within program
	integer, dimension(4) :: x,y,z
    integer, dimension(4,4) :: A

    x= (/1,2,3,4/) !initialize x

    y = c*(x*x) !c*((x(1)*x(1),x(2)*x(2),...,x(4)*x(4))

    print *, 'x=',x
    print *, 'y=',y

!    construct matrix A = [x' 2*x' 3*x' 4*x']
    do i1=1,4
        A(:,i1) = i1*x
        print *, 'column',i1,'of A=',A(:,i1)
    end do

    !illustrate matrix multiplication with matmul
    print *, 'A*x = ',matmul(A,x)

end program array1
!--------------------------------------------------------------------

!to compile: gfortran -o array1.exe array1.f90
!to run: ./array1.exe


	 
