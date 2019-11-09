! This codes illustrates array operations in Fortran with allocatable arrays
!---------------------------------------------------------------------------
program array2

	!Variable declarations:
	implicit none
	integer :: i1,j1,N
    integer, parameter :: c = 2
	integer, allocatable, dimension(:) :: x,y,z
    integer, allocatable, dimension(:,:) :: A

    allocate(x(4),y(4))

    x= (/1,2,3,4/) !initialize x

    y = c*(x*x) !c*((x(1)*x(1),x(2)*x(2),...,x(4)*x(4))

    print *, 'x=',x
    print *, 'y=',y

    deallocate(y)

!    construct matrix A = [x' 2*x' 3*x' 4*x']
    allocate(A(4,4))
    do i1=1,4
        A(:,i1) = i1*x
        print *, 'column',i1,'of A=',A(:,i1)
    end do

    !illustrate matrix multiplication with matmul
    print *, 'A*x = ',matmul(A,x)
    deallocate(A,x)

end program array2
!--------------------------------------------------------------------

!to compile: gfortran -o array2.exe array2.f90
!to run: ./array2.exe


	 
