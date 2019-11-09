! Lab 4, task 1

program task1
    implicit none
    integer :: i1
    integer, parameter :: N=5
    real(kind=8) :: odd_sum

    odd_sum = 0.d0
!Add a do-loop which sums the first N odd integers and prints the result
    do i1=1,N
        odd_sum = odd_sum + (2.d0*i1)-1.d0
    end do
    
    print *, 'odd_sum=',odd_sum


end program task1

! To compile this code:
! $ gfortran -o task1.exe lab41.f90
! To run the resulting executable: $ ./task1.exe
