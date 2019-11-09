! Lab 4, task 2

program task2
    implicit none
    integer, parameter :: N=5, display_iteration = 1
    real(kind=8) :: odd_sum
    


    call computeSum(N,odd_sum,display_iteration)
    print *, 'odd_sum=',odd_sum


end program task2

!------------
!Subroutine which sums the first N odd integers
subroutine computeSum(N,odd_sum,display_iteration)
    implicit none
    integer, intent(in) :: N, display_iteration
    real(kind=8), intent(out) :: odd_sum
    integer :: i1
    

    odd_sum = 0.d0

    do i1=1,N
        odd_sum = odd_sum + (2.d0*i1)-1.d0
        if (display_iteration == 1) then
            print *, 'iteration', i1, 'sum=', odd_sum
        end if        
    end do

end subroutine computeSum

! To compile this code:
! $ gfortran -o task2.exe lab42soln.f90
! To run the resulting executable: $ ./task2.exe
