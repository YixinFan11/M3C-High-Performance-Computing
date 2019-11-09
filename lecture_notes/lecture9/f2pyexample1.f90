!--------------------------------------------------------------------
!function to sum two numbers provided as input
!note that the function name is a variable whose
!type must be declared
function sumxy(x,y)
    implicit none
    double precision, intent(in) :: x,y
    double precision :: sumxy

    sumxy = x + y
end function sumxy

!--------------------------------------------------------------------
!--------------------------------------------------------------------
!subroutine to sum two numbers provided as input
subroutine sumxy2(x,y,sumxy)
    implicit none
    double precision, intent(in) :: x,y
    double precision, intent(out) :: sumxy

    sumxy = x + y
    print *, 'test3'
end subroutine sumxy2

