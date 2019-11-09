!Simple example of Fortran function
!Two numbers are added together

program function_example
    implicit none
    real(kind=8) :: x,y,z
    real(kind=8), external :: sumxy,sumxy_alternate !function called in main program

    x = 2.d0
    y = 3.d0
    z = sumxy(x,y)

    print *, 'x,y,z=',x,y,z

end program function_example


!--------------------------------------------------------------------
!function to sum two numbers provided as input
!note that the function name is a variable whose
!type can be declared in the header
function sumxy(x,y)
    implicit none
    real(kind=8), intent(in) :: x,y
    real(kind=8) :: sumxy

    sumxy = x + y
end function sumxy

!--------------------------------------------------------------------
!Note that the variable corresponding to the
!function name can also be declared in the header
!
real(kind=8) function sumxy_alternate(x,y)
    implicit none
    real(kind=8), intent(in) :: x,y

    sumxy_alternate = x + y
end function sumxy_alternate

!--------------------------------------------------------------------
