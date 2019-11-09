!module for computing circumference and area of circle
module circle
    implicit none
    real(kind=8) :: circle_pi
    save
contains
!-------------------------
subroutine initialize_pi()
    implicit none

    circle_pi = acos(-1.d0)

end subroutine initialize_pi
!------------------------------------------
function circumference(radius)
    !compute circumference of circle given the radius
    implicit none
    real(kind=8), intent(in) :: radius
    real(kind=8) :: circumference

    circumference = 2.d0*circle_pi*radius

end function circumference

!---------------------------------
function area(radius)
    !compute area of circle given the radius
    implicit none
    real(kind=8), intent(in) :: radius
    real(kind=8) :: area

    area = circle_pi*(radius**2)

end function area
!----------------
end module circle




