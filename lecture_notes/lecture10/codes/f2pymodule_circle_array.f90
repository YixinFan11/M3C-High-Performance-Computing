!module for computing circumference, area, and "mass" of circle
module circle
    implicit none
    real(kind=8) :: pi
    real(kind=8), allocatable, dimension(:) :: weights,mass
    save
contains

subroutine initialize_pi()
    implicit none

    pi = acos(-1.d0)

end subroutine initialize_pi

function circumference(radius)
    !compute circumference of circle given the radius
    implicit none
    real(kind=8), intent(in) :: radius
    real(kind=8)  :: circumference

    circumference = 2.d0*pi*radius

end function circumference


function area(radius)
    !compute area of circle given the radius
    implicit none
    real(kind=8), intent(in) :: radius
    real(kind=8)  :: area

    area = pi*(radius**2)

end function area

subroutine compute_mass(radius)
    !compute mass = weights*area
    implicit none
    real(kind=8), intent(in) :: radius

    if (.not. allocated(mass)) allocate(mass(size(weights)))
    mass = weights*area(radius)
    print *, 'weights=',weights
    print *, 'mass=', mass
    print *, 'pi=', pi


end subroutine compute_mass

end module circle
