module sine_array
  implicit none
  real(kind=8), allocatable, dimension(:) :: x

contains

subroutine sin_loop(n,s)
  implicit none
  integer, intent(in) :: n
  real(kind=8), dimension(n), intent(out) :: s
  integer :: i1

  do i1 = 1,n
    s(i1) = sin(x(i1))
  end do

end subroutine sin_loop


subroutine sin_vec(n,s)
  implicit none
  integer, intent(in) :: n
  real(kind=8), dimension(n), intent(out) :: s
  integer :: i1

  s = sin(x)

end subroutine sin_vec


end module sine_array


program test_sine
  use sine_array
  implicit none
  real(kind=8), allocatable, dimension(:) :: s

  allocate(x(1000),s(1000))
  call random_number(x)

!  call sin_loop(1000,s)
  call sin_vec(1000,s)

end program test_sine
