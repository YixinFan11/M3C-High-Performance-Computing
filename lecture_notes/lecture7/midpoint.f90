program midpoint
  implicit none
  !variable declarations
  integer :: n,i1
  real(kind=8) :: dx, x, fsum

  !read in n, number of intervals
  open(unit=9,file='data.in')
  read(9,*) n
  close(9)

  dx = 1.d0/n
  fsum = 0.d0
  !loop through subintervals
    do i1 = 1,n
    !compute area of subintervals
      x = dx*(dble(i1)-0.5d0)
      fsum = fsum + dx*4.d0/(1+x**2)
    !and add to running sum

    end do
  !output total sum
  print *, 'fsum=',fsum


end program midpoint
