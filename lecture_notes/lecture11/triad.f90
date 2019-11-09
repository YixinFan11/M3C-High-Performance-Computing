!module to test perfomance with triad test
!compile with:
!f2py3 -c triad.f90 -m trf90
!and then use triad.py to run and display results.
module triad

contains

subroutine compute_triad(n,time)
    implicit none
    integer, intent(in) :: n
    real(kind=8), intent(out) :: time

    integer :: i1,j1
    integer(kind=8) :: start,stop,clockrate
    real(kind=8), dimension(n) :: a,b,c,d


	!initialize arrays
  a = 0.d0
  b = 1.d0
  c = 2.d0
  d = 3.d0

    !----
    !principal computation
  call system_clock(start)
  do j1 = 1,1000000
    do i1 = 1,n
      a(i1) = b(i1) + c(i1) * d(i1)
    end do
    if (a(2) < 0) call dummy(a,b,c,d)
  end do
    call system_clock(stop,clockrate)
    !------

    time = float(stop-start)/float(clockrate)
    time = time/n

end subroutine compute_triad
!----------------------------------------------
subroutine dummy(a,b,c,d)
    !prevents compiler from removing outer loop in triad computation
    implicit none
    real(kind=8), dimension(:) :: a,b,c,d

    if (a(2)>0) print *, a(2)

end subroutine dummy
end module triad
!----------------------------------------------
program call_triad
    !simple driver program
    use triad
    implicit none
    integer :: n
    real(kind=8) :: time


    open(unit=12,file='data.in')
    read(12,*) n
    close(12)

    call compute_triad(n,time)

    print *, 'n=',n, 'time=',time
end program call_triad
