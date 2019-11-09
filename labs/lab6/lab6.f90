module sinf
  !Routines to compute g = sin(x) using loop and vectorized approaches
  implicit none
  real(kind=8), allocatable, dimension(:) :: x,g

  contains

  subroutine sin_loop()
    implicit none
    integer :: i1

    if (.not. allocated(g)) allocate(g(size(x)))

    do i1 = 1,size(x)
      g(i1) = sin(x(i1))
    end do

  end subroutine sin_loop

  subroutine sin_vec()
    implicit none
    integer :: i1

    if (.not. allocated(g)) allocate(g(size(x)))
    g = sin(x)

  end subroutine sin_vec

  subroutine sin_loop_omp()
    !Parallelize this subroutine with OpemMP
    implicit none
    integer :: i1

    if (.not. allocated(g)) allocate(g(size(x)))

    do i1 = 1,size(x)
      g(i1) = sin(x(i1))
    end do

  end subroutine sin_loop_omp

end module sinf

!------------
program lab6
  !Compile: $ gfortran -fopenmp -O3 -o lab6.exe lab6.f90
  !Run: $ ./lab6.exe
  use sinf
  implicit none
  integer, parameter :: n = 4000000
  integer :: i1,numThreads
  integer(kind=8) :: t1,t2,rate
    allocate(x(n),g(n))
    call random_number(x)

    !This loop helps improve quality of timing results below
    do i1=1,10
      call sin_vec()
    end do

    !vectorized, serial
    call system_clock(t1)
    call sin_vec()
    call system_clock(t2,rate)
    print *, 'serial, vectorized time:', dble(t2-t1)/dble(rate)

    !loop, serial
    call system_clock(t1)
    call sin_loop()
    call system_clock(t2,rate)
    print *, 'serial, loop time:', dble(t2-t1)/dble(rate)

    !loop, parallel
    !Add code here to get the number of threads used in parallel regions
    print *, 'numThreads=', numThreads
    call system_clock(t1)
    call sin_loop_omp()
    call system_clock(t2,rate)
    print *, 'parallel, loop time:', dble(t2-t1)/dble(rate)

end program lab6
