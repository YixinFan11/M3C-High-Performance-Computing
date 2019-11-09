!midpoint_p.f90
!Parallel version of midpoint.f90
!Compute the integral of 4/(1+x^2) from 0 to 1 using the midpoint rule
!Input: number of intervals, N
!Output (to screen): number of intervals, procs, estimated value of integral, and error
!To compile this code: mpif90 -o midpoint_p.exe midpoint_p.f90
!To run: mpiexec -n 2 midpoint_p.exe
program midpoint_p
    use mpi
    implicit none
    integer :: i1,N
    real(kind=8) :: pi
    real(kind=8) :: dx, sum_i, xm, sum_total, f, a, error, sum_proc

    integer :: myid, numprocs, ierr
    integer :: Nper_proc,istart,iend


    ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)


    pi = asin(1.d0)*2.d0
    sum_proc = 0.d0 !initialize integral



    !read data from data.in
	open(unit=10, file='data.in')
        read(10,*) N
	close(10)

    dx = 1.d0/dble(N) !interval size


    !set number of intervals per processor
    Nper_proc =  (N + numprocs - 1)/numprocs

    if (myid==0) then !output interval allocation details
        print *, 'number of intervals = ', N
        print *, 'number of procs = ', numprocs
        print *, 'Nper_proc= ', Nper_proc
    end if

    !starting and ending points for processor
    istart = myid * Nper_proc + 1
    iend = (myid+1) * Nper_proc
    if (iend>N) iend = N

    !loop over intervals computing each interval's contribution to integral
    do i1 = istart,iend
        xm = dx*(i1-0.5d0) !midpoint of interval i1
        call integrand(xm,f)
        sum_i = dx*f
        sum_proc = sum_proc + sum_i !add contribution from interval to total integral
    end do

    print *, 'The partial sum on proc #', myid, 'is:', sum_proc

    !reduce sums
    !collect double precision variable, sum, with size 1 on process 0 using the MPI_SUM option
    call MPI_REDUCE(sum_proc,sum_total,1,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)


    !output
    if (myid == 0) then
        error = abs(sum_total - pi)
        print *, 'N=', N
        print *, 'sum=', sum_total
        print *, 'error=', error
    end if
    call MPI_FINALIZE(ierr)
end program midpoint_p
!--------------------------------------------------------------------

!----------------------------------
!subroutine integrand
!   compute integrand, 4.0/(1+a^2)
!----------------------------------

subroutine integrand(a,f)
    implicit none
    double precision, intent(in) :: a
    double precision, intent(out) :: f
    f = 4.0/(1.0 + a*a)
end subroutine integrand


