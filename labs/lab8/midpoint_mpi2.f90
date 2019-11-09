!midpoint_mpi2.f90
!Starting code for Lab 8, part 1
!Notes:
!1. The computation of dx, Nper_proc, istart, and iend
!has been moved to subroutine decomposition
!2. A new variable, tol, has been introduced which is read from data.in
!3. A new integer has been introduced, recompute, which should be set to 1 if error>tol

!Compute the integral of 4/(1+x^2) from 0 to 1 using the midpoint rule
!Input: number of intervals, N
!output (to screen): number of intervals, procs, estimated value of integral, and error
!To compile this code: mpif90 -o midpoint_p2.exe midpoint_p2.f90
!To run: mpiexec -n 2 midpoint_p2.exe
program midpoint_mpi2
    use mpi
    implicit none
    integer :: i1,N
    real(kind=8) :: pi, dx, sum_i, xm, sum_total, f, a, error, sum_proc

    integer :: myid, numprocs, ierr
    integer :: Nper_proc,istart,iend

    real(kind=8) :: tol !cf. Note #2 in header
    integer :: recompute

    pi = asin(1.d0)*2

    ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)



    !read data from data.in
	open(unit=10, file='data.in')
        read(10,*) N
        read(10,*) tol !cf. Note #2 in header
	close(10)

    sum_proc = 0.0 !initialize integral
    recompute = 0
    call decomposition(N,numprocs,myid,Nper_proc,istart,iend,dx) !cf. Note #1 in the header

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

        !*Task 2*: Check if the error is larger than the tolerance,
        !if it is, double N and set recompute=1
        if (error>tol) then
            N = N*2
            recompute = 1
        end if
    end if

        !*Task 3*:
        ! Broadcast "recompute" from P0 to the other processes.
        !*** call MPI_BCAST(...)
         if (recompute==1) then
        ! broadcast the new value of N,
        !*** call MPI_BCAST(...)
        ! recompute the decomposition and the integral. Output the new results.
            sum_proc=0.0
            call decomposition(N,numprocs,myid,Nper_proc,istart,iend,dx)
             !loop over intervals computing each interval's contribution to integral
            do i1 = istart,iend
                xm = dx*(i1-0.5d0) !midpoint of interval i1
                call integrand(xm,f)
                sum_i = dx*f
                sum_proc = sum_proc + sum_i !add contribution from interval to total integral
            end do
            call MPI_REDUCE(sum_proc,sum_total,1,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)

            !output
            if (myid == 0) then
                error = abs(sum_total - pi)
                print *, 'N=', N
                print *, 'sum=', sum_total
                print *, 'error=', error

            end if
        end if



    call MPI_FINALIZE(ierr)
end program midpoint_mpi2
!--------------------------------------------------------------------

!----------------------------------
!subroutine integrand
!   compute integrand, f = 4.0/(1+a^2)
!----------------------------------

subroutine integrand(a,f)
    implicit none
    real(kind=8), intent(in) :: a
    real(kind=8), intent(out) :: f
    f = 4.d0/(1.d0 + a*a)
end subroutine integrand

!--------------------------------------------------------------------

!----------------------------------
!subroutine decomposition
!   compute the number of points
!   per processor and starting
!   and ending points
!   input: N,numprocs,myid
!   output: Nper_proc,istart,iend,dx
!----------------------------------

subroutine decomposition(N,numprocs,myid,Nper_proc,istart,iend,dx)
    implicit none
    integer, intent(in) :: N,numprocs,myid
    integer, intent(out) :: Nper_proc,istart,iend
    real(kind=8), intent(out) :: dx

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

end subroutine decomposition
