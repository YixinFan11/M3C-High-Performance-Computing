! This code illustrates differentation of data distributed across subdomains
! Code outline:
! 1. Initialize MPI
! 2. Read Ntotal from data.in
! 3. Construct domain decomposition
! 4. Make grid and field, f=sin(2*pi*x), in the local subdomain
! 5. Compute derivative ***Complete this part for Exercise***
! 6. Output error
!To compile this code: mpif90 -o gradient_p.exe gradient_p.f90
!To run: mpiexec -n 2 gradient_p.exe
!----------------------------------------------------------------------
program gradient
    use mpi
	implicit none
	integer :: i1,j1,Ntotal,Nlocal,istart,iend
    real(kind=8) :: dx,buffer
    real(kind=8), allocatable, dimension(:) :: x,f,df,exact
    integer :: myid, numprocs, ierr, sender, receiver, request
    integer, dimension(MPI_STATUS_SIZE) :: status
    real(kind=8) :: pi

    pi = acos(-1.d0)

    ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)


	!read data from data.in
	open(unit=10, file='data.in')
        read(10,*) Ntotal
	close(10)

    !construct decomposition: assign Nlocal points from istart to iend  to each processor
    call MPE_DECOMP1D( Ntotal, numprocs, myid, istart, iend)
    Nlocal = iend - istart + 1
    allocate(x(Nlocal),f(Nlocal+2),df(Nlocal),exact(Nlocal)) !set dimensions of x,f,df

    !make grid and field
    call make_grid(Ntotal,Nlocal,istart,iend,x)
    dx = x(2)-x(1)
    print *, 'proc', myid, ' has been assigned the interval x=', x(1),x(Nlocal)

    call make_field(Nlocal,x,f(2:Nlocal+1)) !note: f(1) and f(Nlocal+2) must be obtained from neighboring processors

!Now, compute derivative, df/dx
!-----------------------------------------------------------
!Send data at top boundary up to next processor
!i.e. send f(nlocal+1) to myid+1 and store it there as f(1)
!data from myid=numprocs-1 is sent to myid=0
!-----------------------------------------------------------
    if (myid<numprocs-1) then
        receiver = myid+1
    else
         receiver = 0
    end if

    if (myid>0) then
        sender = myid-1
    else
        sender = numprocs-1
    end if

    call MPI_ISEND(f(Nlocal+1),1,MPI_DOUBLE_PRECISION,receiver,0,MPI_COMM_WORLD,request,ierr)
    call MPI_RECV(f(1),1,MPI_DOUBLE_PRECISION,sender,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)

    call MPI_BARRIER(MPI_COMM_WORLD,ierr)

!-----------------------------------------------------------
!Send data at bottom boundary down to previous processor
!i.e. send f(2) to myid-1 and store it there as f(nlocal+2)
!data from myid=0 is sent to myid=numprocs-1
!-----------------------------------------------------------
    if (myid>0) then
        !EX: receiver = ?
    else
        !EX: receiver = ?
    end if

    if (myid<numprocs-1) then
        !EX: sender = ?
    else
        !EX: sender = ?
    end if
    call MPI_ISEND(**EX4**,1,MPI_DOUBLE_PRECISION,receiver,0,MPI_COMM_WORLD,request,ierr)
    call MPI_RECV(**EX4**,1,MPI_DOUBLE_PRECISION,sender,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
!----finished sending/receiving
    df = (f(3:Nlocal+2) - f(1:Nlocal))/(2.d0*dx) !approximate derivative

    exact = 2.d0*pi*cos(2.d0*pi*x)

    print *, 'myid=', myid, 'error=', maxval(abs(exact-df))

    call MPI_FINALIZE(ierr)

end program gradient
!--------------------------------------------------------------------
!--------------------------------------------------------------------
!--------------------------------------------------------------------
!--------------------------------------------------------------------
!---------------------------------------------------------------------
!subroutine make_grid
!
!Let xtotal be a uniformly spaced grid from 0 to 1 with Ntotal+1 points
!This subroutine generates x = xtotal(istart:iend)
!---------------------------------------------------------------------
subroutine make_grid(Ntotal,Nlocal,istart,iend,x)
    implicit none
    integer :: i1
    integer, intent(in) :: Ntotal,Nlocal,istart,iend
    real(kind=8), dimension(Nlocal), intent(out) :: x

    do i1 = istart,iend
        x(i1-istart+1) = dble(i1-1)/dble(Ntotal)
    end do

end subroutine make_grid
!--------------------------------------------------------------------

!---------------------------------------------------------------------
!subroutine make_field
!generates a simple sinusoidal function, f(X)
!input: N, size of field
!       x, grid points
!---------------------------------------------------------------------
subroutine make_field(N,x,f)
    implicit none
    integer, intent(in) :: N
    real(kind=8), dimension(N), intent(in) :: x
    real(kind=8), dimension(N), intent(out) :: f
    real(kind=8) :: pi

    pi = acos(-1.d0)
    f = sin(2.d0*pi*x)

end subroutine make_field

!--------------------------------------------------------------------
!  (C) 2001 by Argonne National Laboratory.
!      See COPYRIGHT in online MPE documentation.
!  This file contains a routine for producing a decomposition of a 1-d array
!  when given a number of processors.  It may be used in "direct" product
!  decomposition.  The values returned assume a "global" domain in [1:n]
!
subroutine MPE_DECOMP1D( n, numprocs, myid, s, e )
    implicit none
    integer :: n, numprocs, myid, s, e
    integer :: nlocal
    integer :: deficit

    nlocal  = n / numprocs
    s       = myid * nlocal + 1
    deficit = mod(n,numprocs)
    s       = s + min(myid,deficit)
    if (myid .lt. deficit) then
        nlocal = nlocal + 1
    endif
    e = s + nlocal - 1
    if (e .gt. n .or. myid .eq. numprocs-1) e = n

end subroutine MPE_DECOMP1D
