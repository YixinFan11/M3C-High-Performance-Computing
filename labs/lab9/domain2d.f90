! Very simple example illustrating creation of 2-D process grid
! and construction of domain decomposition using process grid.
! To compile: mpif90 -o d2d.exe domain2d.f90
! To run with 4 processes: mpiexec -n 4 --hostfile hostfile.txt d2d.exe
! Four or more processes should be used in the mpiexec command.
!----------------------------------------------------------------------
program domain2d
    use mpi
	implicit none
	integer :: i1,j1,Ntotalx,Ntotaly,Nlocalx,Nlocaly,istart,iend,jstart,jend !note we now have jstart and jend for the 2d problem
    double precision :: dx,buffer,pi
    double precision, allocatable, dimension(:) :: x,f,df,exact
    integer :: myid, numprocs, ierr, sender, receiver
    integer, dimension(MPI_STATUS_SIZE) :: status

    !Variables needed for process grid
    integer :: new_comm, newid, displace,direction
    integer, parameter :: dim = 2 !2-dimensional process grid
    integer, dimension(dim) :: proc_dim, mycoords, spat_dim
    logical, dimension(dim) :: boundary

    pi = acos(-1.d0)

    ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)


	!read data from data.in
	open(unit=10, file='data.in')
        read(10,*) Ntotalx
        read(10,*) Ntotaly
	close(10)

    !----------------------------------------------------------
    !Create 2-D process grid for numprocs
    !----------------------------------------------------------
    proc_dim = 0

    call MPI_dims_create(numprocs,dim,proc_dim,ierr)
    if (myid==0) print *, 'process grid dimensions', proc_dim


    !open boundary conditions in first dimension, periodic in second
    boundary(1) = .false.
    boundary(2) = .true.
    call MPI_cart_create(MPI_COMM_WORLD, dim, proc_dim, boundary,.false.,new_comm,ierr) !creates new communicator, new_comm


    !get new myid from new communicator
    call MPI_comm_rank(new_comm,newid,ierr)

    !get grid coordinates for process
    call MPI_cart_coords(new_comm,newid,dim,mycoords,ierr)

    print *, 'newid=',newid, 'my coordinates=', mycoords

    call MPI_BARRIER(new_comm,ierr)

    !----------------------------------------------------------
    !Create 2-D domain decomposition using process coordinates
    !----------------------------------------------------------

    !---Construct decomposition in y---
    call MPE_DECOMP1D( Ntotaly, proc_dim(1), mycoords(1), istart, iend)
    Nlocaly = iend - istart + 1


    !---Construct decomposition in x---
    call MPE_DECOMP1D( Ntotalx, proc_dim(2), mycoords(2), jstart, jend)
    Nlocalx = jend - jstart + 1


    if (newid==0) print *, '-------------- 2D domain decomposition----------------'
    call MPI_BARRIER(new_comm,ierr)
    print *, 'proc #',newid, 'has been assigned istart,iend=', istart,iend, 'and jstart,jend=',jstart,jend

    call MPI_BARRIER(new_comm,ierr)

    !------------------------------------------------------------------------------------------
    !Obtain id's of neighboring process for moving data to the "left" (in horizontal direction)
    !------------------------------------------------------------------------------------------

    displace = -1 !can be +/- 1
    direction = 1 ! can be 0 or 1
    call MPI_cart_shift(new_comm,direction,displace,sender,receiver,ierr)

    if (newid==0) print *, '-------------- Find processes to the left ----------------'
    if (newid==0) print *,'***neighboring process, direction=',direction,'displace=',displace
    call MPI_BARRIER(new_comm,ierr)

    print *, 'proc #',newid, 'sender=',sender,'receiver=',receiver

    call MPI_BARRIER(new_comm,ierr)

    !-----------------------------------------------------------------------------------
    !Obtain id's of neighboring process  for moving data "above" (in vertical direction)
    !------------------------------------------------------------------------------------
    !****COMPLETE FOR LAB 9*****
    !  displace = ?
    !  direction = ?
    if (1==2) then !change this when needed
        call MPI_cart_shift(new_comm,direction,displace,sender,receiver,ierr)

        if (newid==0) print *, '-------------- Find processes above ----------------'
        if (newid==0) print *,'***neighboring process, direction=',direction,'displace=',displace
        call MPI_BARRIER(new_comm,ierr)

        print *, 'proc #',newid, 'sender=',sender,'receiver=',receiver
    end if

    !-----------------------------------------------------------------------------------------------------
    ! Exchange boundary data:
    ! send and receive statements would be placed here as usual with the sender and receiver ids
    !-----------------------------------------------------------------------------------------------------



    call MPI_FINALIZE(ierr)

end program domain2d
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
    double precision, dimension(Nlocal), intent(out) :: x

    do i1 = istart,iend
        x(i1-istart+1) = dble(i1-1)/dble(Ntotal)
    end do

end subroutine make_grid


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





