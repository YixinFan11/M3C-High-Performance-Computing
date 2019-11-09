! sendExample.f90
!This code illustrates basic send/recv usage
! The i1th component of array1 is sent from process 0 to 1 and stored
!----------------------------------------------------------------------
program sendExample
    use mpi
    implicit none
	integer :: i1,j1,N
	double precision :: var1, var2
	double precision, dimension(5) :: array1
    integer :: myid, numprocs, ierr
    integer, dimension(MPI_STATUS_SIZE) :: status

    ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    array1 = 0.d0 !initialize array1
    N = size(array1)

    if (myid==0) then
    	i1 = 3 !component of array1 which will be sent
        call calculations(N,array1) !fill in array1
        call MPI_SEND(array1(i1),1,MPI_DOUBLE_PRECISION,1,i1,MPI_COMM_WORLD,ierr)

    elseif (myid==1) then
        call MPI_RECV(var1,1,MPI_DOUBLE_PRECISION,0,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
        j1 = status(MPI_TAG) !location where var1 will be stored in array1
        array1(j1) = var1
    end if

    print *, 'Proc', myid, 'array1=', array1


    call MPI_FINALIZE(ierr)
end program sendExample
!--------------------------------------------------------------------

!-----------------------
!subroutine calculations
!-----------------------
subroutine calculations(N,array)
    implicit none
    integer, intent(in) :: N
    double precision, dimension(N), intent(out) :: array
    integer :: i1
    double precision :: var1

    do i1 = 1,N !loop from 1 to N
            var1 = dble(i1) !convert integer to double precision number
            array(i1) = sin(var1)
    end do
end subroutine calculations

!--------------------------------------------------------------------
