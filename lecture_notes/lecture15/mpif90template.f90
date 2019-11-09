! Basic MPI + Fortran 90 code structure

!1. Header
program template
    use mpi
    implicit none

    !2a. Variable declarations (e.g. integers, real numbers,...)
    integer :: myid, numprocs, ierr

    !2b. Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    !3. basic code: input, loops, if-statements, subroutine calls
    print *, 'this is proc # ',myid, 'of ', numprocs


!4. End program
    call MPI_FINALIZE(ierr)
end program template

! To compile this code:
! $ mpif90 -o mpif90template.exe mpif90template.f90
! To run the resulting executable with 4 processes: $ mpiexec -n 4 mpif90template.exe