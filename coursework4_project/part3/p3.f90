! Tudor Trita Trita
!CID: 01199397

!Part 3 Question 2:
!One key advantage of using m^2 squares is that total communication between processors
!would be reduced, as there are less border points between processes,
!thus the speed of the program may be increased.

!One key disadvantage of using m^2 squares is that it would be more difficult to implement
!as a process may have to communicate with more processes than the other method, e.g.
!if a process is in the 'middle' of a grid, it may have to communicate with 4 or 8 other
!processes, which makes it more difficult to implement, thus increasing developing time.

program part3_mpi
    use mpi
    implicit none

    integer :: n! number of grid points (corresponds to (n+2 x n+2) grid)
    integer :: kmax !max number of iterations
    real(kind=8) :: tol, g !convergence tolerance, death rate
    real(kind=8), allocatable, dimension(:) :: deltac
    real(kind=8), allocatable, dimension(:,:) :: C !concentration matrix
    real(kind=8) :: S0,r0,t0 !source parameters
    real(kind=8) :: k_bc !r=1 boundary condition parameter
    integer :: i1,j1
    integer :: myid, numprocs, ierr

 ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

!gather input
    open(unit=10,file='data.in')
        read(10,*) n
        read(10,*) kmax
        read(10,*) tol
        read(10,*) g
        read(10,*) S0
        read(10,*) r0
        read(10,*) t0
        read(10,*) k_bc
   close(10)

    allocate(C(0:n+1,0:n+1),deltaC(kmax))     !C IS A N+2 BY N+2 MATRIX

!compute solution
    call simulate_mpi(MPI_COMM_WORLD,numprocs,n,kmax,tol,g,S0,r0,t0,k_bc,C,deltaC) !ONLY THING TO CONSIDER: OUTPUT C AND DELTAC

!output solution (after completion of gather in euler_mpi)
     call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      if (myid==0) then
        open(unit=11,file='C.dat')
        do i1=0,n+1
            write(11,('(1000E28.16)')) C(i1,:)
        end do
        close(11)

        open(unit=12,file='deltac.dat')
        do i1=1,kmax
          write(12,*) deltac(i1)
        end do
        close(12)
      end if
    !can be loaded in python with: c=np.loadtxt('C.dat')

    call MPI_FINALIZE(ierr)
end program part3_mpi


!Simulate contamination model with AOS iteration
subroutine simulate_mpi(comm,numprocs,n,kmax,tol,g,S0,r0,t0,k_bc,C,deltaC)
    use mpi
    implicit none
    integer, intent (in) :: comm,numprocs,n,kmax
    real(kind=8), intent(in) ::tol,g,S0,r0,t0,k_bc
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    real(kind=8), intent(out) :: deltac(kmax)
    real(kind=8) :: deltac2(kmax)
    integer :: i1,i2,j0,j1,i,j,k,istart,iend
    integer :: myid,ierr,nl !nlocal = nl
    real(kind=8) :: t1d(0:n+1),del,pi
    real(kind=8), allocatable, dimension(:) :: r1d,Cprev,Cafter,Cnprev,Cnafter
    real(kind=8), allocatable, dimension(:,:) :: r,t,Clocal,Cnlocal,rinv2
    real(kind=8), allocatable, dimension(:,:) :: fac, facp, facm, fac2, Sdel2local
    integer, dimension(numprocs) :: disps,Nper_proc

    call MPI_COMM_RANK(comm, myid, ierr)
    print *, 'start simulate_mpi, myid=',myid

    !Set up theta
    pi = acos(-1.d0)
    del = pi/dble(n+1)
    do i1 = 0,n+1
      t1d(i1) = i1*del
    end do

    !generate decomposition and allocate sub-domain variables
    call MPE_DECOMP1D(n,numprocs,myid,istart,iend)
    print *, 'istart,iend,threadID=',istart,iend,myid
    nl = iend-istart+1
    allocate(r1d(nl+2))

    do i1=istart-1,iend+1
      r1d(i1-istart+2) = 1.d0 + i1*del
    end do

    allocate(r(0:nl+1,0:n+1),t(0:nl+1,0:n+1),Clocal(0:nl+1,0:n+1),Cnlocal(0:nl+1,0:n+1))

    do j1=0,n+1
      r(:,j1) = r1d
    end do
    do i1=0,nl+1
      t(i1,:) = t1d
    end do

	  !Allocating factors used in update equation:
    allocate(rinv2(0:nl+1,0:n+1),fac(0:nl+1,0:n+1),facp(0:nl+1,0:n+1))
    allocate(facm(0:nl+1,0:n+1),fac2(0:nl+1,0:n+1),Sdel2local(0:nl+1,0:n+1))

	  !Computing local factors used in update equation:
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*g)
    facp = (1.d0 + 0.5d0*del/r)*fac
    facm = (1.d0 - 0.5d0*del/r)*fac
    fac2 = fac*rinv2

    ! Computing Clocal and Cnlocal for each process
    Clocal = (sin(k_bc*t)**2)*(pi+1.d0-r)/pi
    Cnlocal = (sin(k_bc*t)**2)*(pi+1.d0-r)/pi

    ! Computing Source function
    Sdel2local = s0*exp(-20.d0*((r-r0)**2+(t-t0)**2))*(del**2)*fac

    ! Allocating vector Cprev and Cafter to be used to send information to every other processor
    allocate(Cprev(0:n+1),Cafter(0:n+1),Cnprev(0:n+1),Cnafter(0:n+1))

	!AOS iteration---------------------------------------------------------------

    do k = 1,kmax
      !Communicating boundary information
	    !First need to make sure to send border points over to other processes.
      Cprev = Clocal(nl,:)
      Cafter = Clocal(1,:)
      if (myid == 0) then
          !Send information of C_i-1,: to process 1
          call MPI_SEND(Cprev,n+2,MPI_DOUBLE_PRECISION,myid+1,0,COMM,ierr)
      elseif((myid .LT. (numprocs-1)) .AND. (myid .NE. 0))   then
          !Send information to previous and after processor about C
          call MPI_SEND(Cprev,n+2,MPI_DOUBLE_PRECISION,myid+1,0,COMM,ierr)
          call MPI_SEND(Cafter,n+2,MPI_DOUBLE_PRECISION,myid-1,1,COMM,ierr)
      elseif(myid == (numprocs-1)) then
          !Send information to second last processor about C_i+1,j
          call MPI_SEND(Cafter,n+2,MPI_DOUBLE_PRECISION,myid-1,1,COMM,ierr)
      end if

      call MPI_BARRIER(comm,ierr) !Needs to be syncronised before receiving information

      !Receiving information and setting local information with it.
      if (myid == 0) then
          call MPI_RECV(Cafter,n+2,MPI_DOUBLE_PRECISION,myid+1,1,comm,MPI_STATUS_IGNORE,ierr)
          Clocal(nl+1,:) = Cafter
	     elseif((myid .LT. (numprocs-1)) .AND. (myid .NE. 0))   then
          call MPI_RECV(Cprev,n+2,MPI_DOUBLE_PRECISION,myid-1,0,COMM,MPI_STATUS_IGNORE,ierr)
          call MPI_RECV(Cafter,n+2,MPI_DOUBLE_PRECISION,myid+1,1,COMM,MPI_STATUS_IGNORE,ierr)
          Clocal(0,:) = Cprev
          Clocal(nl+1,:) = Cafter
      elseif(myid == (numprocs-1)) then
          call MPI_RECV(Cprev,n+2,MPI_DOUBLE_PRECISION,myid-1,0,COMM,MPI_STATUS_IGNORE,ierr)
          Clocal(0,:) = Cprev
      end if

      !White update:
      do i=1,nl
        if (mod(i+istart-1,2) == 1) then
          do j=1,n,2
          Cnlocal(i,j) = 0.5d0*(-Clocal(i,j) + 3.d0*(facm(i,j)*Clocal(i-1,j) + &
                              fac2(i,j)*(Clocal(i,j-1)+Clocal(i,j+1)) + &
                              facp(i,j)*Clocal(i+1,j) + Sdel2local(i,j)))
          end do
        elseif (mod(i+istart-1,2) == 0) then
          do j=2,n,2
          Cnlocal(i,j) = 0.5d0*(-Clocal(i,j) + 3.d0*(facm(i,j)*Clocal(i-1,j) + &
                              fac2(i,j)*(Clocal(i,j-1)+Clocal(i,j+1)) + &
                              facp(i,j)*Clocal(i+1,j) + Sdel2local(i,j)))
          end do
        end if
      end do

	  !Sending newly computed Cnlocal boundary points to other processes.
    Cnprev = Cnlocal(nl,:)
    Cnafter = Cnlocal(1,:)
      if (myid == 0) then
          call MPI_SEND(Cnprev,n+2,MPI_DOUBLE_PRECISION,myid+1,0,COMM,ierr)
      elseif((myid .LT. (numprocs-1)) .AND. (myid .NE. 0))   then
          call MPI_SEND(Cnprev,n+2,MPI_DOUBLE_PRECISION,myid+1,0,COMM,ierr)
          call MPI_SEND(Cnafter,n+2,MPI_DOUBLE_PRECISION,myid-1,1,COMM,ierr)
      elseif(myid == (numprocs-1)) then
          call MPI_SEND(Cnafter,n+2,MPI_DOUBLE_PRECISION,myid-1,1,COMM,ierr)
      end if

	  call MPI_BARRIER(comm,ierr) !Needs to be syncronised before receiving information

      !Receiving Information
      if (myid == 0) then
          call MPI_RECV(Cnafter,n+2,MPI_DOUBLE_PRECISION,myid+1,1,COMM,MPI_STATUS_IGNORE,ierr)
          Cnlocal(nl+1,:) = Cnafter
      elseif((myid .LT. (numprocs-1)) .AND. (myid .NE. 0))   then
          call MPI_RECV(Cnprev,n+2,MPI_DOUBLE_PRECISION,myid-1,0,COMM,MPI_STATUS_IGNORE,ierr)
          call MPI_RECV(Cnafter,n+2,MPI_DOUBLE_PRECISION,myid+1,1,COMM,MPI_STATUS_IGNORE,ierr)
          Cnlocal(0,:)=Cnprev
          Cnlocal(nl+1,:) = Cnafter
      elseif(myid == (numprocs-1)) then
          call MPI_RECV(Cnprev,n+2,MPI_DOUBLE_PRECISION,myid-1,0,COMM,MPI_STATUS_IGNORE,ierr)
          Cnlocal(0,:) = Cnprev
      end if

      !Black update
      do i = 1,nl
          if (mod(i+istart-1,2) == 1) then
              do j=2,n,2
                Cnlocal(i,j) = 0.5d0*(-Clocal(i,j) + 3.d0*(facm(i,j)*Cnlocal(i-1,j) + &
                            fac2(i,j)*(Cnlocal(i,j-1)+Cnlocal(i,j+1)) + &
                            facp(i,j)*Cnlocal(i+1,j) + Sdel2local(i,j)))
              end do
          elseif (mod(i+istart-1,2) == 0) then
              do j=1,n,2
                Cnlocal(i,j) = 0.5d0*(-Clocal(i,j) + 3.d0*(facm(i,j)*Cnlocal(i-1,j) + &
                            fac2(i,j)*(Cnlocal(i,j-1)+Cnlocal(i,j+1)) + &
                            facp(i,j)*Cnlocal(i+1,j) + Sdel2local(i,j)))
              end do
          end if
      end do

    !Computing change in C on each process
    deltac2(k) = maxval(abs(Cnlocal(1:nl,1:n)-Clocal(1:nl,1:n)))
    Clocal(1:nl,1:n)=Cnlocal(1:nl,1:n)  !Assigning new C to old C

		call MPI_BARRIER(comm,ierr)
    !Sending maximum of deltac(k) to every thread:
		call MPI_ALLREDUCE(deltac2(k),deltac(k),1,MPI_DOUBLE_PRECISION,MPI_MAX,comm,ierr)

		! Check final convergence on process 0
    if (deltac(k)<tol) exit

		if (myid == 0) then
			if (mod(k,1000)==0) print *, k,deltac(k), 'max deltac on all processes'
    end if
	end do

    !---------------------------------------------------------
    call MPI_BARRIER(comm,ierr)

    if (myid == 0) print*, 'final k = ', k
    !Code below constructs C from the Clocal on each process
    print *, 'Before collection',myid, maxval(abs(Clocal))

    i1= 1
    i2 = nl

    if (myid==0) then
      i1=0
      nl = nl + 1
    elseif (myid==numprocs-1) then
      i2 = nl + 1
      nl = nl+1
    end if

    call MPI_GATHER(nl,1,MPI_INT,NPer_proc,1,MPI_INT,0,comm,ierr)
    !collect Clocal from each processor onto myid=0

    if (myid==0) then
        disps(1)=0
        do j1=2,numprocs
          disps(j1) = disps(j1-1) + Nper_proc(j1-1)*(n+2)
        end do

        print *, 'nper_proc=',NPer_proc
        print *, 'disps=',disps
    end if

  !collect Clocal from each processor onto myid=0

     call MPI_GATHERV(transpose(Clocal(i1:i2,:)),nl*(n+2),MPI_DOUBLE,C,Nper_proc*(n+2), &
                 disps,MPI_DOUBLE,0,comm,ierr)

      C = transpose(C)
    if (myid==0) print *, 'finished',maxval(abs(C)),sum(C)
end subroutine simulate_mpi

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
    integer :: nl
    integer :: deficit

    nl  = n / numprocs
    s       = myid * nl + 1
    deficit = mod(n,numprocs)
    s       = s + min(myid,deficit)
    if (myid .lt. deficit) then
        nl = nl + 1
    endif
    e = s + nl - 1
    if (e .gt. n .or. myid .eq. numprocs-1) e = n

end subroutine MPE_DECOMP1D
