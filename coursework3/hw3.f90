! M3C 2018 Homework 3
! Tudor Trita Trita
! CID: 01199397
! Date: 29/11/2018

module tribes
  use omp_lib
  implicit none
  integer :: numthreads
  real(kind=8) :: tr_b,tr_e,tr_g
contains

subroutine simulate2_f90(n,nt,m,s,fc_ave)
  implicit none
  integer, intent(in) :: n,nt,m
  integer, intent(out), dimension(n,n,m) :: s
  real(kind=8), intent(out), dimension(nt+1) :: fc_ave
  integer :: i1,j1
  real(kind=8) :: n2inv
  integer, dimension(n,n,m) :: nb,nc
  integer, dimension(n+2,n+2,m) :: s2
  real(kind=8), dimension(n,n,m) :: f,p,a,pden,nbinv
  real(kind=8), dimension(n+2,n+2,m) :: f2,f2s2
  real(kind=8), allocatable, dimension(:,:,:) :: r !random numbers

  !---Problem setup----
  !Initialize arrays and problem parameters

  !initial condition
  s=1
  j1 = (n+1)/2
  s(j1,j1,:) = 0

  n2inv = 1.d0/dble(n*n)
  fc_ave(1) = sum(s)*(n2inv/m)

  s2 = 0
  f2 = 0.d0

  !Calculate number of neighbors for each point
  nb = 8
  nb(1,2:n-1,:) = 5
  nb(n,2:n-1,:) = 5
  nb(2:n-1,1,:) = 5
  nb(2:n-1,n,:) = 5
  nb(1,1,:) = 3
  nb(1,n,:) = 3
  nb(n,1,:) = 3
  nb(n,n,:) = 3

  nbinv = 1.d0/nb
  allocate(r(n,n,m))
  !---finished Problem setup---


  !----Time marching----
  do i1=1,nt

     !Random numbers used to update s every time step
     call random_number(r) !Random numbers used to update s every time step
    !Set up coefficients for fitness calculation in matrix, a
    a = 1
    where(s==0)
      a=tr_b
    end where

    !create s2 by adding boundary of zeros to s
    s2(2:n+1,2:n+1,:) = s

    !Count number of C neighbors for each point
    nc = s2(1:n,1:n,:) + s2(1:n,2:n+1,:) + s2(1:n,3:n+2,:) + &
         s2(2:n+1,1:n,:)                  + s2(2:n+1,3:n+2,:) + &
         s2(3:n+2,1:n,:)   + s2(3:n+2,2:n+1,:)   + s2(3:n+2,3:n+2,:)

    !Calculate fitness matrix, f----
    f = nc*a
    where(s==0)
      f = f + (nb-nc)*tr_e
    end where
    f = f*nbinv
    !-----------

    !Calculate probability matrix, p----
    f2(2:n+1,2:n+1,:) = f
    f2s2 = f2*s2

    !Total fitness of cooperators in community
    p = f2s2(1:n,1:n,:) + f2s2(1:n,2:n+1,:) + f2s2(1:n,3:n+2,:) + &
           f2s2(2:n+1,1:n,:) + f2s2(2:n+1,2:n+1,:)  + f2s2(2:n+1,3:n+2,:) + &
          f2s2(3:n+2,1:n,:)   + f2s2(3:n+2,2:n+1,:)   + f2s2(3:n+2,3:n+2,:)

    !Total fitness of all members of community
    pden = f2(1:n,1:n,:) + f2(1:n,2:n+1,:) + f2(1:n,3:n+2,:) + &
           f2(2:n+1,1:n,:) + f2(2:n+1,2:n+1,:)  + f2(2:n+1,3:n+2,:) + &
          f2(3:n+2,1:n,:)   + f2(3:n+2,2:n+1,:)   + f2(3:n+2,3:n+2,:)


    p = (p/pden)*tr_g + 0.5d0*(1.d0-tr_g) !probability matrix
    !----------

    !Set new affiliations based on probability matrix and random numbers stored in R
    s = 0
    where (R<=p)
        s = 1
    end where

    fc_ave(i1+1) = sum(s)*(n2inv/m)

  end do
end subroutine simulate2_f90

SUBROUTINE simulate2_omp(n,nt,m,s,fc_ave)
  !My approach to parallelization was to parallelize whenever possible on
  !variables which did not depend on each other.
  !I first parallelized the initialization of fc_ave.
  !Then I decided to create a new parallel 1,M loop which would load values into
  !the S matrix. I chose to make the parallel loop outside of the 1,Nt loop as to
  !avoid repeated instances of creating parallel regions.
  !To make calculations possible, I had to make most of the variables inside the loop
  !private as to not create conflicts between threads, where multiple threads would
  !change values of the same variables at the same time.
  !Regarding the inefficiencies in memory usage of the routine simulate2_f90, I
  !fixed these by making the matrices nb,nc,s2,nbinv,a,f,p,pden 2D matrices.
  !Instead of storing values in the (N,N,m) matrices, I chose to update the values
  !at each timestep keeping them private. This meant that when m was large,
  !(say over 1000), the program would run and not crash, unlike the Non-OMP version.
  !This results in a massive freeup in memory when executing the program, thus increased
  !efficiency in the results.
  implicit none
  integer, intent(in) :: n,nt,m
  integer, intent(out), dimension(n,n,m) :: s
  real(kind=8), intent(out), dimension(nt+1) :: fc_ave
  integer :: i1,i2,j1
  integer, dimension(n,n) :: nb, nc !neighbor matrices
  integer, dimension(n+2,n+2) :: s2
  real(kind=8), allocatable, dimension(:,:) :: r !random numbers
  real(kind=8) :: n2inv
  real(kind=8), dimension(n,n) :: nbinv, a, f, p, pden
  real(kind=8), dimension(n+2,n+2) :: f2,f2s2

  !initial condition and r allocation
  s=1
  j1 = (n+1)/2
  s(j1,j1,:) = 0
  ALLOCATE(r(n,n))

  n2inv = 1.d0/dble(n*n)
  !$OMP PARALLEL DO
  DO I1 = 1,NT
    fc_ave(I1) = 0
  END DO
  !$OMP END PARALLEL DO
  fc_ave(1) = sum(s)*(n2inv/m)
  CALL omp_set_num_threads(numthreads) !Set number of threads to be used

  !Setting up neighbours
  nb = 8
  nb(1,2:n-1) = 5
  nb(n,2:n-1) = 5
  nb(2:n-1,1) = 5
  nb(2:n-1,n) = 5
  nb(1,1) = 3
  nb(1,n) = 3
  nb(n,1) = 3
  nb(n,n) = 3
  nbinv = 1.d0/nb

  !Main loop: Looping over m first to avoid uneccesary OMP commands
  !$OMP PARALLEL DO PRIVATE(R, A, I1, S2, NC, F, F2, F2S2, P, PDEN) REDUCTION(+:fc_ave)
  DO i2 = 1,m
    s2 = 0 !Resetting S2
    f2 = 0.d0 !Resetting F2
    DO i1 = 1,nt
         CALL RANDOM_NUMBER(R) !Assigning random numbers at each timestep

         a = 1.d0
         WHERE(s(:,:,i2) == 0)
           a = tr_b
         END WHERE

         s2(2:n+1,2:n+1) = s(:,:,i2) 

         nc = s2(1:n,1:n) + s2(1:n,2:n+1) + s2(1:n,3:n+2) + &
                      s2(2:n+1,1:n) + s2(2:n+1,3:n+2)+&
                      s2(3:n+2,1:n)+s2(3:n+2,2:n+1)+s2(3:n+2,3:n+2)

         f = dble(nc)*a
         WHERE(s(:,:,i2)==0)
             f = f + (nb-nc)*tr_e
         END WHERE
         f = f*nbinv

         f2(2:n+1,2:n+1) = f
         f2s2 = f2*s2

         p = f2s2(1:n,1:n) + f2s2(1:n,2:n+1) + f2s2(1:n,3:n+2) + &
                     f2s2(2:n+1,1:n) + f2s2(2:n+1,2:n+1) + f2s2(2:n+1,3:n+2) + &
                     f2s2(3:n+2,1:n) + f2s2(3:n+2,2:n+1) + f2s2(3:n+2,3:n+2)

         pden = f2(1:n,1:n) + f2(1:n,2:n+1) + f2(1:n,3:n+2) + &
                        f2(2:n+1,1:n) + f2(2:n+1,2:n+1) + f2(2:n+1,3:n+2) + &
                        f2(3:n+2,1:n) + f2(3:n+2,2:n+1) + f2(3:n+2,3:n+2)

         p = (p/pden)*tr_g + 0.5d0*(1.d0-tr_g)

         s(:,:,i2) = 0 !Updating S based on probabilities.
         WHERE (R <= p)
             s(:,:,i2) = 1
         END WHERE

         fc_ave(i1+1) = fc_ave(i1+1) + sum(s(:,:,i2))*(n2inv/m) !Updating fc_ave at each timestep for every m

      END DO
    END DO
  !$OMP END PARALLEL DO
  DEALLOCATE(r) !Frees up memory
END SUBROUTINE simulate2_omp
END MODULE tribes
