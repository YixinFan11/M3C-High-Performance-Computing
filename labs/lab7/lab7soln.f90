!lab7 solution code, parallel matrix multiplication
!To compile: gfortran -fopenmp -o test.exe lab7soln.f90
!To run: ./test.exe
!When using (M,N)=(200,400), running on two threads will probably be slower
!than running with one. However, when (M,N)=(2000,4000), you should see
!a speedup of about 2, i.e., the code will run about twice as fast when
!using two threads.

program lab7
    use omp_lib
    implicit none
    integer :: i1,j1,M,N,numThreads
    integer(kind=8) :: t1,t2,rate !timer variables
    real(kind=8) :: Csum
    real(kind=8), allocatable, dimension(:) :: temp
    real(kind=8), allocatable, dimension(:,:) :: A,B,C


    !read in problem parameters
    open(unit=10,file='data.in')
        read(10,*) M
        read(10,*) N
    close(10)


    !initialize variables
    allocate(A(M,N),B(N,M),C(M,M),temp(M))


      call random_number(A)
      call random_number(B)

      call system_clock(t1)

Csum = 0.d0
!Task 1, 2
!$OMP parallel do reduction(+:Csum)
do i1=1,M
	C(:,i1) = matmul(A,B(:,i1))
	Csum = Csum + sum(C(:,i1))
end do
!$OMP end parallel do

call system_clock(t2,rate)

print *, Csum,sum(C)
print *, 'test:',maxval(abs(C-matmul(A,B)))
print *, 'test2:',(Csum-sum(C))/Csum
print *, 'wall time:',float(t2-t1)/float(rate)





end program lab7
