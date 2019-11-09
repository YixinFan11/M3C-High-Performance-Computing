!Simple example of OpenMP parallel loop
program norm_omp1
	use omp_lib !makes OpenMP routines, variables available
	implicit none
	integer :: NumThreads,threadID
    integer :: i1
    real(kind=8) :: norm,partial_norm
    real(kind=8), dimension(4) :: x,y,z

    do i1=1,size(y)
        y(i1)=i1
    end do
    z = sin(y)
    x = z+y
norm=0.d0
partial_norm=0.d0

!$OMP parallel firstprivate(partial_norm),private(threadID)
!$OMP do
do i1 = 1,size(x)
    partial_norm = partial_norm + abs(x(i1))
end do
!$OMP end do

!$OMP critical
threadID = omp_get_thread_num()
print *, 'Thread number:',threadID, 'partial norm=',partial_norm
norm = norm + partial_norm
!$OMP end critical


!$OMP end parallel

print *, 'test:',norm-sum(abs(x))


end program norm_omp1
