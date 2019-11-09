!Simple example of OpenMP parallel loop with reduction
program norm_omp2
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

!$OMP parallel do reduction(+:norm)
do i1 = 1,size(x)
    norm = norm + abs(x(i1))
end do
!$OMP end parallel do

print *, 'test:',norm-sum(abs(x))


end program norm_omp2




