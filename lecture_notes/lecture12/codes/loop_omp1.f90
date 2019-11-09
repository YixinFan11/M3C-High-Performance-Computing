!Simple example of OpenMP parallel loop
program loop_omp1
	use omp_lib !makes OpenMP routines, variables available
	implicit none
	integer :: NumThreads,threadID
    integer :: i1
    real(kind=8), dimension(4) :: x,y,z

    do i1=1,size(y)
        y(i1)=i1
    end do
    z = sin(y)

!$OMP parallel do private(threadID)
do i1 = 1,size(x)
    x(i1) = y(i1) + z(i1)
    threadID = omp_get_thread_num()
    print *, 'iteration ',i1,' assigned to thread ',threadID
end do
!$OMP end parallel do

print *, 'test:', maxval(abs(x-y-z))

end program loop_omp1




