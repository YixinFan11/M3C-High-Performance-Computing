!example illustrating use of lapack routine, dgesv, to solve AX=B
program dgesv_test
    implicit none
!declare variables
    integer :: N,NRHS,LDA,LDB,INFO
    real(kind=8), allocatable, dimension(:,:) :: A,B,Atemp,Btemp,x
    integer, allocatable, dimension(:) :: IPIV



    !set dimensions of A,B
    N = 4
    NRHS = 1
    LDA = N
    LDB = N

    !allocate arrays, create matrices A and B
    allocate(A(LDA,N),B(LDB,NRHS),IPIV(N))
    allocate(Atemp(size(A,1),size(A,2)),Btemp(LDB,NRHS))
    A(1,:) = (/1.0,2.0,3.0,4.0/)
    A(2,:) = (/4.0,3.0,2.0,1.0/)
    A(3,:) = A(1,:)**2
    A(4,:) = sqrt(A(2,:))

    B(:,1) = (/-2.0,2.0,-1.0,1.0/)

    !solve Ax = B
    Atemp = A
    Btemp = B
    call dgesv(N, NRHS, Atemp, LDA, IPIV, Btemp, LDB, INFO)
    print *, 'INFO=',INFO

    !extract soln from Btemp
    allocate(x(N,NRHS))
    x = Btemp(1:N,:)

    print *, 'A=',A
    print *, 'x=',x
    print *, 'b=',b
    print *, 'test:',matmul(A,x)-B

end program dgesv_test


