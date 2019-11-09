!M3C 2018 Homework 2
!Name: Tudor Mihai Trita Trita
!CID: 01199397
!TID: 167
!Course: Mathematics G103 Year 3

module nmodel
  implicit none
  real(kind=8), allocatable, dimension(:,:) :: nm_x
  integer, allocatable, dimension(:) :: nm_y

contains

subroutine data_init(n,d)
  implicit none
  integer, intent(in) :: n,d
  if (allocated(nm_x)) deallocate(nm_x)
  if (allocated(nm_y)) deallocate(nm_y)
  allocate(nm_x(n,d),nm_y(d))
end subroutine data_init


subroutine snmodel(fvec,n,d,c,cgrad)
  implicit none
  integer, intent(in) :: n,d !training data sizes
  real(kind=8), dimension(n+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(n+1), intent(out) :: cgrad !gradient of cost
  !Declare other variables as needed
  real(kind=8) :: dinv,dcdb
  real(kind=8),dimension(d) :: z,a,e,gamma,dadb,eg
  real(kind=8), dimension(n) :: dcdw

  dinv = 1.d0/dble(d)

  !Compute inner layer activation
  z = matmul(fvec(1:n),nm_x) + fvec(n+1)
  a = 1.d0/(1.d0+exp(-z))

  !Compute cost
  gamma = a*(1.d0-a)
  e = a-nm_y
  c = 0.5d0*dinv*sum(e**2)

  !Compute gradient of cost
  eg = e*gamma
  cgrad(n+1) = dinv*sum(eg) !dcdb
  cgrad(1:n) = dinv*matmul(nm_x,eg) !dcdw

end subroutine snmodel

!Following subroutine is used to find activation data for snmodel, to compute test_error
subroutine tst(fvec,X_test,y_test,n,d,terror)
  implicit none
  integer, intent(in) :: n, d
  real(kind=8), dimension(n+1), intent(in) :: fvec
  real(kind=8), dimension(n,d), intent(in) :: X_test
  integer, dimension(d), intent(in) :: y_test
  real(kind=8), intent(out) :: terror
  integer :: i1, ncct
  real(kind=8), dimension(d) :: z, activ
  ncct = 0
  do i1 = 1,d
    z(i1) = sum(fvec(1:n)*X_test(1:n,i1)) + fvec(n+1)
    activ(i1) = 1.d0/(1.d0 + exp(-z(i1)))
    activ(i1) = nint(activ(i1))
    if (activ(i1)==y_test(i1)) then
      ncct = ncct + 1
    end if
  end do
  terror = 1.d0 - (dble(ncct)/dble(d))
end subroutine tst

subroutine nnmodel(fvec,n,m,d,c,cgrad)
  implicit none
  integer, intent(in) :: n,m,d !training data and inner layer sizes
  real(kind=8), dimension(m*(n+2)+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(m*(n+2)+1), intent(out) :: cgrad !gradient of cost
  integer :: i1,j1,l1
  real(kind=8), dimension(m,n) :: w_inner
  real(kind=8), dimension(m) :: b_inner,w_outer
  real(kind=8) :: dinv,b_outer
  !Declare other variables as needed
  real(kind=8), dimension(m,d) :: z_inner,a_inner,g_inner
  real(kind=8), dimension(d) :: z_outer,a_outer,e_outer,g_outer,eg_outer
  real(kind=8) :: dcdb_outer
  real(kind=8),dimension(m) :: dcdw_outer
  real(kind=8), dimension(m) :: dcdb_inner
  real(kind=8), dimension(m,n) :: dcdw_inner

  dinv = 1.d0/dble(d)

  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*m+1
    w_inner(:,i1) = fvec(j1:j1+m-1) !inner layer weight matrix
  end do
  b_inner = fvec(n*m+1:n*m+m) !inner layer bias vector
  w_outer = fvec(n*m+m+1:n*m+2*m) !output layer weight vector
  b_outer  = fvec(n*m+2*m+1) !output layer bias

  !Add code to compute c and cgrad

  !Compute inner layer activation vector, a_inner
  z_inner = matmul(w_inner,nm_x)
  do i1=1,d
    z_inner(:,i1) = z_inner(:,i1) + b_inner
  end do
  a_inner = 1.d0/(1.d0 + exp(-z_inner))

  !Compute outer layer activation (a_outer) and cost
  z_outer = matmul(w_outer,a_inner) + b_outer
  a_outer = 1.d0/(1.d0+exp(-z_outer))
  e_outer = a_outer-nm_y
  c = 0.5d0*dinv*sum((e_outer)**2)

  !Compute dc/db_outer and dc/dw_outer
  g_outer = a_outer*(1.d0-a_outer)
  eg_outer = e_outer*g_outer
  dcdb_outer = dinv*sum(eg_outer)
  dcdw_outer = dinv*matmul(a_inner,eg_outer)

  !Compute dc/db_inner and dc/dw_inner
  g_inner = a_inner*(1.d0-a_inner)
  dcdb_inner = dinv*w_outer*matmul(g_inner,eg_outer)
  do l1 = 1,n
    dcdw_inner(:,l1) = dinv*w_outer*matmul(g_inner,(nm_x(l1,:)*eg_outer))
  end do

  !Pack gradient into cgrad
  do i1=1,n
    j1 = (i1-1)*m+1
    cgrad(j1:j1+m-1) = dcdw_inner(:,i1)
  end do
  cgrad(n*m+1:n*m+m) = dcdb_inner
  cgrad(n*m+m+1:n*m+2*m) = dcdw_outer
  cgrad(n*m+2*m+1) = dcdb_outer

end subroutine nnmodel

!Compute test error provided fitting parameters
!and with testing data stored in nm_x_test and nm_y_test
subroutine run_nnmodel(fvec,n,m,d,test_error)
  implicit none
  integer, intent(in) :: n,m,d !training data and inner layer sizes
  real(kind=8), dimension(m*(n+2)+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: test_error !test_error
  integer :: i1,j1
  real(kind=8), dimension(m,n) :: w_inner
  real(kind=8), dimension(m) :: b_inner,w_outer
  real(kind=8) :: b_outer
  !Declare other variables as needed
  real(kind=8), dimension(m,d) :: z_inner,a_inner
  real(kind=8), dimension(d) :: z_outer,a_outer
  integer, dimension(d) :: e_outer


  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*m+1
    w_inner(:,i1) = fvec(j1:j1+m-1) !inner layer weight matrix
  end do
  b_inner = fvec(n*m+1:n*m+m) !inner layer bias vector
  w_outer = fvec(n*m+m+1:n*m+2*m) !output layer weight vector
  b_outer  = fvec(n*m+2*m+1) !output layer bias


  !Compute inner layer activation vector, a_inner
  z_inner = matmul(w_inner,nm_x)
  do i1=1,d
    z_inner(:,i1) = z_inner(:,i1) + b_inner
  end do
  a_inner = 1.d0/(1.d0 + exp(-z_inner))

  !Compute outer layer activation (a_outer) and cost
  z_outer = matmul(w_outer,a_inner) + b_outer
  a_outer = 1.d0/(1.d0+exp(-z_outer))
  e_outer = nint(a_outer-nm_y)

  test_error = dble(sum(e_outer))/dble(d)

end subroutine run_nnmodel

subroutine sgd(fvec_guess,n,m,d,alpha,fvec)
  implicit none
  integer, intent(in) :: n,m,d
  real(kind=8), dimension(:), intent(in) :: fvec_guess

  real(kind=8), intent(in) :: alpha
  real(kind=8), dimension(size(fvec_guess)), intent(out) :: fvec
  integer :: i1, j1, i1max=400
  real(kind=8) :: c

  real(kind=8), dimension(size(fvec_guess)) :: cgrad

  real(kind=8), allocatable, dimension(:,:) :: xfull
  integer, allocatable, dimension(:) :: yfull

  real(kind=8), dimension(d+1) :: r

  integer, dimension(d+1) :: j1array

  !store full nm_x,nm_y
  allocate(xfull(size(nm_x,1),size(nm_x,2)),yfull(size(nm_y)))
  xfull = nm_x
  yfull = nm_y

  !will only use one image at a time, so need to reallocate nm_x,nm_y
  call data_init(n,1)

  fvec = fvec_guess
  do i1=1,i1max
    call random_number(r)
    j1array = floor(r*d+1.d0) !d random integers falling between 1 and d (inclusive); will compute c, cgrad for one image at a time cycling through these integers

    do j1 = 1,d
      nm_x(:,1) = xfull(:,j1array(j1))
      nm_y = yfull(j1array(j1))

      !compute cost and gradient with randomly selected image
      if (m==0) then
        call snmodel(fvec,n,1,c,cgrad)
      else
        call nnmodel(fvec,n,m,1,c,cgrad)
      end if
      fvec = fvec - alpha*cgrad !update fitting parameters using gradient descent step
    end do

    if (mod(i1,50)==0) print *, 'completed epoch # ', i1

  end do

 !reset nm_x,nm_y to intial state at beginning of subroutine
  call data_init(size(xfull,1),size(xfull,2))
  nm_x = xfull
  nm_y = yfull
  deallocate(xfull,yfull)
end subroutine sgd


end module nmodel
