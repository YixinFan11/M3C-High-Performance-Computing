!illustrate use of circle module
!requires circle.mod obtained by compiling module.circle.f90
!!To compile this code: 
!gfortran -o circle.exe module_circle.f90 main_circle.f90 
!To run:
!$ ./circle.exe
program main
    use circle
    implicit none
    real(kind=8) :: radius,C,A

    call initialize_pi()

    radius = 2.d0

    C = circumference(radius)

    A = area(radius)

    print *, 'radius=', radius
    print *, 'circumference=', C
    print *, 'area = ', A
end program main

