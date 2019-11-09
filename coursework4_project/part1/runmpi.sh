mpif90 -o p1mpi.exe p1mpi.f90 p1main_mpi.f90

mpiexec -n 4 p1mpi.exe
