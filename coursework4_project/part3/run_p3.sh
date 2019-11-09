rm C.dat
rm deltac.dat

mpif90 -o p3.exe p3.f90

mpiexec -n 4 p3.exe
