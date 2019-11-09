"""Tests Fortran module jacobi1s_omp.f90 which has been compiled with:
f2py --f90flags='-fopenmp' -lgomp -c jacobi1s_omp.f90 -m j1
"""
from j1 import jacobi as jac
import numpy as np
import matplotlib.pyplot as plt


def runj1(n,a,b,S,showPlot=False):
   
    T = jac.jacobi1(n,a,b,S) #call fortran jacobi routine
    dT = jac.deltat[jac.deltat.nonzero()]
    x = np.linspace(0,1,n+2)
    Texact = (b-a)*x + a + S*np.sin(np.pi*x)/np.pi**2 #exact solution
    
    if showPlot:
        plt.figure()
        plt.plot(x,T)
        plt.plot(x,Texact,'r--')
        plt.xlabel('x')
        plt.ylabel('T')
        plt.title('Jacobi soln to heat eqn., n,a,b,S0=%d,%2.1f,%2.1f,%2.1f' %(n,a,b,S))
        plt.legend(('Computed','Exact'),loc='best')
        
        plt.figure()
        plt.semilogy(dT)
        plt.xlabel('iteration')
        plt.ylabel('Relative error')
        plt.title('Convergence of Jacobi iterations, n,a,b,S0=%d,%2.1f,%2.1f,%2.1f' %(n,a,b,S))
    return np.mean(np.abs(T[1:-1]-Texact[1:-1]))
    
#--------------------------
if __name__ == '__main__':
    #set parameters    
    jac.kmax = 100000
    jac.tol = 1.0e-12
    jac.deltat = np.zeros(jac.kmax)
    jac.numthreads = 2
    nvalues = [100,1000]
    a=0.0
    b=1.0
    S = 10.0
    
    #run jacobi code
    eps=[]
    for n in nvalues:
        eps = eps + [runj1(n,a,b,S,True)]
    plt.show()


