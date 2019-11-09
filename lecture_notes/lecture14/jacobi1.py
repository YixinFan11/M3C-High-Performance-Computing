"""module to solve steady-state heat eqn with Jacobi iteration
Needs j1.so generated with:
f2py --f90flags='-fopenmp' -lgomp -c jacobi1s_omp.f90 -m j1"""

from j1 import jacobi as jac
import numpy as np
import matplotlib.pyplot as plt


def runj1(n,a,b,S):
    """input: n: number of grid points
            a,b: boundary conditions
            S: source
        calls f2py-generated jac.jacobi1, plots
       results, and returns error"""     
    T = jac.jacobi1(n,a,b,S)
    
    x = np.linspace(0,1,n+2)
    Texact = (b-a)*x + a + S*np.sin(np.pi*x)/np.pi**2
    
    plt.figure()
    plt.plot(x,T)
    plt.plot(x,Texact,'r--')
    plt.title(n)
    
    plt.figure()
    plt.semilogy(jac.deltat)
    plt.title(n)
    return np.mean(abs(T[1:-1]-Texact[1:-1]))
    

if __name__ == '__main__':
    jac.kmax = 1000000
    jac.tol = 1.0e-12
    jac.deltat = np.zeros(jac.kmax)
    nvalues = [100,1000]
    a=0.0
    b=1.0
    S = 10.0
    eps=[]
    for n in nvalues:
        eps = eps + [runj1(n,a,b,S)]
    plt.show()


