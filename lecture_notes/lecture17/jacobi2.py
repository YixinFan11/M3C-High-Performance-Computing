"""module to solve steady-state heat eqn with Jacobi iteration
1D test needs jacf90.so generated with:
f2py -c jacobi2s.f90 -m jf90

"""    

from jf90 import jacobi as jac
import numpy as np
import matplotlib.pyplot as plt


def runj1d(n,a,b,S):
    """input: n: number of grid points
            a,b: boundary conditions
            S: source
        calls f2py-generated jac.jacobi1d, plots
       results, and returns error"""     
    T = jac.jacobi1d(n,a,b,S)
    
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
    

def runj2d(n,u,d,l,r,S):
    """input: n: number of grid points
            u,d,l,r: boundary conditions
            S: source amplitude
        calls f2py-generated jac.jacobi2d, plots
       results, and returns error"""     
    T = jac.jacobi2d(n,u,d,l,r,S)
    
    x = np.linspace(0,1,n+2)
    Texact = (r-l)*x + l + S*np.sin(np.pi*x)/np.pi**2
    
    plt.figure()
    plt.contour(x,x,T)
 #   plt.plot(x,Texact,'r--')
 #   plt.title(n)
    
    plt.figure()
    plt.semilogy(jac.deltat)
    plt.title(n)
#    return np.mean(abs(T[1:-1]-Texact[1:-1]))

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
        eps = eps + [runj1d(n,a,b,S)]
    plt.show()


