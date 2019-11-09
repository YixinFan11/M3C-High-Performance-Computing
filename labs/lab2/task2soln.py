"""Brownian motion example
This module contains functions for simulating Brownian motion
and analyzing the results (lectures 4,5 and lab 2, task 2)
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time

def brown1(Nt,M,dt=1):
    """Run M Brownian motion simulations each consisting of Nt time steps
    with time step = dt
    Returns: X: the M trajectories; Xm: the mean across these M samples; Xv:
    the variance across these M samples
    Inefficient implementation.
    """
    X = np.zeros((M,Nt+1)) #initialize array

    #Run M Nt-step simulations
    for i in range(M):
        for j in range(Nt):
            X[i,j+1] = X[i,j] + np.sqrt(dt)*np.random.randn()

    #Compute statistics
    Xm = np.mean(X,axis=0)
    Xv = np.var(X,axis=0)

    return X,Xm,Xv

def brown2(Nt,M,dt=1):
    """Run M Brownian motion simulations each consisting of Nt time steps
    with time step = dt
    Returns: X: the M trajectories; Xm: the mean across these M samples; Xv:
    the variance across these M samples
    Partially vectorized implementation.
    """
    X = np.zeros((M,Nt+1)) #initialize array
    R = np.sqrt(dt)*np.random.randn(M,Nt)

    #Run M Nt-step simulations
    for j in range(Nt):
        X[:,j+1] = X[:,j] + R[:,j]

    #Compute statistics
    Xm = np.mean(X,axis=0)
    Xv = np.var(X,axis=0)

    return X,Xm,Xv


def brown3(Nt,M,dt=1):
    """Run M Brownian motion simulations each consisting of Nt time steps
    with time step = dt
    Returns: X: the M trajectories; Xm: the mean across these M samples; Xv:
    the variance across these M samples
    Loop-free version
    """
    from numpy.random import randn

    R = np.sqrt(dt)*np.random.randn(M,Nt+1)
    R[:,0] = 0
    X = np.cumsum(R,axis=1) #replace time-marching loop with built-in cumsum function

    Xm = np.mean(X,axis=0)
    Xv = np.var(X,axis=0)
    return X,Xm,Xv


def analyze(display=False):
    """Lab2 task 2: Compute variance in M Brownian motion simulations with varying M
    Compute and return error along with M values and variances.
    Plot error if input variable display=True
    To run this function, first import brown in python terminal, then: brown.analyze(True)
    """

    Mvalues = [100,1000,10000,100000] #Compute error for these values of M
    Nt=100
    Xvarray = np.zeros(len(Mvalues)) #initialize array to store variances at t=Nt+1
    errorv = np.zeros(len(Mvalues))
    #Compute variances for each M
    for i,M in enumerate(Mvalues):
        _,_,Xv = brown3(Nt,M)
        print(i,M,Xv[-1])
        Xvarray[i] = Xv[-1]
        t = np.arange(Nt+1)

    errorv = np.abs(Xvarray-Nt) #we expect Xv(t) = t for dt = 1

    #construct least-squares fit to: error  = A M^n
    #we expect n=-1/2, but in practice, the observed value
    #can vary substantially, and an average over time
    #is needed to obtain "better" results
    p=np.polyfit(np.log(Mvalues),np.log(errorv),1)

    if display:
        plt.figure()
        plt.loglog(Mvalues,errorv,'x--')

        n = p[0]; A = p[1]
        plt.plot(Mvalues,np.exp(A)*(Mvalues)**n,'r--')
        plt.legend(('simulation','least-squares fit with n=%f'%(n)),loc='best')
        plt.xlabel('M')
        plt.ylabel('$\epsilon$')
        plt.title('Variation of variance with sample size')
    return Mvalues,Xvarray,errorv,p


if __name__ == '__main__':
    #Compare speed of brown0 and brown1, use run command in terminal to
    #execute this block
    t1 = time()
    _,_,_ = brown0(1,500,200)
    t2 = time()
    dt0 = t2-t1

    t3 = time()
    _,_,_ = brown1(500,200)
    t4 = time()
    dt1 = t4-t3

    print("dt0=",dt0)
    print("dt1=",dt1)
