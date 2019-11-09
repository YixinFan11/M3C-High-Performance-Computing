"""Lab 2 Task 2
This module contains functions for simulating Brownian motion
and analyzing the results
"""
import numpy as np
import matplotlib.pyplot as plt

def brown1(Nt,M,dt=1,display=False):
    """Run M Brownian motion simulations each consisting of Nt time steps
    with time step = dt
    Returns: X: the M trajectories; Xm: the mean across these M samples; Xv:
    the variance across these M samples
    """
    from numpy.random import randn

    #Initialize variable
    X = np.zeros((M,Nt+1))

    R = randn(M, Nt)
    R = np.sqrt(dt)*R

    #1D Brownian motion: X_j+1 = X_j + sqrt(dt)*N(0,1)
    for j in range(Nt):
        X[:,j+1] = X[:,j] + R[:,j]

    Xm = np.mean(X,axis=0)
    Xv = np.var(X,axis=0)

    if display:
        #Display figure
        plt.figure()
        plt.plot(X[::40,:].T)
        plt.plot(Xm,'k--')
        plt.plot(np.sqrt(Xv),'r--')
        plt.show()
    return X,Xm,Xv


def analyze(display=True):
    """Complete this function to analyze simulation error
    """

def brown3(Nt,M,dt=1,display=False):
    """Run M Brownian motion simulations each consisting of Nt time steps
    with time step = dt
    Returns: X: the M trajectories; Xm: the mean across these M samples; Xv:
    the variance across these M samples
    """
    from numpy.random import randn

    #Initialize variable
    X = np.zeros((M,Nt+1))

    R = randn(M, Nt)
    R = np.sqrt(dt)*R

    #1D Brownian motion: X_j+1 = X_j + sqrt(dt)*N(0,1)
    X = np.cumsum(R,axis=1)
    Xm = np.mean(X,axis=0)
    Xv = np.var(X,axis=0)

    if display:
        #Display figure
        plt.figure()
        plt.plot(X[::40,:].T)
        plt.plot(Xm,'k--')
        plt.plot(np.sqrt(Xv),'r--')
        plt.show()
    return X,Xm,Xv
