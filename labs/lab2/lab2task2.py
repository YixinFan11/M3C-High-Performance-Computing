"""Lab 2 Task 2
This module contains functions for simulating Brownian motion
and analyzing the results
"""
import numpy as np
import matplotlib.pyplot as plt

def brown1(Nt,M,dt=1):
    """Run M Brownian motion simulations each consisting of Nt time steps
    with time step = dt
    Returns: X: the M trajectories; Xm: the mean across these M samples; Xv:
    the variance across these M samples
    """
    from numpy.random import randn

    #Initialize variable
    X = np.zeros((M,Nt+1))

    #1D Brownian motion: X_j+1 = X_j + sqrt(dt)*N(0,1)
    for i in range(M):
        for j in range(Nt):
            X[i,j+1] = X[i,j] + np.sqrt(dt)*randn(1)

    Xm = np.mean(X,axis=0)
    Xv = np.var(X,axis=0)
    return X,Xm,Xv


def analyze(display=False):
    """Complete this function to analyze simulation error
    """