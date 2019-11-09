"""Simulate 1D Brownian motion
"""
import numpy as np
import matplotlib.pyplot as plt

#Set parameters
dt = 1
Nt = 200
M = 400

#Initialize variables
X = np.zeros((M,Nt+1))

#Brownian calculation
#X_j+1 = X_j + sqrt(dt) N(0,1)
for i in range(M):
    for j in range(Nt):
        X[i,j+1] = X[i,j] + np.sqrt(dt)*np.random.randn()


#Compute mean variance
Xmean = X.mean(axis=0)
Xvar = X.var(axis=0)

#Display results
plt.figure()
plt.plot(X[::40,:].T)
plt.plot(Xmean,'k--')
plt.plot(np.sqrt(Xvar),'r-.')
plt.show()
