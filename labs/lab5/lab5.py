#Assumes lab5 fortran code has been compiled with: $ f2py -c lab5.f90 -m l5
import numpy as np
import matplotlib.pyplot as plt
from l5 import quad

#set parameters, initialze arrays
nvalues = [6,12,24,48]
I_mid = np.zeros(len(nvalues))
I_sim = np.zeros(len(nvalues))

#compute integrals with different resolutions
for i,n in enumerate(nvalues):
    # Compute integrals with n intervals with each method, storing the results
    # in I_mid and I_sim

#Plot errors on log-log plots
