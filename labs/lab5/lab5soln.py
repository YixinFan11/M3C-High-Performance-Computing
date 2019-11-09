#Assumes lab5 fortran code has been compiled with: $ f2py -c lab5soln.f90 -m l5
import numpy as np
import matplotlib.pyplot as plt
from l5 import quad

#set parameters, initialze arrays
nvalues = [6,12,24,48]
I_mid = np.zeros(4)
I_sim = np.zeros(4)

#compute integrals with different resolutions
for i,n in enumerate(nvalues):
    quad.quad_n = n
    quad.midpoint()
    I_mid[i] = quad.quad_sum
    quad.simpson()
    I_sim[i] = quad.quad_sum

#Plot errors on log-log plots
plt.figure()
plt.loglog(nvalues,np.abs(I_mid-np.pi),'x')
plt.loglog(nvalues,np.abs(I_sim-np.pi),'r.')
plt.xlabel('n')
plt.ylabel('error')
plt.legend(('midpoint','Simpson'))
plt.title('Lab5 comparing midpoint and Simpson rules')
plt.grid(True)
plt.show()
