"""Python module that imports fortran module for
vector triad calculation and runs it with a range
of problem sizes.
Fortran module converted to .so file with:
f2py -c triad.f90 -m trf90
"""
import numpy as np
import matplotlib.pyplot as plt
from trf90 import triad as tr

NVALUES = np.logspace(2, 5, 60)
TARRAY = np.empty_like(NVALUES)

#compute triad for range of n
for n in enumerate(NVALUES):
    TARRAY[n[0]] = tr.compute_triad(n[1])
    print("i=%d, n=%d, t=%f"%(n[0], n[1], TARRAY[n[0]]))


#display results
plt.figure()
plt.semilogx(4*NVALUES, TARRAY)
plt.xlabel('4*n')
plt.ylabel('walltime/n')
plt.axis('tight')
plt.show()
