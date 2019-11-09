"""Script to test results of p3."""

import numpy as np
import matplotlib.pyplot as plt

n = 100
r = np.linspace(1, 1+np.pi, n+2)
t = np.linspace(0, np.pi, n+2)  # theta

C = np.loadtxt('C.dat')

plt.figure()
plt.contour(t, r, C, 50)
plt.xlabel('theta')
plt.ylabel('r')
plt.title('Final concentration field')
plt.show()
