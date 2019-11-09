import numpy as np


#compute sin(x) using loop
def sin_loop(x):
    s = np.zeros_like(x)
    for i,xi in enumerate(x):
        s[i] = np.sin(xi)
    return s

#vectorized
def sin_vec(x):
    s = np.sin(x)
    return s
