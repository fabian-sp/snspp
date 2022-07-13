import numpy as np
from numba import njit
import warnings


@njit()
def sparse_xi_inner(f, z):
    
    vals = np.zeros((f.N,1))
    
    for i in np.arange(f.N):
        vals[i,:] = f.g(z[i,:], i)
    
    return vals