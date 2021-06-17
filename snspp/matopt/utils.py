"""
author: Fabian Schaipp
"""
import numpy as np
from numba import njit

@njit()
def mat_inner(Y,X):
    """
    calculates <Y,X> = Tr(Y.T @ X)
    """
    (p,q) = X.shape
    res = 0
    for j in np.arange(q):
        res += np.dot(Y[:,j], X[:,j])
        
    return res

@njit()
def multiple_mat_inner(A,X):
    (p,q,m) = A.shape
    res = np.zeros(m)
    
    for j in np.arange(m):
         res[j] = mat_inner(A[:,:,j], X)
         
    return res
