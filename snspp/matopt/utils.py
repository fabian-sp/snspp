"""
author: Fabian Schaipp
"""
import numpy as np
from numba import njit, prange

@njit()
def matdot(Y,X):
    """
    calculates <Y,X> = Tr(Y.T @ X)
    """
    res = np.sum(Y*X)
        
    return res

# @njit(parallel = True)
# def matdot2(Y,X):
#     (p,q) = X.shape
#     res = 0
#     for j in prange(q):
#         res += np.dot(Y[:,j], X[:,j])
#     return res

@njit(parallel = True)
def multiple_matdot(A,X):
    (p,q,m) = A.shape
    res = np.zeros(m)
    
    for j in prange(m):
         res[j] = matdot(A[:,:,j], X)
         
    return res

@njit()
def compute_full_xi(f, X):
    xi = np.zeros(f.N)
    for i in np.arange(f.N):
        xi[i] = f.g(matdot(f.A[i,:,:], X))
        
    return xi

