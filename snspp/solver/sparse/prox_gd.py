"""
@author: Fabian Schaipp
"""

import numpy as np                           
import time
import warnings

from numba.typed import List
from numba import njit

from scipy.sparse.csr import csr_matrix

from ...helper.utils import stop_scikit_saga


@njit()
def sparse_saga_loop(f, phi, x_t, A, N, tol, alpha, gradients, n_epochs, reg):
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    g_sum = (1/N)*gradients.sum(axis = 0)
    
    
    for iter_t in np.arange(N * n_epochs):
        
        if eta <= tol:
            break
             
        # sample, result is int!
        j = np.random.randint(low = 0, high = N, size = 1)[0]
        
        # z_j should be array of dim m_j
        z_j = np.array((A.row(j) @ x_t))
        g = A.row(j).T @ f.g(z_j, j)
            
        g_j = gradients[j,:].reshape(-1)
        old_g = (-1)*g_j + g_sum
        
        w_t = (1 - alpha*reg)*x_t - alpha*(g + old_g)
        #w_t = x_t - alpha * (g + old_g)
        
        # store new gradient
        gradients[j,:] = g
        g_sum = g_sum - (1/N)*g_j + (1/N)*g
        
        # compute prox step
        x_t = phi.prox(w_t, alpha)
        
        # stop criterion
        if iter_t % N == N-1:
            eta = stop_scikit_saga(x_t, x_old)
            x_old = x_t
            
        # store everything (at end of each epoch)
        if iter_t % N == N-1:
            x_hist.append(x_t)
            step_sizes.append(alpha)
        
    return x_t, x_hist, step_sizes, eta




