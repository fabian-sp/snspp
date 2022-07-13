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
    """
    shapes:
        
        gradients(N,)
        g_sum (n,)
        g_j_diff (n,)
        
    """
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    
    g_sum = (1/N)*A.transpose().mult_vec(gradients)
    
    for iter_t in np.arange(N * n_epochs):
        
        if eta <= tol:
            break
             
        # sample, result is int!
        j = np.random.randint(low = 0, high = N, size = 1)[0]
        
        # z_j should be array of dim m_j
        z_j = A.row(j) @ x_t
        new_g_j = f.g(z_j, j)
            
        g_j = gradients[j]
        g_j_diff = A.row(j) * (new_g_j - g_j) 
        
        w_t = (1 - alpha*reg)*x_t - alpha*(g_j_diff + g_sum)
        #w_t = x_t - alpha * (g + old_g)
        
        # store new gradient
        gradients[j] = new_g_j
        g_sum = g_sum + (1/N)*g_j_diff
        
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




