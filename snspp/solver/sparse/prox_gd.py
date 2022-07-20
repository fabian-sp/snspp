"""
@author: Fabian Schaipp
"""

import numpy as np                           
import time
import warnings

from numba.typed import List
from numba import njit

from .sparse_utils import sparse_xi_inner, sparse_batch_gradient, compute_AS, create_csr, sparse_gradient_table

from ...helper.utils import stop_scikit_saga


def sparse_saga_loop(f, phi, x_t, A, N, tol, alpha, n_epochs, reg, measure_freq):
    """
    shapes:
        
        gradients(N,)
        g_sum (n,)
        g_j_diff (n,)
        
    """
    # create sparse numba matrix
    A_csr = create_csr(A)
    
    # initialize for diagnostics
    x_hist = List()
    runtime= List()
    step_sizes = List()
    
    # measure_freq = how many measurements per epoch
    loop_length = int(N/measure_freq)
    max_iter = int(N*n_epochs/loop_length)
    
    eta = 1e10
    x_old = x_t
    
    s0 = time.time()
    gradients = sparse_xi_inner(f, A@x_t)
    g_sum = (1/N)*(A.T @ gradients)
    e0 = time.time()
    
    for iter_t in np.arange(max_iter):
        
        if eta <= tol:
            break
             
        start = time.time()
        x_t, g_sum = sparse_saga_epoch(f, phi, x_t, A_csr, N, alpha, gradients, reg, g_sum, loop_length)
        end = time.time()
        
        # stop criterion
        eta = stop_scikit_saga(x_t, x_old)
        x_old = x_t
            
        # store everything (at end of each epoch)
        x_hist.append(x_t)
        if iter_t == 0:
            runtime.append(end-start+e0-s0)
        else:
            runtime.append(end-start)
        step_sizes.append(alpha)
        
    return x_t, x_hist, runtime, step_sizes, eta

@njit()
def sparse_saga_epoch(f, phi, x_t, A, N, alpha, gradients, reg, g_sum, loop_length):
    for iter_t in np.arange(loop_length):
        # sample, result is int!
        j = np.random.randint(low = 0, high = N, size = 1)[0]
        
        # z_j should be array of dim m_j
        A_j = A.row(j)
        z_j = A_j @ x_t
        new_g_j = f.g(z_j, j)
            
        g_j = gradients[j]
        g_j_diff = A_j * (new_g_j - g_j) 
        
        w_t = (1 - alpha*reg)*x_t - alpha*(g_j_diff + g_sum)
        
        # store new gradient
        gradients[j] = new_g_j
        g_sum += (1/N)*g_j_diff
        
        # compute prox step
        x_t = phi.prox(w_t, alpha)
        
    return x_t, g_sum
        
    

#%%

def sparse_svrg_loop(f, phi, x_t, A, N, tol, alpha, n_epochs, batch_size, m_iter):
    
    # create sparse numba matrix
    A_csr = create_csr(A)
    
    # initialize for diagnostics
    x_hist = List()
    runtime = List()
    runtime_fullg = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    
    S_iter = int(n_epochs*N / (batch_size*m_iter))
    
    for s in np.arange(S_iter):
        
        if eta < tol:
            break
        
        start = time.time()
        full_g = sparse_xi_inner(f, A@x_t)
        g_tilde = (1/N) * (A.T@full_g)
        end1 = time.time()
        
        x_t = sparse_svrg_epoch(f, phi, x_t, A_csr, N, alpha, batch_size, m_iter, full_g, g_tilde)
        end2 = time.time()
               
        # stop criterion
        eta = stop_scikit_saga(x_t, x_old)
        x_old = x_t
        
        # store in each outer iteration
        x_hist.append(x_t)    
        runtime.append(end2-start)
        runtime_fullg.append(end1-start)
        step_sizes.append(alpha)    


    return x_t, x_hist, runtime, runtime_fullg, step_sizes, eta

@njit()
def sparse_svrg_epoch(f, phi, x_t, A, N, alpha, batch_size, m_iter, full_g, g_tilde):
    
    for t in np.arange(m_iter):
            
        S = np.random.randint(low = 0, high = N, size = batch_size)
        
        # compute the gradient
        A_S = compute_AS(A, S)
        v_t = sparse_batch_gradient(f, A, x_t, S)
        
        g_S = (1/batch_size) * A_S.T @ (v_t - full_g[S])
        g_t = g_S + g_tilde

        w_t = x_t - alpha*g_t
        x_t = phi.prox(w_t, alpha)
        
    return x_t
