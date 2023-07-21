"""
@author: Fabian Schaipp
"""

import numpy as np                           
import time
import warnings

from numba import njit

from .sparse_utils import compute_AS


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
        

@njit()
def sparse_svrg_epoch(f, phi, x_t, A, N, alpha, batch_size, loop_length, full_g, g_tilde):
    
    for t in np.arange(loop_length):
            
        S = np.random.randint(low = 0, high = N, size = batch_size)
        
        # compute the gradient
        A_S = compute_AS(A, S)
        v_t = f.g(A_S@x_t, S)
        
        g_S = (1/batch_size) * A_S.T @ (v_t - full_g[S])

        w_t = x_t - alpha*(g_S + g_tilde)
        x_t = phi.prox(w_t, alpha)
        
    return x_t

#%%

@njit()
def sparse_adagrad_epoch(f, phi, x_t, A, N, alpha, delta, epoch_iter, batch_size, G):
    for j in np.arange(epoch_iter):
        # sample
        S = np.random.randint(low = 0, high = N, size = batch_size)
        
        # mini-batch gradient step
        A_S = compute_AS(A, S)
        G_t = (1/batch_size) * (A_S.T @ f.g(A_S@x_t, S))
        G += G_t * G_t
        
        L_t = (1/alpha) * (delta + np.sqrt(G))
        
        w_t = x_t - (1/L_t) * G_t
        
        # compute Adagrad prox step
        x_t = phi.adagrad_prox(w_t, L_t)

    return x_t, G

@njit()
def sparse_batch_saga_epoch(f, phi, x_t, A, N, alpha, gradients, reg, g_sum, loop_length, batch_size):
    
    for iter_t in np.arange(loop_length):
                     
        # sample, result is int!
        S = np.random.randint(low = 0, high = N, size = batch_size)
        
        # compute the gradient, 
        A_S = compute_AS(A, S)
        new_v_t = f.g(A_S@x_t, S)

        v_t = gradients[S]
        g_t = A_S.T @ (new_v_t - v_t) 
        
        w_t = (1 - alpha*reg)*x_t - alpha*((1/batch_size)*g_t + g_sum)
        
        # store new gradient
        gradients[S] = new_v_t
        g_sum += (1/N)*g_t
        
        # compute prox step
        x_t = phi.prox(w_t, alpha)
        
    return x_t, g_sum
