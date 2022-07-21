"""
author: Fabian Schaipp
"""
import numpy as np
from ..helper.utils import compute_batch_gradient, stop_scikit_saga

from numba.typed import List
from numba import njit

import time


def sgd_loop(f, phi, x_t, A, N, tol, alpha, beta, n_epochs, batch_size):
    # initialize for diagnostics
    x_hist = List()
    runtime = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    
    epoch_iter = int(N/batch_size)
    
    for iter_t in np.arange(n_epochs):
        
        if eta <= tol:
            break
        
        alpha_t = alpha/(iter_t+1)**beta
    
        start = time.time()
        x_t = sgd_epoch(f, phi, x_t, A, alpha_t, batch_size, epoch_iter)
        end = time.time()
        
        # stop criterion (at end of each epoch)
        eta = stop_scikit_saga(x_t, x_old)
        x_old = x_t
        
        x_hist.append(x_t)
        runtime.append(end-start)
        step_sizes.append(alpha_t)
        
    return x_t, x_hist, runtime, step_sizes, eta


@njit()
def sgd_epoch(f, phi, x_t, A, alpha_t, batch_size, epoch_iter):
    
    for iter_t in np.arange(epoch_iter):        
        # sample
        S = np.random.randint(low = 0, high = f.N, size = batch_size)
        
        # mini-batch gradient step
        A_S = A[S,:]
        g_t = (1/batch_size) * (A_S.T @ compute_batch_gradient(f, A_S@x_t, S))
        
        w_t = x_t - alpha_t*g_t
        # compute prox step
        x_t = phi.prox(w_t, alpha_t)
        
    return x_t