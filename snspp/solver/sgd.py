"""
author: Fabian Schaipp
"""
import numpy as np
from ..helper.utils import compute_batch_gradient, stop_scikit_saga

from numba.typed import List
from numba import njit

@njit()
def sgd_loop(f, phi, x_t, tol, alpha, n_epochs, batch_size, truncate = False):
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    
    # construct array which tells us when to store (at the END of each epoch)
    max_iter = int(f.N * n_epochs/batch_size)
    counter = np.arange(max_iter)*batch_size/f.N % 1
    counter = np.append(counter,0)
    
    store = (counter[1:] <= counter[:-1])
    assert len(store) == max_iter
    assert store[-1]
    
    for iter_t in np.arange(max_iter):
        
        if eta <= tol:
            break
        
        # sample
        S = np.random.randint(low = 0, high = f.N, size = batch_size)
        
        # mini-batch gradient step
        g_t = compute_batch_gradient(f, x_t, S)
        
      
        # determine step size
        beta = 0.51
        if truncate:
        # no prox step
            gamma_t = alpha/(iter_t+1)**beta            
            u_t = phi.subg(x_t)            
            alpha_t = np.minimum(gamma_t, (f.eval_batch(x_t, S) + phi.eval(x_t))/np.linalg.norm(g_t + u_t)**2)  
    
            x_t = x_t - alpha_t* (g_t + u_t)
                
        else:
        # prox step
            gamma_t = alpha/(iter_t+1)**beta
            alpha_t = np.minimum(gamma_t, (f.eval_batch(x_t, S))/np.linalg.norm(g_t)**2)  
            
            w_t = x_t - alpha_t*g_t
        
            # compute prox step
            x_t = phi.prox(w_t, alpha_t)
        
        # stop criterion (at end of each epoch)
        if store[iter_t]:
            eta = stop_scikit_saga(x_t, x_old)
            x_old = x_t
            
        # store everything (at end of each epoch)
        if store[iter_t]:
            x_hist.append(x_t)
            step_sizes.append(alpha_t)

    return x_t, x_hist, step_sizes, eta

