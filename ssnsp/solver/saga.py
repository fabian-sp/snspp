"""
author: Fabian Schaipp
"""

import numpy as np
from ..helper.utils import compute_x_mean, compute_gradient_table, stop_optimal, stop_scikit_saga
import time
import warnings


def saga(f, phi, x0, tol = 1e-3, params = dict(), verbose = False, measure = False):
    """
    implementation of the SAGA algorithm for problems of the form 
    min 1/N * sum f_i(A_i x) + phi(x)

    """
    # initialize all variables
    A = f.A.copy()
    n = len(x0)
    m = f.m.copy()
    N = len(m)
    assert n == A.shape[1], "wrong dimensions"
    
    x_t = x0.copy()
    x_mean = x_t.copy()
    
    # creates a vector with nrows like A in order to index the relevant A_i from A
    dims = np.repeat(np.arange(N),m)

    # initialize object for storing all gradients 
    gradients = compute_gradient_table(f, x_t)
    assert gradients.shape == (N,n)
    
    if 'n_epochs' not in params.keys():    
        params['n_epochs'] = 10
    
    if 'reg' not in params.keys():    
            params['reg'] = 0
    
    if 'gamma' not in params.keys():
        if f.name == 'squared':
            L = 2 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = f.A)**2).max()
        elif f.name == 'logistic':
            L = .25 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = f.A)**2).max()
        else:
            warnings.warn("We could not determine the correct SAGA step size! The default step size is maybe too large (divergence) or too small (slow convergence).")            
            L = 100
        gamma = 1./(3*L) 
    else:
        gamma = params['gamma']
    
    # initialize for stopping criterion
    status = 'not optimal'
    eta = np.inf
    
    # initialize for diagnostics
    x_hist = list()
    step_sizes = list()
    obj = list(); obj2 = list()
    runtime = list()
    
    hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
    if verbose and measure:
        print(hdr_fmt % ("iter", "obj (x_t)", "obj(x_mean)", "gamma", "eta"))
    
    
    g_sum = (1/N)*gradients.sum(axis = 0)
    for iter_t in np.arange(f.N * params['n_epochs']):
        
        if measure:
            start = time.time()
            
        if eta <= tol:
            status = 'optimal'
            break
        
        x_old = x_t.copy()
        
        # sample
        j = np.random.randint(low = 0, high = N, size = 1).squeeze()
        A_j = A[dims == j,:].copy()
        
        # compute the gradient
        g = A_j.T @ f.g(A_j@x_t, j)
        g_j = gradients[j,:].squeeze()
        old_g = (-1) * g_j + g_sum
        w_t = x_t - gamma * (g + old_g)
        
        # store new gradient
        gradients[j,:] = g.copy()
        g_sum = g_sum - (1/N)*g_j + (1/N)*g
        
        # compute prox step
        x_t = phi.prox(w_t, gamma)
        
        # stop criterion
        #eta = stop_optimal(x_t, f, phi)
        if measure:
            end = time.time()
            runtime.append(end-start)
        
        if iter_t % N == 1:
            eta = stop_scikit_saga(x_t, x_old)
        
        # store everything
        x_hist.append(x_t)
        step_sizes.append(gamma)
                  
        if measure and iter_t % N == 1:
            obj.append(f.eval(x_t.astype('float64')) + phi.eval(x_t))
        
        # calculate x_mean       
        if measure and iter_t % N == 1:
            x_mean = compute_x_mean(x_hist, step_sizes = None)
            obj2.append(f.eval(x_mean.astype('float64')) + phi.eval(x_mean))
            
            if verbose:
                print(out_fmt % (iter_t, obj[-1], obj2[-1] , gamma, eta))
          
        
    if eta > tol:
        status = 'max iterations reached'    
        
    print(f"SAGA terminated after {iter_t} iterations with accuracy {eta}")
    print(f"SAGA status: {status}")
    
    info = {'objective': np.array(obj), 'objective_mean': np.array(obj2), 'iterates': np.vstack(x_hist), 'step_sizes': np.array(step_sizes), \
            'gradient_table': gradients, 'runtime': np.array(runtime)}
    
    return x_t, x_mean, info