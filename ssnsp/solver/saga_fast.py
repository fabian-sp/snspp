"""
author: Fabian Schaipp
"""
import numpy as np
from ..helper.utils import compute_gradient_table, stop_optimal, stop_scikit_saga
import time

from numba.typed import List
from numba import njit

def saga_fast_wrapper(f, phi, A, x0, tol, params = dict(), verbose = False, measure = False):
    
    # set functions
    if f.name == 'logistic':
        f_eval = lambda x: f_logistic(x, f.A, f.N)
        g_eval = g_logistic
        
    else:
        raise ValueError("SAGA fast can only handle specific functions...")
        
    if phi.name == '1norm':
        phi_eval = lambda x: phi_1norm(x, phi.lambda1)
        phi_prox = lambda x, alpha: prox_1norm(x, phi.lambda1, alpha)        
    else:
        raise ValueError("SAGA fast can only handle specific functions...")
    
    # set step size
    if 'gamma' not in params.keys():
        if f.name == 'squared':
            L = 2 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)**2).max()
        elif f.name == 'logistic':
            L = .25 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)**2).max()
        else:
            print("Determination of step size not possible! Probably get divergence..")
            L = 1
        gamma = 1./(3*L) 
    else:
        gamma = params['gamma']
    
    params['gamma'] = np.float64(gamma)
    
    # run saga fast
    x_t, x_mean, info = saga_fast(f_eval, g_eval, phi_eval, phi_prox, A, f.m, x0, tol, params, verbose, measure)
    
    return x_t, x_mean, info

def saga_fast(f_eval, g_eval, phi_eval, phi_prox, A, m, x0, tol = 1e-3, params = dict(), verbose = False, measure = False):
    """
    fast implementation of the SAGA algorithm for problems of the form 
    min 1/N * sum f_i(A_i x) + phi(x)
    
    speedup achieved by numba
    """
    # initialize all variables
    n = len(x0)
    N = len(m)
    assert n == A.shape[1], "wrong dimensions"
    
    x_t = x0.copy().astype('float64')
    
    # creates a vector with nrows like A in order to index the relevant A_i from A
    dims = np.repeat(np.arange(N),m)

    # initialize object for storing all gradients 
    #gradients = compute_gradient_table(f, x_t).astype('float64')
    gradients = np.zeros((N,n)).astype('float64')
    assert gradients.shape == (N,n)
    
    if 'n_epochs' not in params.keys():    
        params['n_epochs'] = 10
    
    # Main loop
    start = time.time()
    x_t, x_hist, step_sizes, obj, eta  = saga_loop(f_eval, g_eval, phi_eval, phi_prox, x_t, A, dims, N, tol, params['gamma'], gradients, params['n_epochs'])
    
    end = time.time()
    
    x_hist = np.vstack(x_hist)
    n_iter = x_hist.shape[0]
    
    # compute x_mean retrospectivly and evaluate objective
    obj2= list()
    xmean_hist = x_hist.cumsum(axis=0) / (np.arange(n_iter) + 1)[:,np.newaxis]
    for j in np.arange(n_iter):
        obj2.append(f_eval(xmean_hist[j,:]) + phi_eval(xmean_hist[j,:]))
    x_mean = xmean_hist[-1,:].copy()
    
    # distribute runtime uniformly on all iterations
    runtime = [(end-start)/n_iter]*n_iter
    
    if eta > tol:
        status = 'max iterations reached'
    else:
        status = 'optimal'
        
    print(f"SAGA terminated after {n_iter} iterations")
    print(f"SAGA status: {status}")
    
    info = {'objective': np.array(obj), 'objective_mean': np.array(obj2), 'iterates': x_hist, 'step_sizes': np.array(step_sizes), \
            'gradient_table': gradients, 'runtime': np.array(runtime)}
    
    return x_t, x_mean, info


@njit()
def saga_loop(f_eval, g_eval, phi_eval, phi_prox, x_t, A, dims, N, tol, gamma, gradients, n_epochs):
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    obj = List()
    
    eta = 1e10
    
    for iter_t in np.arange(N * n_epochs):
        
        if eta <= tol:
            break
        
        x_old = x_t
        # sample
        j = np.random.randint(low = 0, high = N, size = 1)
        A_j = A[dims == j,:]
        
        # compute the gradient
        g = A_j.T @ g_eval(A_j@x_t, j)
        old_g = (-1) * gradients[j,:] + (1/N)*gradients.sum(axis = 0)
        w_t = x_t - gamma * (g + old_g)[0,:]
        
        # store new gradient
        gradients[j,:] = g
        
        # compute prox step
        x_t = phi_prox(w_t, gamma)
        
        # stop criterion
        eta = stop_scikit_saga(x_t, x_old)
        
        # store everything
        x_hist.append(x_t)
        step_sizes.append(gamma)
        obj.append(f_eval(x_t) + phi_eval(x_t))
        
    return x_t, x_hist, step_sizes, obj, eta

#%%

@njit()
def f_logistic(x, A, N):
    """
    computes f(x) as in paper
    """
    return 1/N * np.log(1+np.exp(-A@x)).sum()

@njit()
def g_logistic(x, i):
    """
    computes the gradient of function f_i at x
    """
    return -1/(1+np.exp(x))


@njit()
def phi_1norm(x, lambda1):
    return lambda1*np.abs(x).sum()

@njit()
def prox_1norm(x, alpha, lambda1):
    l = alpha * lambda1
    return np.sign(x) * np.maximum( np.abs(x) - l, 0.)



#%%

#x0 = np.zeros(n)
#x_saga, x_mean_saga, info = saga_fast(f, phi, x0, tol = 1e-3, params = dict(), verbose = True, measure = False)
