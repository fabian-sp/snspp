"""
author: Fabian Schaipp
"""
import numpy as np
from ..helper.utils import compute_gradient_table, compute_batch_gradient, stop_optimal, stop_scikit_saga
import time

from numba.typed import List
from numba import njit


def stochastic_gradient(f, phi, x0, solver = 'saga', tol = 1e-3, params = dict(), verbose = False, measure = False):
    """
    fast implementation of the SAGA algorithm for problems of the form 
    min 1/N * sum f_i(A_i x) + phi(x)
    
    works only if m_i = 1 forall i, i.e. one sample gives one summand!
    speedup achieved by numba, hence classes f and phi need to be jitted beforehand. If not possible use saga.py instead.
    """
    
    name = solver.upper()
        
    # initialize all variables
    n = len(x0)
    m = f.m.copy()
    assert np.all(f.m==1), "These implementations are restricted to the case m_i = 1, use saga.py if not the case"
    N = len(f.m)
    A = f.A.astype('float64')
    assert n == A.shape[1], "wrong dimensions"
    
    x_t = x0.copy().astype('float64')
    
    # creates a vector with nrows like A in order to index the relevant A_i from A
    dims = np.repeat(np.arange(N),m)

    # initialize object for storing all gradients 
    gradients = compute_gradient_table(f, x_t).astype('float64')
    #gradients = np.zeros((N,n)).astype('float64')
    assert gradients.shape == (N,n)
    
    # set step size
    if 'gamma' not in params.keys():
        if solver == 'saga':
            if f.name == 'squared':
                L = 2 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)**2).max()
            elif f.name == 'logistic':
                L = .25 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)**2).max()
            else:
                print("Determination of step size not possible! Probably get divergence..")
                L = 1
            gamma = 1./(3*L)
        
        elif solver == 'adagrad':
            gamma = .05
    else:
        gamma = params['gamma']
    
    gamma = np.float64(gamma)  
    
    if 'n_epochs' not in params.keys():    
        params['n_epochs'] = 5
    
    #########################################################
    ## Solver dependent parameters
    #########################################################
    if solver == 'saga':
        if 'reg' not in params.keys():    
            params['reg'] = 0
    
    elif solver == 'adagrad':
        if 'delta' not in params.keys():    
                params['delta'] = 1e-6
        if 'batch_size' not in params.keys():    
            params['batch_size'] = max(int(f.N * 0.01), 1)
    #########################################################
    # Main loop
    #########################################################
    start = time.time()
    
    if solver == 'saga':
        x_t, x_hist, step_sizes, eta  = saga_loop(f, phi, x_t, A, N, tol, gamma, gradients, params['n_epochs'], params['reg'])
    
    elif solver == 'adagrad':
        x_t, x_hist, step_sizes, eta  = adagrad_loop(f, phi, x_t, A, N, tol, gamma, params['delta'] , params['n_epochs'], params['batch_size'])
    end = time.time()
    
    if verbose:
        print(f"{name} main loop finished after {end-start} sec")
    
    #########################################################
    x_hist = np.vstack(x_hist)
    n_iter = x_hist.shape[0]
    
    # compute x_mean retrospectivly and evaluate objective
    obj = list(); obj2= list()
    xmean_hist = x_hist.cumsum(axis=0) / (np.arange(n_iter) + 1)[:,np.newaxis]
    x_mean = xmean_hist[-1,:].copy()
    
    if measure:
        # evaluate objective at x_t after every epoch
        for j in np.arange(n_iter):
            obj.append(f.eval(x_hist[j,:]) + phi.eval(x_hist[j,:]))
        
        # evaluate objective at x_mean after every epoch
        for j in np.arange(n_iter):
            obj2.append(f.eval(xmean_hist[j,:]) + phi.eval(xmean_hist[j,:]))
        
        
    # distribute runtime uniformly on all iterations
    runtime = [(end-start)/n_iter]*n_iter
    
    if eta > tol:
        status = 'max iterations reached'
    else:
        status = 'optimal'
        
    print(f"{name} terminated during epoch {n_iter} with tolerance {eta}")
    print(f"{name} status: {status}")
    
    info = {'objective': np.array(obj), 'objective_mean': np.array(obj2), 'iterates': x_hist, 'step_sizes': np.array(step_sizes), \
             'gradient_table': gradients, 'runtime': np.array(runtime)}
    #info = None
    return x_t, x_mean, info


@njit()
def saga_loop(f, phi, x_t, A, N, tol, gamma, gradients, n_epochs, reg = 0):
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    g_sum = (1/N)*gradients.sum(axis = 0)
    
    
    for iter_t in np.arange(N * n_epochs):
        
        if eta <= tol:
            break
             
        # sample, result is ARRAY!
        j = np.random.randint(low = 0, high = N, size = 1)
        
        # compute the gradient, A_j is array of shape (n,1)
        A_j = A[j,:]
        g = A_j.T @ f.g(A_j @ x_t, j)
            
        g_j = gradients[j,:].reshape(-1)
        old_g = (-1) * g_j + g_sum
        w_t = (1-gamma*reg) * x_t - gamma * (g + old_g)
        
        # store new gradient
        gradients[j,:] = g
        g_sum = g_sum - (1/N)*g_j + (1/N)*g
        
        # compute prox step
        x_t = phi.prox(w_t, gamma)
        
        # stop criterion
        if iter_t % N == N-1:
            eta = stop_scikit_saga(x_t, x_old)
            x_old = x_t
            
        # store everything (at end of each epoch)
        if iter_t % N == N-1:
            x_hist.append(x_t)
            step_sizes.append(gamma)
        
    return x_t, x_hist, step_sizes, eta


#%%
@njit()
def adagrad_loop(f, phi, x_t, A, N, tol, gamma, delta, n_epochs, batch_size):
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    
    n = len(x_t)
    G = np.zeros(n)
    
    for iter_t in np.arange(N * n_epochs):
        
        if eta <= tol:
            break
        
        # sample
        S = np.random.randint(low = 0, high = N, size = batch_size)
        S = np.sort(S)
        
        # mini-batch gradient step
        G_t = compute_batch_gradient(f, x_t, S)
        G += G_t * G_t
        
        L_t = (1/gamma) * (delta + np.sqrt(G))
        
        w_t = x_t - (1/L_t) * G_t
        
        # compute Adagrad prox step
        x_t = phi.adagrad_prox(w_t, L_t)
        
        # stop criterion
        if iter_t % N == N-1:
            eta = stop_scikit_saga(x_t, x_old)
            x_old = x_t
            
        # store everything (at end of each epoch)
        if iter_t % N == N-1:
            x_hist.append(x_t)
            step_sizes.append(gamma)

    return x_t, x_hist, step_sizes, eta


# x_t = np.random.rand(100)
# adagrad_loop(f, phi, x_t, f.A, f.N, tol = 1e-3, gamma = .1, delta = 0.1, n_epochs = 2, batch_size = 10)