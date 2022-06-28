"""
author: Fabian Schaipp
"""
from ..helper.utils import compute_gradient_table, compute_batch_gradient, compute_batch_gradient_table, compute_xi_inner,\
                            compute_x_mean_hist, stop_scikit_saga, derive_L

from .sgd import sgd_loop
                            
import numpy as np                           
import time
import warnings

from numba.typed import List
from numba import njit


def vr_default_step_size(f):
    """
    Default step size for SVRG/SAGA. Theoretical step sizes vary for different assumptions on (strong) convexity. See
    - Bach, Defazio 2014: SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives
    - Sra, Reddi 2016: Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization
    - Xiao Zhang 2014: A Proximal Stochastic Gradient Method with Progressive Variance Reduction
    
    For simplicity, we simply take as default step size
    
    alpha = 1/(3L)
    
    where L is a L-smoothness constant for all f_i(A_i .). This choice comes from Bach, Defazio 2014.
    
    """
                
    normA_max =  (np.linalg.norm(f.A, axis = 1)**2).max()
    L = derive_L(f) * normA_max
    alpha = 1./(3*L)
    
    return alpha

def stochastic_gradient(f, phi, x0, solver = 'saga', tol = 1e-3, params = dict(), verbose = False, measure = False):
    """
    fast implementation of first-order methods for problems of the form 
    
    min 1/N * sum f_i(A_i x) + phi(x)
    
    * contains: SAGA, SVRG, AdaGrad, SGD
    * works only if m_i = 1 forall i, i.e. one sample gives one summand!
    * speedup achieved by numba, hence classes f and phi need to be jitted beforehand.
    
    """
    
    name = solver.upper()
        
    # initialize all variables
    n = len(x0)
    assert np.all(f.m==1), "These implementations are restricted to the case m_i = 1, use SNSPP if not the case"
    N = len(f.m)
    A = f.A.astype('float64')
    assert n == A.shape[1], "wrong dimensions"
    
    x_t = x0.copy().astype('float64')

    # initialize object for storing all gradients (used for SAGA)
    gradients = compute_gradient_table(f, x_t).astype('float64')
    assert gradients.shape == (N,n)
    
    if 'n_epochs' not in params.keys():    
        params['n_epochs'] = 50
    
    #########################################################
    ## Solver dependent parameters
    #########################################################
    
    if solver == 'saga':
        if 'reg' not in params.keys():    
            params['reg'] = 0.
    
    elif solver == 'adagrad':
        if 'delta' not in params.keys():    
            params['delta'] = 1e-12
           
    elif solver == 'sgd':
        if 'style' not in params.keys(): 
            params['style'] = 'vanilla'
        if 'beta' not in params.keys(): 
            params['beta'] = 0.51
        assert params['style'] in ['vanilla', 'polyak']
    
    #########################################################
    ## Batch size
    #########################################################
    if solver in ['adagrad', 'svrg', 'sgd', 'batch-saga']:
        if 'batch_size' not in params.keys():    
            params['batch_size'] = max(int(f.N * 0.005), 1)
        
        if solver == 'svrg':
            m_iter = int(N/params['batch_size']) 
            
    #########################################################
    ## Step size 
    #########################################################
    if 'alpha' not in params.keys():
        if solver in ['adagrad', 'sgd']:
            alpha = 1e-3
        else:
            alpha = vr_default_step_size(f)
        
        warnings.warn("Using a default step size. This may lead to divergence (if too big) or slow convergence (if too small). A script for tuning the step size is contained in snspp/experiments/experiment_utils. Provide a step size via params[\"alpha\"].")

    else:
        alpha = params['alpha']
                      
    alpha = np.float64(alpha)  
    
    if verbose :
        print(f"Step size of {solver}: ", alpha)
       
    #########################################################
    ## Main loop
    #########################################################
    start = time.time()
    
    if solver == 'saga':
        # run SAGA with batch size 1
        x_t, x_hist, step_sizes, eta  = saga_loop(f, phi, x_t, A, N, tol, alpha, gradients, params['n_epochs'], params['reg'])     
    elif solver == 'batch-saga':
        x_t, x_hist, step_sizes, eta  = batch_saga_loop(f, phi, x_t, A, N, tol, alpha, gradients, params['n_epochs'], params['batch_size'])
    elif solver == 'svrg':
        x_t, x_hist, step_sizes, eta  = svrg_loop(f, phi, x_t, A, N, tol, alpha, params['n_epochs'], params['batch_size'], m_iter)
    elif solver == 'adagrad':
        x_t, x_hist, step_sizes, eta  = adagrad_loop(f, phi, x_t, A, N, tol, alpha, params['delta'] , params['n_epochs'], params['batch_size'])
    elif solver == 'sgd':
        x_t, x_hist, step_sizes, eta = sgd_loop(f, phi, x_t, tol, alpha, params['beta'], params['n_epochs'], params['batch_size'], \
                                                params['style'])
    else:
        raise NotImplementedError("Not a known solver option!")
    
    end = time.time()
    
    if verbose:
        print(f"{name} main loop finished after {end-start} sec")
    
    #########################################################
    x_hist = np.vstack(x_hist)
    n_iter = x_hist.shape[0]
    
    # compute x_mean retrospectivly (disabled)
    if False:
        xmean_hist = compute_x_mean_hist(np.vstack(x_hist))
        x_mean = xmean_hist[-1,:].copy()
    else:
        xmean_hist = None; x_mean = None
        
    # evaluate objective at x_t after every epoch
    obj = list()
    if measure:
        for j in np.arange(n_iter):
            obj.append(f.eval(x_hist[j,:]) + phi.eval(x_hist[j,:]))
        
        
    # distribute runtime uniformly on all iterations
    runtime = [(end-start)/n_iter]*n_iter
    
    # compute number of gradient evaluations retrospectively
    # one entry in x_hist marks one epoch for all solvers (SVRG has full gradient additionally)
    num_eval = np.ones(n_iter)
    if solver == 'svrg':
        num_eval *= 2.
    
    if eta > tol:
        status = 'max iterations reached'
    else:
        status = 'optimal'
        
    if verbose:
        print(f"{name} terminated during epoch {n_iter} with tolerance {eta}")
        print(f"{name} status: {status}")
    
    info = {'objective': np.array(obj), 'iterates': x_hist, \
            'mean_hist': xmean_hist, 'step_sizes': np.array(step_sizes), \
            'runtime': np.array(runtime), 'evaluations': num_eval}

    return x_t, x_mean, info

#%%
@njit()
def saga_loop(f, phi, x_t, A, N, tol, alpha, gradients, n_epochs, reg):
    
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
        
        # compute the gradient, A_j is array of shape (1,n)
        A_j = A[j,:]
        g = A_j.T @ f.g(A_j @ x_t, j)
            
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


#%%
@njit()
def adagrad_loop(f, phi, x_t, A, N, tol, alpha, delta, n_epochs, batch_size):
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    
    n = len(x_t)
    G = np.zeros(n)
    
    # construct array which tells us when to store (at the END of each epoch)
    max_iter = int(N * n_epochs/batch_size)
    counter = np.arange(max_iter)*batch_size/N % 1
    counter = np.append(counter,0)
    
    store = (counter[1:] <= counter[:-1])
    assert len(store) == max_iter
    assert store[-1]
    
    for iter_t in np.arange(max_iter):
        
        if eta <= tol:
            break
        
        # sample
        S = np.random.randint(low = 0, high = N, size = batch_size)
        #S = np.random.choice(a = np.arange(N), size = batch_size, replace = True)
        
        # mini-batch gradient step
        G_t = compute_batch_gradient(f, x_t, S)
        G += G_t * G_t
        
        L_t = (1/alpha) * (delta + np.sqrt(G))
        
        w_t = x_t - (1/L_t) * G_t
        
        # compute Adagrad prox step
        x_t = phi.adagrad_prox(w_t, L_t)
        
        # stop criterion (at end of each epoch)
        if store[iter_t]:
            eta = stop_scikit_saga(x_t, x_old)
            x_old = x_t
            
        # store everything (at end of each epoch)
        if store[iter_t]:
            x_hist.append(x_t)
            step_sizes.append(np.linalg.norm(1/L_t))

    return x_t, x_hist, step_sizes, eta

#%%
@njit()
def svrg_loop(f, phi, x_t, A, N, tol, alpha, n_epochs, batch_size, m_iter):
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    
    S_iter = int(n_epochs*N / (batch_size*m_iter))
    
    for s in np.arange(S_iter):
        
        if eta < tol:
            break
        #full_g = compute_batch_gradient_table(f, x_t, np.arange(N))  
        full_g = A * compute_xi_inner(f, x_t)
        g_tilde = (1/N) * full_g.sum(axis=0)

        
        for t in np.arange(m_iter):
            
            # np.random.choice is slower than np.random.randint 
            #S = np.random.choice(a = np.arange(N), size = batch_size, replace = True)
            S = np.random.randint(low = 0, high = N, size = batch_size)
            
            # compute the gradient
            v_t = compute_batch_gradient(f, x_t, S)
            g_t = v_t - (1/batch_size) * full_g[S,:].sum(axis=0) + g_tilde

            w_t = x_t - alpha*g_t
            x_t = phi.prox(w_t, alpha)
        
   
        # stop criterion
        eta = stop_scikit_saga(x_t, x_old)
        x_old = x_t
        # store in each outer iteration
        x_hist.append(x_t)    
        step_sizes.append(alpha)    


    return x_t, x_hist, step_sizes, eta

#%%
@njit()
def batch_saga_loop(f, phi, x_t, A, N, tol, alpha, gradients, n_epochs, batch_size):
    
    # initialize for diagnostics
    x_hist = List()
    step_sizes = List()
    
    eta = 1e10
    x_old = x_t
    g_sum = (1/N)*gradients.sum(axis = 0)
    
    max_iter = int(N * n_epochs/batch_size)
    counter = np.arange(max_iter)*batch_size/N % 1
    counter = np.append(counter,0)
    
    store = (counter[1:] <= counter[:-1])
    assert store[-1]
    
    for iter_t in np.arange(max_iter):
        
        if eta <= tol:
            break
             
        # sample
        #S = np.random.choice(a = np.arange(N), size = batch_size, replace = True)
        S = np.random.randint(low = 0, high = N, size = batch_size)
        S = np.sort(S)
        
        # compute the gradient
        batch_g = compute_batch_gradient_table(f, x_t, S)
        batch_g_sum = batch_g.sum(axis=0)
        
        g_j = gradients[S,:].sum(axis=0)
        old_g = (-1/batch_size) * g_j + g_sum
        
        w_t = x_t - alpha * ((1/batch_size)*batch_g_sum + old_g)
        
        # store new gradient
        gradients[S,:] = batch_g
        g_sum = g_sum - (1/N)*g_j + (1/N)*batch_g_sum
        
        # compute prox step
        x_t = phi.prox(w_t, alpha)
        
        # stop criterion
        if store[iter_t]:
            eta = stop_scikit_saga(x_t, x_old)
            x_old = x_t
            
        # store everything (at end of each epoch)
        if store[iter_t]:
            x_hist.append(x_t)
            step_sizes.append(alpha)
        
    return x_t, x_hist, step_sizes, eta

#%%
# @njit()
# def prox_svrg2(f, phi, x_t, A, N, tol, alpha, n_epochs, batch_size, m_iter):
    
#     # initialize for diagnostics
#     x_hist = List()
#     step_sizes = List()
    
#     eta = 1e10
#     x_old = x_t
    
#     S_iter = int(n_epochs*N / (batch_size*m_iter))
    
#     for s in np.arange(S_iter):
          
#         gradient_store = compute_xi_inner(f, x_t)
#         g_tilde = (1/N) * (A*gradient_store).sum(axis=0)

        
#         for t in np.arange(m_iter):
            
#             #S = np.random.choice(a = np.arange(N), size = batch_size, replace = True)
#             S = np.random.randint(low = 0, high = N, size = batch_size)
        
#             # compute the gradient
#             v_t, A_t = compute_svrg_grad(f, x_t, S)
#             #A_t = A[S,:]
#             g_t = (1/batch_size) * (A_t*(v_t - gradient_store[S,:])).sum(axis=0) + g_tilde

#             w_t = x_t - alpha*g_t
#             x_t = phi.prox(w_t, alpha)
        
   
#         # stop criterion
#         eta = stop_scikit_saga(x_t, x_old)
#         x_old = x_t
#         # store in each outer iteration
#         x_hist.append(x_t)    
#         step_sizes.append(alpha)    


#     return x_t, x_hist, step_sizes, eta

# @njit()
# def compute_svrg_grad(f, x, S):
    
#     vals = np.zeros((len(S),1))
#     A_S = np.zeros((len(S), len(x)))
    
#     for i in np.arange(len(S)):
#         A_i = np.ascontiguousarray(f.A[S[i],:]).reshape(1,-1)
#         vals[i,:] = f.g(A_i @ x, S[i])
#         A_S[i,:] = A_i
        
#     return vals, A_S

