"""
author: Fabian Schaipp
"""
from ..helper.utils import compute_gradient_table, compute_batch_gradient, compute_batch_gradient_table, compute_xi_inner,\
                            compute_x_mean_hist, stop_scikit_saga

from .sgd import sgd_loop
                            
import numpy as np                           
import time
import warnings

from numba.typed import List
from numba import njit

def derive_L(f):
    """
    Given a loss f, calculates the L-smoothness constant.
    """
    
    normA =  (np.apply_along_axis(np.linalg.norm, axis = 1, arr = f.A)**2).max()
    
    if f.name == 'squared':
        L = 2 * normA   
    elif f.name == 'logistic':
        L = .25 * normA
    elif f.name == 'tstudent':
        L =  (2/f.v) * normA
    else:
        warnings.warn("For the given loss f, we could not determine the correct Lischitz smoothness constant. The default step size is maybe too large (divergence) or too small (slow convergence).")
        L = 1e2
    
    return L

def saga_theretical_step_size(L, reg = 0, N = 1):
    
    if reg > 0:
        alpha_1 = 1/(2*(N*reg + L))
    else:
        alpha_1 = 0
                
    alpha = max(alpha_1, 1./(3*L))
    
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

    # initialize object for storing all gradients 
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
        if 'batch_size' not in params.keys():    
            params['batch_size'] = max(int(f.N * 0.01), 1)
    
    elif solver == 'sgd':
        if 'batch_size' not in params.keys():    
            params['batch_size'] = max(int(f.N * 0.01), 1)
        if 'style' not in params.keys(): 
            params['style'] = 'vanilla'
        
        assert params['style'] in ['vanilla', 'polyak']
            
    #########################################################
    ## Step size + batch size
    #########################################################
    # see Defazio et al. 2014 for (convex) SAGA step size and Sra, Reddi et al 2016 for minibatch SAGA and PROX-SVRG step size
    if 'alpha' not in params.keys():
        if solver == 'adagrad':
            alpha_0 = 0.001
            warnings.warn("Using a default step size for AdaGrad. This may lead to bad performance. A script for tuning the step size is contained in snspp/experiments/experimnet_utils. Provide a step size via params[\"alpha\"].")
        else:
            alpha_0 = 1.
    else:
        alpha_0 = params['alpha']
    
    # for SAGA/SVRG we use the theoretical step size * alpha_0
    #########################################################
    ## SAGA/SVRG
    #########################################################
    if solver in ['saga', 'batch saga', 'svrg']:
        if f.name == 'squared':
            L = 2 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)**2).max()    
        elif f.name == 'logistic':
            L = .25 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)**2).max()
        elif f.name == 'tstudent':
            L =  (2/f.v) * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)**2).max()
        else:
            warnings.warn("We could not determine the correct Lischitz smoothness constant! The default step size is maybe too large (divergence) or too small (slow convergence).")
            L = 1e2
        
        if solver == 'saga':
            # if we regularize, f_i is strongly-convex and we can use a larger step size (see Defazio et al.)
            if params['reg'] > 0:
                alpha_1 = 1/(2*(f.N*params['reg'] + L))
            else:
                alpha_1 = 0
                
            alpha = alpha_0 * max(alpha_1, 1./(3*L))
            
        elif solver == 'batch saga':
            if 'batch_size' not in params.keys():    
                params['batch_size'] = int(f.N**(2/3))
            alpha = alpha_0 * 1./(5*L)
        
        elif solver == 'svrg':
            if 'batch_size' not in params.keys():    
                params['batch_size'] = 1
            alpha = alpha_0 * 1./(3*L)
            m_iter = int(N/params['batch_size']) 
            
    #########################################################
    ## ADAGRAD
    #########################################################
    elif solver == 'adagrad':
        # for ADAGRAD we use the step size alpha_0
        alpha = alpha_0
    
    #########################################################
    ## SGD
    #########################################################
    elif solver == 'sgd':
        alpha = alpha_0
        if 'beta' not in params.keys(): 
            params['beta'] = 0.51
            
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
    elif solver == 'batch saga':
        # run SAGA with batch size n^(2/3)
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

