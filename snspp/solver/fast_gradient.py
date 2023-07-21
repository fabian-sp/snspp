"""
author: Fabian Schaipp
"""
from ..helper.utils import compute_batch_gradient_table, compute_xi_inner,\
                            stop_scikit_saga, derive_L, compute_fnat

from .sgd import sgd_loop
from .sparse.sparse_utils import create_csr                         
from .sparse.prox_gd import sparse_svrg_epoch, sparse_saga_epoch, sparse_adagrad_epoch, sparse_batch_saga_epoch

import numpy as np                           
import time
import warnings

from numba.typed import List
from numba import njit

from scipy.sparse.csr import csr_matrix


def vr_default_step_size(f, A):
    """
    Default step size for SVRG/SAGA. Theoretical step sizes vary for different assumptions on (strong) convexity. See
    - Bach, Defazio 2014: SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives
    - Sra, Reddi 2016: Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization
    - Xiao Zhang 2014: A Proximal Stochastic Gradient Method with Progressive Variance Reduction
    
    For simplicity, we simply take as default step size
    
    alpha = 1/(3L)
    
    where L is a L-smoothness constant for all f_i(A_i .). This choice comes from Bach, Defazio 2014.
    
    """
                
    normA_max =  (np.linalg.norm(A, axis = 1)**2).max()
    L = derive_L(f) * normA_max
    alpha = 1./(3*L)
    
    return alpha

def stochastic_gradient(f, phi, A, x0, solver = 'saga', tol = 1e-3, params = dict(), verbose = False, measure = False):
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
    assert n == A.shape[1], "wrong dimensions"
    
    # check whether A is in sparse format
    if isinstance(A, csr_matrix):
        sparse_format = True
    else:
        sparse_format = False

    
    x_t = x0.copy().astype('float64')
    
    if 'n_epochs' not in params.keys():    
        params['n_epochs'] = 50
    
    #########################################################
    ## Solver dependent parameters
    #########################################################
    
    if solver in ['saga', 'batch-saga']:
        if 'reg' not in params.keys():    
            params['reg'] = 0.
        
    if solver in ['saga', 'svrg', 'batch-saga']:
        if 'measure_freq' not in params.keys():
            # one means measure once per epoch
            # higher means higher frequency 
            params['measure_freq'] = 1
        else:
            assert params['measure_freq'] >= 1
    
    elif solver == 'adagrad':
        if 'delta' not in params.keys():    
            params['delta'] = 1e-12
           
    elif solver == 'sgd':
        if 'beta' not in params.keys(): 
            params['beta'] = 0.51
    
    #########################################################
    ## Batch size
    #########################################################
    if solver in ['adagrad', 'svrg', 'sgd']:
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
            if not sparse_format:
                alpha = vr_default_step_size(f, A)
            else:
                alpha = 1e-3
                
        warnings.warn(f"Using a default step size of {alpha}. This may lead to divergence (if too big) or slow convergence (if too small). You can provide a step size via params[\"alpha\"].")

    else:
        alpha = params['alpha']
                      
    alpha = np.float64(alpha)  
    
    if verbose :
        print(f"Step size of {solver}: ", alpha)
       
    #########################################################
    ## Main loop
    #########################################################
    start1 = time.time()
    
    if solver == 'saga':
        # run SAGA with batch size 1
        x_t, x_hist, runtime, step_sizes, eta  = saga_loop(f, phi, x_t, A, N, tol, alpha, params['n_epochs'], params['reg'], params['measure_freq'], sparse_format, params.get('batch_size'))     
    #elif solver == 'batch-saga':
    #    # run SAGA with batch size > 1
    #    x_t, x_hist, runtime, step_sizes, eta  = saga_loop(f, phi, x_t, A, N, tol, alpha, params['n_epochs'], params['reg'], params['measure_freq'], sparse_format, params['batch_size'])
    elif solver == 'svrg':
        x_t, x_hist, runtime, step_sizes, eta  = svrg_loop(f, phi, x_t, A, N, tol, alpha, params['n_epochs'], params['batch_size'], m_iter, params['measure_freq'], sparse_format)
    elif solver == 'adagrad':
        x_t, x_hist, runtime, step_sizes, eta  = adagrad_loop(f, phi, x_t, A, N, tol, alpha, params['delta'] , params['n_epochs'], params['batch_size'], sparse_format)
    elif solver == 'sgd':
        x_t, x_hist, runtime, step_sizes, eta = sgd_loop(f, phi, x_t, A, N, tol, alpha, params['beta'], params['n_epochs'], params['batch_size'])
    #elif solver == 'tick-svrg':
    #    assert sparse_format
    #    x_t, x_hist, runtime, step_sizes, eta  = solve_with_tick(f, phi, A, alpha, params['n_epochs'], tol, verbose)
    else:
        raise NotImplementedError("Not a known solver option!")
    
    end1 = time.time()
    
    if verbose:
        print(f"{name} main loop finished after {end1-start1} sec")
    
    #########################################################
    x_hist = np.vstack(x_hist)
    n_iter = x_hist.shape[0]
        
    # evaluate objective at x_t after every epoch
    obj = list()
    fnat = list()
    if measure:
        for j in np.arange(n_iter):
            obj.append(f.eval(A @ x_hist[j,:]) + phi.eval(x_hist[j,:]))
            fnat.append(compute_fnat(f, phi, x_hist[j,:], A))
        
    
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
            'step_sizes': np.array(step_sizes), \
            'runtime': np.array(runtime), 'evaluations': num_eval,
            'fnat': np.array(fnat)}
    
        
    return x_t, info

#%%
def saga_loop(f, phi, x_t, A, N, tol, alpha, n_epochs, reg, measure_freq, sparse_format, batch_size=None):
    
    # initialize for diagnostics
    runtime = List()
    x_hist = List()
    step_sizes = List()
    
    # sparse format
    if sparse_format:
        A_csr = create_csr(A)
        
    eta = 1e10
    x_old = x_t
    
    s0 = time.time()
    gradients = compute_xi_inner(f, A@x_t)
    g_sum = (1/N) * (A.T @ gradients)
    e0 = time.time()
    
    # measure_freq = how many measurements per epoch
    if batch_size is None:
        loop_length = int(N/measure_freq)
        max_iter = int((N*n_epochs)/loop_length)
    else:
        loop_length = int(N/(batch_size*measure_freq))
        max_iter = int((N*n_epochs)/(loop_length*batch_size))

    for iter_t in np.arange(max_iter):
        
        if eta <= tol:
            break
        
        if batch_size is None:
            if sparse_format:
                start = time.time()
                x_t, g_sum = sparse_saga_epoch(f, phi, x_t, A_csr, N, alpha, gradients, reg, g_sum, loop_length)
                end = time.time()
            else:    
                start = time.time()
                x_t, g_sum = saga_epoch(f, phi, x_t, A, N, alpha, gradients, reg, g_sum, loop_length)
                end = time.time()
        else:
            if sparse_format:
                start = time.time()
                x_t, g_sum = sparse_batch_saga_epoch(f, phi, x_t, A_csr, N, alpha, gradients, reg, g_sum, loop_length, batch_size)
                end = time.time()
            else:    
                start = time.time()
                x_t, g_sum = batch_saga_epoch(f, phi, x_t, A, N, alpha, gradients, reg, g_sum, loop_length, batch_size)
                end = time.time()


        if iter_t == 0:
            runtime.append(end-start+e0-s0)
        else:
            runtime.append(end-start)
            
        eta = stop_scikit_saga(x_t, x_old)
        x_old = x_t
            
        x_hist.append(x_t)
        step_sizes.append(alpha)
        
    return x_t, x_hist, runtime, step_sizes, eta

@njit()
def saga_epoch(f, phi, x_t, A, N, alpha, gradients, reg, g_sum, loop_length):
        
    for iter_t in np.arange(loop_length):
                     
        # sample, result is int!
        j = np.random.randint(low = 0, high = N, size = 1)[0]
        
        # compute the gradient, 
        A_j = A[j,:]
        new_g_j =  f.g(np.dot(A_j, x_t), j)
        
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

def adagrad_loop(f, phi, x_t, A, N, tol, alpha, delta, n_epochs, batch_size, sparse_format):
    
    # initialize for diagnostics
    x_hist = List()
    runtime = List()
    step_sizes = List()
    
    # sparse format
    if sparse_format:
        A_csr = create_csr(A)
        
    eta = 1e10
    x_old = x_t
    
    n = len(x_t)
    G = np.zeros(n)
    
    epoch_iter = int(N/batch_size)
    
    for iter_t in np.arange(n_epochs):
        
        if eta <= tol:
            break
        if sparse_format:
            start = time.time()
            x_t, G = sparse_adagrad_epoch(f, phi, x_t, A_csr, N, alpha, delta, epoch_iter, batch_size, G)
            end = time.time()
        else:
            start = time.time()
            x_t, G = adagrad_epoch(f, phi, x_t, A, N, alpha, delta, epoch_iter, batch_size, G)
            end = time.time()
        
        # stop criterion (at end of each epoch)
        eta = stop_scikit_saga(x_t, x_old)
        x_old = x_t
        
        x_hist.append(x_t)
        runtime.append(end-start)
        step_sizes.append(alpha)

    return x_t, x_hist, runtime, step_sizes, eta

@njit()
def adagrad_epoch(f, phi, x_t, A, N, alpha, delta, epoch_iter, batch_size, G):
    for j in np.arange(epoch_iter):
        # sample
        S = np.random.randint(low = 0, high = N, size = batch_size)
        #S = np.random.choice(a = np.arange(N), size = batch_size, replace = True)
        
        # mini-batch gradient step
        A_S = A[S,:]
        G_t = (1/batch_size) * (A_S.T @ f.g(A_S@x_t, S))
        G += G_t * G_t
        
        L_t = (1/alpha) * (delta + np.sqrt(G))
        
        w_t = x_t - (1/L_t) * G_t
        
        # compute Adagrad prox step
        x_t = phi.adagrad_prox(w_t, L_t)

    return x_t, G

#%%

def svrg_loop(f, phi, x_t, A, N, tol, alpha, n_epochs, batch_size, m_iter, measure_freq, sparse_format):
    
    # initialize for diagnostics
    x_hist = List()
    runtime = List()
    step_sizes = List()
    
    # sparse format
    if sparse_format:
        A_csr = create_csr(A)
    
    eta = 1e10
    x_old = x_t
    
    S_iter = int(n_epochs*N / (batch_size*m_iter))
    
    assert m_iter >= measure_freq, "measuring frequency is too high"
    loop_length = int(m_iter/measure_freq)
    
    for s in np.arange(S_iter):
        
        if eta < tol:
            break
        
        s0 = time.time()
        xis = compute_xi_inner(f, A@x_t)
        full_g = (1/N) * (A.T @ xis)
        e0 = time.time()

        # split up one inner lop in measure_freq parts        
        for j in np.arange(measure_freq):   
            
            if sparse_format:
                s1 = time.time()
                x_t = sparse_svrg_epoch(f, phi, x_t, A_csr, N, alpha, batch_size, loop_length, xis, full_g)
                e1 = time.time()
            else:    
                s1 = time.time()
                x_t = svrg_epoch(f, phi, x_t, A, N, alpha, batch_size, loop_length, xis, full_g)
                e1 = time.time()
                
            # store
            x_hist.append(x_t)    
            if j == 0:
                runtime.append(e1-s1+e0-s0)
            else:
                runtime.append(e1-s1)
        
        # stop criterion
        eta = stop_scikit_saga(x_t, x_old)
        x_old = x_t
        step_sizes.append(alpha)    

    return x_t, x_hist, runtime, step_sizes, eta

@njit()
def svrg_epoch(f, phi, x_t, A, N, alpha, batch_size, loop_length, xis, full_g):
    
    for t in np.arange(loop_length):
            
        # np.random.choice is slower than np.random.randint 
        #S = np.random.choice(a = np.arange(N), size = batch_size, replace = True)
        S = np.random.randint(low = 0, high = N, size = batch_size)
        
        # compute the gradient
        A_S = A[S,:]
        v_t = f.g(A_S@x_t, S)
        g_t = (1/batch_size) * A_S.T @ (v_t - xis[S]) + full_g

        w_t = x_t - alpha*g_t
        x_t = phi.prox(w_t, alpha)
        
    return x_t
        
#%%

@njit()
def batch_saga_epoch(f, phi, x_t, A, N, alpha, gradients, reg, g_sum, loop_length, batch_size):
    
    for iter_t in np.arange(loop_length):
                     
        # sample, result is int!
        S = np.random.randint(low = 0, high = N, size = batch_size)
        
        # compute the gradient, 
        A_S = A[S,:]
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



