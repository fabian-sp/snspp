"""
author: Fabian Schaipp
"""

import numpy as np
import time


def saga(f, phi, x0, eps = 1e-3, params = dict(), verbose = False, measure = False):
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
    gradients = list()
    for i in np.arange(N):
        A_i =  A[dims == i].copy()
        tmp_i = A_i.T @ f.g( A_i @ x_t, i)
        gradients.append(tmp_i)
        
    gradients = np.vstack(gradients)
    assert gradients.shape == (N,n)
    
    if 'max_iter' not in params.keys():    
        params['max_iter'] = 70
    
    if 'gamma' not in params.keys():
        L = 2 * np.apply_along_axis(np.linalg.norm, axis = 1, arr = f.A).max()
        gamma = 1./3*L
    else:
        gamma = params['gamma']
    
    # initialize for stopping criterion
    status = 'not optimal'
    eta = np.inf
    
    # initialize for diagnostics
    x_hist = list()
    step_sizes = list()
    obj = list()
    
    hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
    if verbose:
        print(hdr_fmt % ("iter", "obj (x_t)", "obj(x_mean)", "gamma", "eta"))
    
    for iter_t in np.arange(params['max_iter']):
        
        if eta <= eps:
            status = 'optimal'
            break
        
        # sample
        j = np.random.randint(low = 0, high = N, size = 1).squeeze()
        A_j = A[dims == j,:].copy()
        
        # compute the gradient
        g = A_j.T @ f.g(A_j@x_t, j)
        old_g = - gradients[j,:].squeeze() + (1/N)*gradients.sum(axis = 0)
        w_t = x_t - gamma * (g + old_g)
        
        # store new gradient
        gradients[j,:] = g.copy()
        
        # compute prox step
        x_t = phi.prox(w_t, gamma)
        
        
        x_hist.append(x_t)
        step_sizes.append(gamma)
        obj.append(f.eval(x_t) + phi.eval(x_t))
        
        if verbose:
            print(out_fmt % (iter_t, obj[-1], np.pi , gamma, eta))
        
        
        
    if eta > eps:
        status = 'max iterations reached'    
        
    print(f"SAGA terminated after {iter_t} iterations with accuracy {eta}")
    print(f"SAGA status: {status}")
    
    info = {'objective': np.array(obj), 'iterates': np.vstack(x_hist), 'step_sizes': np.array(step_sizes)}
    
    return x_t, info