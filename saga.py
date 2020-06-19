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
    
    # object for storing all gradients
    gradients = np.zeros(N, n)
    # creates a vector with nrows like A in order to index the relevant A_i from A
    dims = np.repeat(np.arange(N),m)
    
    if 'max_iter' not in params.keys():    
        params['max_iter'] = 70
    
    if 'gamma_0' not in params.keys():
        gamma = 1.
    else:
        gamma = params['gamma_0']
        
    
    for iter_t in np.arange(params['max_iter']):
        
        # sample
        j = np.random.randint(low = 0, high = N, size = 1).squeeze()
        A_j = A[dims == j,:].copy()
        
        # compute the gradient
        g = A_j.T @ f.g(A_j@x_t, j)
        
        old_g = gradients[j,:].squeeze()
        
        
        w_t = x_t - gamma * g