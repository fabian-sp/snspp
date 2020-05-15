"""
author: Fabian Schaipp
"""

import numpy as np
import time

def sampler(N, size):
    
    assert size <= N, "specified a bigger sample size than N"
    S = np.random.choice(a = np.arange(N), p = (1/N) * np.ones(N), size = size, replace = False)
    
    S = S.astype('int')
    return S


def solve_subproblem(x, xi, alpha, A, m, f, phi, sample_size, newton_params = None, verbose = False):
    
    
    N = len(m)
    # creates a vector with nrows like A in order to index th relevant A_i from A
    dims = np.repeat(np.arange(N),m)
    
    S = sampler(N, sample_size)
    M = m[S].sum()
    
    subA = A[np.isin(dims, S), :]
    
    assert subA.shape[0] == M
    
    xi = np.zeros(M)
    
    condA = False
    condB = False
    
    sub_iter = 0
    
    while not(condA or condB) and sub_iter < 10:
        
        z = x - (alpha/sample_size) * (subA.T @ xi)
    
        U = phi.jacobian_prox(z, alpha = alpha)
    
        (alpha/sample_size) * subA @ U @ subA.T
    
    
    return 1





def stochastic_ssnal(phi, x0, eps = 1e-4, params = None, verbose = False, measure = False):
    
    d = len(x0)
    x_t = x0.copy()
    
    eta = np.inf

    
    
    # initialize 
    status = 'not optimal'
    
    max_iter = 100
    
    # initialize for measurements
    runtime = np.zeros(max_iter)
    
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
            
        if eta <= eps:
            status = 'optimal'
            break
        
        if verbose:
            print(f"------------Iteration {iter_t} of the Stochastic SSNAL algorithm----------------")
    
    
    if eta > eps:
        status = 'max iterations reached'    
        
    print(f"Stochastic SSNAL terminated after {iter_t} iterations with accuracy {eta}")
    print(f"Stochastic SSNAL status: {status}")
    
    
    
    return x_t