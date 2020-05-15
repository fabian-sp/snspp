"""
author: Fabian Schaipp
"""

import numpy as np
from basic_linalg import block_diag
import time

def sampler(N, size):
    
    assert size <= N, "specified a bigger sample size than N"
    S = np.random.choice(a = np.arange(N), p = (1/N) * np.ones(N), size = size, replace = False)
    
    S = S.astype('int')
    # sort S in order to avoid problems with indexing later on
    S = np.sort(S)
    
    return S


def solve_subproblem(x, xi, alpha, A, m, f, phi, sample_size, newton_params = None, verbose = False):
    """
    m: vector with all dimensions m_i, i = 1,..,N
    
    """
    
    N = len(m)
    # creates a vector with nrows like A in order to index th relevant A_i from A
    dims = np.repeat(np.arange(N),m)
    
    S = sampler(N, sample_size)
    # dimension of the problem induced by S
    M = m[S].sum()
    
    # IMPORTANT: subA is ordered, i.e. it is in the order as np.arange(N) and NOT of S 
    subA = A[np.isin(dims, S), :]
    
    assert subA.shape[0] == M
   
    xi = dict(zip(np.arange(N), [np.random.rand(m[i]) for i in np.arange(N)]))
    
    xi_stack = np.hstack([xi[i] for i in S])
    assert len(xi_stack) == M
    
    condA = False
    condB = False
    
    sub_iter = 0
    
    while not(condA or condB) and sub_iter < 10:
        
    # step 1: construct Newton matrix
        z = x - (alpha/sample_size) * (subA.T @ xi_stack)
        U = phi.jacobian_prox(z, alpha = alpha)
    
        tmp2 = (alpha/sample_size) * subA @ U @ subA.T
        gstar, Hstar = f.oracle(S)
        # ATTENTION: this produces wrong order if S is not sorted!!!
        tmp = [Hstar[i](xi[i]) for i in S]
        
        W = block_diag(tmp) + tmp2
        
    # step2: solve Newton system
        
    # step 3: backtracking line search
        
    # step 4: update xi
        
        
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