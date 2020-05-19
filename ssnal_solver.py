"""
author: Fabian Schaipp
"""

import numpy as np
from basic_linalg import block_diag
from scipy.sparse.linalg import cg
import time

def sampler(N, size):
    """
    samples a subset of {1,..,N} without replacement
    """
    assert size <= N, "specified a bigger sample size than N"
    S = np.random.choice(a = np.arange(N), p = (1/N) * np.ones(N), size = size, replace = False)
    
    S = S.astype('int')
    # sort S in order to avoid problems with indexing later on
    S = np.sort(S)
    
    return S


def Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA):
    
    sample_size = len(S)
    
    z = x - (alpha/sample_size) * (subA.T @ xi_stack)
    tmp = .5 * np.linalg.norm(z)**2 - phi.moreau(z, alpha)
    
    res = sum([f.fstar(xi_stack[sub_dims == i], i) for i in S]) + (sample_size/alpha) * tmp
    
    return res.squeeze()

def get_default_newton_params():
    
    params = {'rho': .5, 'mu': .25}
    
    return params

def solve_subproblem(f, phi, x, xi, alpha, A, m, sample_size, newton_params = None, verbose = False):
    """
    m: vector with all dimensions m_i, i = 1,..,N
    
    """
    if newton_params is None:
        newton_params = get_default_newton_params()
        
    N = len(m)
    # creates a vector with nrows like A in order to index th relevant A_i from A
    dims = np.repeat(np.arange(N),m)
    
    S = sampler(N, sample_size)
    assert np.all(S == np.sort(S)), "S is not sorted!"
    # dimension of the problem induced by S
    M = m[S].sum()
    
    # IMPORTANT: subA is ordered, i.e. it is in the order as np.arange(N) and NOT of S --> breaks if S not sorted
    subA = A[np.isin(dims, S), :]
    #alternatively (safer but slower): subA = np.vstack([A[dims == i,:] for i in S])
    
    assert subA.shape[0] == M
    assert np.all(list(xi.keys()) == np.arange(N)), "xi has wrong keys"
    # sub_dims is helper array to index xi_stack wrt to the elements of S
    sub_dims = np.repeat(S, m[S])
    xi_stack = np.hstack([xi[i] for i in S])
    
    assert np.all([np.all(xi[i] == xi_stack[sub_dims == i]) for i in S]), "Something went wrong in the sorting/stacking of xi"
    assert len(xi_stack) == M
    
    condA = False
    condB = True
    
    sub_iter = 0
    
    while not(condA and condB) and sub_iter < 10:
        
    # step 1: construct Newton matrix and RHS
        z = x - (alpha/sample_size) * (subA.T @ xi_stack)
        U = phi.jacobian_prox(z, alpha = alpha)
        tmp2 = (alpha/sample_size) * subA @ U @ subA.T
        
        tmp = [f.Hstar(xi[i], i) for i in S]
        W = block_diag(tmp) + tmp2
        rhs = -1 * (np.hstack([f.gstar(xi[i],i) for i in S]) - subA @ phi.prox(z, alpha))
        
    # step2: solve Newton system
        if verbose:
            print("Start CG method")
        d, cg_status = cg(W, rhs, tol = 1e-4, maxiter = 100)
        assert cg_status == 0, "CG method did not converge"
    # step 3: backtracking line search
        if verbose:
            print("Start Line search")
        U_old = Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA)
        beta = newton_params['rho']
        U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA)
        
        while U_new > U_old + newton_params['mu'] * beta * (d @ -rhs):
            beta *= newton_params['rho']
            U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA)
                   
    # step 4: update xi
        if verbose:
            print("Update xi variables")
        xi_stack += beta * d
        for i in S:
            xi[i] = xi_stack[sub_dims == i]
        
        sub_iter += 1
        
    if verbose and not(condA and condB):
        print("Subproblem could not be solve with the given accuracy! -- reached maximal iterations")
        
    
        
    return xi



def stochastic_ssnal(f, phi, x0, eps = 1e-4, params = None, verbose = False, measure = False):
    
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