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
    
    params = {'rho': .9, 'mu': .25, 'eps': 1e-4, 'max_iter': 15}
    
    return params

def solve_subproblem(f, phi, x, xi, alpha, A, m, S, newton_params = None, verbose = False):
    """
    m: vector with all dimensions m_i, i = 1,..,N
    
    """
    if newton_params is None:
        newton_params = get_default_newton_params()
        
    N = len(m)
    # creates a vector with nrows like A in order to index th relevant A_i from A
    dims = np.repeat(np.arange(N),m)
    
    sample_size = len(S)
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
    print(f"Initial xi_stack: {xi_stack}")
    
    assert np.all([np.all(xi[i] == xi_stack[sub_dims == i]) for i in S]), "Something went wrong in the sorting/stacking of xi"
    assert len(xi_stack) == M
    
    sub_iter = 0
    converged = False
    residual = list()
    
    while sub_iter < newton_params['max_iter']:
        
    # step 1: construct Newton matrix and RHS
        z = x - (alpha/sample_size) * (subA.T @ xi_stack)
        rhs = -1 * (np.hstack([f.gstar(xi[i],i) for i in S]) - subA @ phi.prox(z, alpha))
        
        residual.append(np.linalg.norm(rhs))
        if np.linalg.norm(rhs) <= newton_params['eps']:
            converged = True
            break
        
        U = phi.jacobian_prox(z, alpha = alpha)
        tmp2 = (alpha/sample_size) * subA @ U @ subA.T
        
        tmp = [f.Hstar(xi[i], i) for i in S]
        W = block_diag(tmp) + tmp2
        
        
    # step2: solve Newton system
        if verbose:
            print("Start CG method")
        d, cg_status = cg(W, rhs, tol = 1e-6, maxiter = 100)
        #print(f"Direction: {d}")
        assert cg_status == 0, "CG method did not converge"
    # step 3: backtracking line search
        if verbose:
            print("Start Line search")
        U_old = Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA)
        beta = 1.
        U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA)
        
        while U_new > U_old + newton_params['mu'] * beta * (d @ -rhs):
            beta *= newton_params['rho']
            U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA)
        
        #print(f"Step size: {beta}")
    # step 4: update xi
        if verbose:
            print("Update xi variables")
        xi_stack += beta * d
        for i in S:
            xi[i] = xi_stack[sub_dims == i].copy()
        #print(f"New xi_stack: {xi_stack}")
        sub_iter += 1
        
    if not converged:
        print(f"WARNING: reached maximal iterations in semismooth Newton -- accuracy {residual[-1]}")
        
    # update primal iterate
    z = x - (alpha/sample_size) * (subA.T @ xi_stack)
    new_x = phi.prox(z, alpha)
    
        
    return new_x, xi, np.array(residual)



def stochastic_ssnal(f, phi, x0, A, eps = 1e-4, params = None, verbose = False, measure = False):
    
    n = len(x0)
    x_t = x0.copy()
    alpha_t = 100
    sample_size = min(8, f.N)
    
    # get infos related to structure of f
    m = f.m.copy()
    xi = dict(zip(np.arange(f.N), [np.random.rand(m[i]) for i in np.arange(f.N)]))
    
    # initialize 
    status = 'not optimal'
    max_iter = 100
    eta = np.inf
    
    # initialize for measurements
    runtime = list()
    obj = list()
    hist = x_t.copy()
    
    for iter_t in np.arange(max_iter):
        
        if measure:
            start = time.time()
            
        if eta <= eps:
            status = 'optimal'
            break
        
        if verbose:
            print(f"------------Iteration {iter_t} of the Stochastic SSNAL algorithm----------------")
        
        S = sampler(f.N, sample_size)
        x_t, xi, _ = solve_subproblem(f, phi, x_t, xi, alpha_t, A, m, S, newton_params = None, verbose = False)
        
        hist = np.vstack((hist, x_t))
        
        obj.append(f.eval(x_t))
    
    if eta > eps:
        status = 'max iterations reached'    
        
    print(f"Stochastic SSNAL terminated after {iter_t} iterations with accuracy {eta}")
    print(f"Stochastic SSNAL status: {status}")
    
    info = {'objective': np.array(obj), 'iterates': hist}
    
    return x_t, info