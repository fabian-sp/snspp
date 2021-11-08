"""
author: Fabian Schaipp
"""

import numpy as np
from numba import njit
from scipy.sparse.linalg import cg
import warnings

from utils import multiple_matdot, matdot

#%%
def Ueval(xi_sub, f, phi, x, alpha, S, subA, hat_d):
    
    sample_size = len(S)
    
    adjA_xi = np.dot(subA, xi_sub)
    Z = x - (alpha/sample_size) * adjA_xi + hat_d
    
    term2 = .5 * np.linalg.norm(Z)**2 - phi.moreau(Z, alpha)
    
    term1 = f.fstar_vec(xi_sub, S).sum()
    res = term1 + (sample_size/alpha) * term2
    
    return res.squeeze(), Z

@njit()
def calc_AUA(phi, Z, alpha, subA):
    (p,q,b) = subA.shape
    res = np.zeros((b,b))
    
    for i in np.arange(b):
        for j in np.arange(start = i, stop = b):
            res[i,j] = matdot(subA[:,:,i], phi.jacobian_prox(Z, subA[:,:,j], alpha))
    
    # result is symmetric 
    d = np.diag(res)
    res = res + res.T
    np.fill_diagonal(res, d)
    
    return res
    
def solve_subproblem(f, phi, X, xi, alpha, A, S, newton_params = None, reduce_variance = False, xi_tilde = None, full_g = None, verbose = True):
    """
    m: vector with all dimensions m_i, i = 1,..,N   
    """
    if xi_tilde is None or full_g is None:
        assert not reduce_variance
        
    assert alpha > 0 , "step sizes are not positive"
      
    sample_size = len(S)
    assert np.all(S == np.sort(S)), "S is not sorted!"
    
    subA = A[:,:,S]
    xi_sub = xi[S]
    
    sub_iter = 0
    converged = False
    U_new = None
    
    residual = list()
    norm_dir = list()
    step_sz = list()
    obj = list()
    
    # compute var. reduction term
    if reduce_variance:
        hat_d =  (alpha/sample_size) * (subA.T @ xi_tilde[S]) - alpha * full_g    
    else:
        hat_d = 0.
        
    #compute term coming from weak convexity
    # if not f.convex: 
    #     gamma_i = f.weak_conv(S)
    #     hat_d += (alpha/sample_size) * (gamma_i.reshape(1,-1) * subA.T @ (subA @ X))
    
    #adjA_xi = np.sum((xi_sub[:,np.newaxis,np.newaxis]*subA), axis=0)
    adjA_xi = np.dot(subA, xi_sub)   
    Z = X - (alpha/sample_size) * adjA_xi + hat_d
        
    while sub_iter < newton_params['max_iter']:
        
    # step 1: construct Newton matrix and RHS 
        rhs = -1. * (f.gstar_vec(xi_sub, S) - multiple_matdot(subA, phi.prox(Z, alpha)))
    
        residual.append(np.linalg.norm(rhs))
        if np.linalg.norm(rhs) <= newton_params['eps']:
            converged = True
            break
            
        W2 = (alpha/sample_size) * calc_AUA(phi, Z, alpha, subA)
               
        eps_reg = 1e-4
        tmp_d = f.Hstar_vec(xi_sub, S)
        W1 = np.diag(tmp_d + eps_reg)           
        
        W = W1 + W2
        assert not np.isnan(W).any(), "Something went wrong during construction of the Hessian"
        
    # step2: solve Newton system
        cg_tol = min(newton_params['eta'], np.linalg.norm(rhs)**(1+ newton_params['tau']))      
        precond = None      
        d, cg_status = cg(W, rhs, tol = cg_tol, maxiter = 12, M = precond)
        
        if not d@rhs > -1e-8:
            warnings.warn(f"No descent direction, {d@rhs}")
        norm_dir.append(np.linalg.norm(d))

    # step 3: backtracking line search
        if sub_iter > 0:
            U_old = U_new
        else:
            U_old, _ = Ueval(xi_sub, f, phi, X, alpha, S, subA, hat_d)
    
        beta = 1.
        U_new, Z = Ueval(xi_sub + beta*d, f, phi, X, alpha, S, subA, hat_d)
           
        while U_new > U_old + newton_params['mu'] * beta * (d @ -rhs):
            beta *= newton_params['rho']
            U_new, Z = Ueval(xi_sub + beta*d, f, phi, X, alpha, S, subA, hat_d)
            
        step_sz.append(beta)
        obj.append(U_new)
        
    # step 4: update xi
        xi_sub += beta * d         
        sub_iter += 1
        
    if not converged and verbose:
        print(f"WARNING: reached max. iter in semismooth Newton with residual {residual[-1]}")
    
    
    # update xi variable
    xi[S] = xi_sub.copy()
    new_X = phi.prox(Z, alpha)
    
    info = {'residual': np.array(residual), 'direction' : norm_dir, 'step_size': step_sz, 'objective': np.array(obj)}
    
    
    return new_X, xi, info