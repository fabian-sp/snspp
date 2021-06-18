"""
author: Fabian Schaipp
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import cg
import warnings


def Ueval(xi_sub, f, phi, x, alpha, S, subA, hat_d):
    
    sample_size = len(S)
    
    z = x - (alpha/sample_size) * (subA.T @ xi_sub) + hat_d
    term2 = .5 * np.linalg.norm(z)**2 - phi.moreau(z, alpha)
    
    term1 = f.fstar_vec(xi_sub, S).sum()
    res = term1 + (sample_size/alpha) * term2
    
    return res.squeeze()


    
def solve_subproblem_easy(f, phi, x, xi, alpha, A, S, newton_params = None, reduce_variance = False, xi_tilde = None, full_g = None, verbose = True):
    """
    
    """
    if xi_tilde is None or full_g is None:
        assert not reduce_variance
        
    assert alpha > 0 , "step sizes are not positive"
      
    sample_size = len(S)
    #assert np.all(S == np.sort(S)), "S is not sorted!"
    
    subA = A[S,:]
    xi_sub = xi[S]
    
    sub_iter = 0
    converged = False
    
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
    if not f.convex: 
        gamma_i = f.weak_conv(S)
        hat_d += (alpha/sample_size) * (gamma_i.reshape(1,-1) * subA.T @ (subA @ x))
    
    while sub_iter < newton_params['max_iter']:
        
    # step 1: construct Newton matrix and RHS 
        
        z = x - (alpha/sample_size) * (subA.T @ xi_sub) + hat_d
        rhs = -1. * (f.gstar_vec(xi_sub, S) - subA @ phi.prox(z, alpha))
        
        residual.append(np.linalg.norm(rhs))
        if np.linalg.norm(rhs) <= newton_params['eps']:
            converged = True
            break
        
        U = phi.jacobian_prox(z, alpha)
        
        if phi.name == '1norm':
            # U is 1d array with only 1 or 0 --> speedup by not constructing 2d diagonal array
            bool_d = U.astype(bool)
            
            subA_d = subA[:, bool_d].astype('float32')
            tmp2 = (alpha/sample_size) * subA_d @ subA_d.T
        else:
            tmp2 = (alpha/sample_size) * subA @ U @ subA.T
        
        
        eps_reg = 1e-4
        tmp_d = f.Hstar_vec(xi_sub, S)
        
        tmp = np.diag(tmp_d + eps_reg)           
        W = tmp + tmp2
        assert not np.isnan(W).any(), "Something went wrong during construction of the Hessian"
        
    # step2: solve Newton system
        use_cg = True
        
        if use_cg:
            cg_tol = min(newton_params['eta'], np.linalg.norm(rhs)**(1+ newton_params['tau']))
            precond = None
            d, cg_status = cg(W, rhs, tol = cg_tol, maxiter = 12, M = precond)
        else:
            chol,lower = cho_factor(W)
            d = cho_solve((chol,lower), rhs)
        
        if not d@rhs > -1e-8:
            warnings.warn(f"No descent direction, {d@rhs}")
        
        norm_dir.append(np.linalg.norm(d))

    # step 3: backtracking line search
        
        U_old = Ueval(xi_sub, f, phi, x, alpha, S, subA, hat_d)
        beta = 1.
        U_new = Ueval(xi_sub + beta*d, f, phi, x, alpha, S, subA, hat_d)
           
        while U_new > U_old + newton_params['mu'] * beta * (d @ -rhs):
            beta *= newton_params['rho']
            U_new = Ueval(xi_sub + beta*d, f, phi, x, alpha, S, subA, hat_d)
            
        step_sz.append(beta)
        obj.append(U_new)
        
    # step 4: update xi
        xi_sub += beta * d         
        sub_iter += 1
        
    if not converged and verbose:
        print(f"WARNING: reached max. iter in semismooth Newton with residual {residual[-1]}")
    
    
    # update xi variable
    xi[S] = xi_sub.copy()
    # update primal iterate
    z = x - (alpha/sample_size) * (subA.T @ xi_sub) + hat_d
    new_x = phi.prox(z, alpha)
    
    info = {'residual': np.array(residual), 'direction' : norm_dir, 'step_size': step_sz, 'objective': np.array(obj)}
    
    
    return new_x, xi, info