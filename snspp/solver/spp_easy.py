"""
author: Fabian Schaipp
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import cg
import warnings


def Ueval(xi_sub, f, phi, x, alpha, S, subA, hat_d):
    """
    This functions evaluates the objective of the subproblem :math:`\mathcal{U}` at the point ``xi_sub``.
    """
    sample_size = len(S)
    
    z = x - (alpha/sample_size) * (subA.T @ xi_sub) + hat_d
    term2 = .5 * np.linalg.norm(z)**2 - phi.moreau(z, alpha)
    
    term1 = f.fstar_vec(xi_sub, S).sum()
    res = term1 + (sample_size/alpha) * term2
    
    return res.squeeze(), z


    
def solve_subproblem_easy(f, phi, x, xi, alpha, A, S, tol = 1e-3, newton_params = None, reduce_variance = False, xi_tilde = None, full_g = None, verbose = True):
    """
    This function solves the subproblem in each SNSPP iteration. 
    The stopping criterion is reached when the norm of the gradient is below ``tol`` or when the maximum number of iterations is reached.
    
    Parameters
    ----------
    f : loss function object
        This object describes the function :math:`f(x)`. 
        See ``snspp/helper/loss1.py`` for an example.
    phi : regularization function object
        This object describes the function :math:`\phi(x)`.
        See ``snspp/helper/regz.py`` for an example.
    x : array of shape (n,)
        Current iterate.
    xi : array of shape (N,)
        Current dual iterates.
    alpha : float
        Step size.
    A : array of shape (N,n)
        Input matrix.
    S : array
        Mini-batch (should be sorted and has possibly duplicate entries).
    tol : float, optional
        Tolerance for stopping. The default is 1e-3.
    newton_params : dict, optional
        Parameters for the semismooth Newton method for the subproblem. See ``get_default_newton_params()`` in ``/spp_solver.py`` for the default values.
    reduce_variance : boolean, optional
        Whether variance reduction is used. The default is False.
    xi_tilde : array of shape (N,), optional
        If VR is enabled, this is given by :math:`\nabla f_i(A_i \tilde{x}),~~ i=1,\dots,N`.
    full_g : array of shape (n,), optional
        If VR is enabled, this is given by :math:`\nabla f(\tilde{x})`. It is precomputed once for every inner loop in ``stochastic_proximal_point()``.
    verbose : boolean, optional
        Verbosity for the subproblem. The default is True.

    Returns
    -------
    new_x : array of shape (n,)
        New primal iterate.
    xi : array of shape (N,)
        Updated dual iterates.
    info : dict
        Information on the convergence of the subproblem.

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
    U_new = None
    
    residual = list()
    norm_dir = list()
    step_sz = list()
    obj = list()
    num_eval = list()
    
    # compute var. reduction term
    if reduce_variance:
        hat_d =  (alpha/sample_size) * (subA.T @ xi_tilde[S]) - alpha * full_g    
    else:
        hat_d = 0.
        
    #compute term coming from weak convexity
    if not f.convex: 
        gamma_i = f.weak_conv(S)
        hat_d += (alpha/sample_size) * (gamma_i.reshape(1,-1) * subA.T @ (subA @ x))
    
    z = x - (alpha/sample_size) * (subA.T @ xi_sub) + hat_d
    
    while sub_iter < newton_params['max_iter']:
        
    # step 1: construct Newton matrix and RHS     
        rhs = -1. * (f.gstar_vec(xi_sub, S) - subA @ phi.prox(z, alpha))
              
        residual.append(np.linalg.norm(rhs))
        if np.linalg.norm(rhs) <= tol:
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
        #assert not np.isnan(W).any(), "Something went wrong during construction of the Hessian"
        
    # step2: solve Newton system
        use_cg = True
        
        if use_cg:
            cg_tol = min(newton_params['eta'], np.linalg.norm(rhs)**(1+ newton_params['tau']))
            precond = None
            d, cg_status = cg(W, rhs, tol = cg_tol, maxiter = 12, M = precond)
        else:
            chol,lower = cho_factor(W)
            d = cho_solve((chol,lower), rhs)
        
        #if not d@rhs > -1e-8:
        #    warnings.warn(f"No descent direction, {d@rhs}")
        
        norm_dir.append(np.linalg.norm(d))

    # step 3: backtracking line search
        
        # U_old is the last U_new
        if sub_iter > 0:
            U_old = U_new
        else:
            U_old, _ = Ueval(xi_sub, f, phi, x, alpha, S, subA, hat_d)
        
        beta = 1.
        U_new, z = Ueval(xi_sub + beta*d, f, phi, x, alpha, S, subA, hat_d)
           
        while U_new > U_old + newton_params['mu'] * beta * (d @ -rhs):
            beta *= newton_params['rho']
            U_new, z = Ueval(xi_sub + beta*d, f, phi, x, alpha, S, subA, hat_d)
            
        step_sz.append(beta)
        obj.append(U_new)
        # 2 from Hstar, gstar, rest from fstar during Armijo
        num_eval.append((2+ np.log(beta)/np.log(newton_params['rho'])) * sample_size )
        
    # step 4: update xi
        xi_sub += beta * d         
        sub_iter += 1
        
    if not converged and verbose:
        warnings.warn(f"Reached max. iter in semismooth Newton with residual {residual[-1]}")
    
    
    # update xi variable
    xi[S] = xi_sub.copy()
    # update primal iterate
    #z = x - (alpha/sample_size) * (subA.T @ xi_sub) + hat_d
    new_x = phi.prox(z, alpha)
    
    info = {'residual': np.array(residual), 'direction' : norm_dir, 'step_size': step_sz, \
            'objective': np.array(obj), 'evaluations': np.array(num_eval)}
    
    
    return new_x, xi, info