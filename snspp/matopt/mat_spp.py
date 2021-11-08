"""
author: Fabian Schaipp
"""

import numpy as np
from numba import njit
from scipy.sparse.linalg import cg
import warnings
import time

from .utils import multiple_matdot, matdot
from ..solver.spp_solver import sampler, batch_size_constructor, get_default_newton_params, check_newton_params
from ..helper.utils import stop_scikit_saga

def get_default_spp_params():
    de = {'alpha': 1., 'max_iter': 100, 'batch_size': 10, 'sample_style': 'constant', 'reduce_variance': False,\
           'm_iter': 10, 'newton_params': get_default_newton_params()}
    
    return de

def get_xi_start_point(f):
    if f.name == 'mat_squared':
        xi = np.ones(f.N)
    else:
        xi =  np.ones(f.N)

    return xi

def stochastic_prox_point(f, phi, X0, xi = None, tol = 1e-3, params = dict(), verbose = False, measure = False, store_hist = True):
    """
    This implements the semismooth Newton stochastic proximal point method (SNSPP) for solving 
    
    .. math::
        \min{X} f(X) + \phi(X)
        
    where 
    
    .. math::
        f(X) = \frac{1}{N} \sum_{i=1}^{N} f_i(<A_i,X>)
    
    and where :math:`X\in \mathbb{R}{p\times q}`.We assume that each :math:`f_i` maps to :math:`\mathbb{R}`.
    In this case, the dual variables :math:`\\xi` wil be an array of length :math:`N`.
    
    SNSPP is executed with constant step sizes if variance reduction is allowed, i.e. ``params['reduce_variance']==True``.
    If variance reduction is disabled, the step sizes are chosen as 
    
    .. math::
        \alpha_k = \frac{\alpha}{0.51^k}
    
    where ::math::`\alpha` can be specified in ``params``.
    
    Parameters
    ----------
    f : loss function object
        This object describes the function :math:`f(x)`. 
        See ``snspp/matopt/mat_loss.py`` for an example.
    phi : regularization function object
        This object describes the function :math:`\phi(x)`.
        See ``snspp/matopt/nuclear.py`` for an example.
    X0 : array of shape (p,q)
        Starting point. If no better guess is available, use zero.
    xi : array, optional
        Starting values for the dual variables :math:`\\xi`. The default is None.
    tol : float, optional
        Tolerance for the stopping criterion (sup-norm of relative change in the coefficients). The default is 1e-3.
    params : dict, optional
        Dictionary with all parameters. Possible keys:
            
            * ``max_iter``: int, maximal number of iteartions.
            * ``batch_size``: int, batch size.
            * ``alpha``: float, step size of the algorithm.
            * ``reduce_variance``: boolean. Whether to use VR or not (default = True).
            * ``m_iter``: int, number of iteration for inner loop (if VR is used).
            * ``newton_params``: parameters for the semismooth Newton method for the subproblem. See ``get_default_newton_params`` for the default values.
            * ``sample_style``: str, can be 'constant' or 'fast_increasing'. THe latter will increase the batch size over the first iterations which can be more efficient.
            
    verbose : boolean, optional
        Verbosity. The default is False.
    measure : boolean, optional
        Whether to evaluate the objective after each itearion. The default is False.
        For the experiments, needs to be set to ``True``, for actual computation it is recommended to set this to ``False``.
    store_hist : boolean, optional
        Whether to store the history of iterates (might cause memory issues for large problems). The default is True.
         

    Returns
    -------
    X_t : array of shape (p,q)
        Final iterate.
    info : dict
        Information on objective, runtime and subproblem convergence.
    
    """    
    
    A = f.A.copy()
    (p,q) = X0.shape
    assert (p,q) == A.shape[0:2], f"Starting point has wrong dimension {(p,q)} while matrices A_i have dimension {A.shape[0:2]}."
    
    X_t = X0.copy()
    
    status = 'not optimal'
    eta = np.inf
    
    
    #########################################################
    ## Set parameters
    #########################################################
    params_def = get_default_spp_params()
    params_def.update(params)
    params = params_def.copy()
    
    # initialize step size
    alpha_t = params['alpha']
    
    if not params['reduce_variance']:
        warnings.warn("Variance reduction is deactivated. This leads to suboptimal performance. Consider setting the parameter 'reduce_variance' to True.")
    
    
    #########################################################
    ## Sample style
    #########################################################
    if params['sample_style'] == 'increasing':     
        batch_size = batch_size_constructor(np.arange(params['max_iter']), a = params['batch_size']/4, \
                                            b = params['batch_size'], M = params['max_iter']-1)
    elif params['sample_style'] == 'fast_increasing': 
        batch_size = batch_size_constructor(np.arange(params['max_iter']), a = params['batch_size']/4, \
                                            b = params['batch_size'], M = params['max_iter']-1, cutoff = 10)
    else:
        batch_size = params['batch_size'] * np.ones(params['max_iter'], dtype = 'int64')
    
    #########################################################
    ## Initialization
    #########################################################
    if xi is None:
        xi = get_xi_start_point(f)
        
    # for easy problems, xi is an array (and not a dict), because we can index it faster
    assert xi.shape== (f.N,)
    
    X_hist = list(); xi_hist = list()
    step_sizes = list()
    obj = list()
    ssn_info = list(); S_hist = list()
    runtime = list(); num_eval = list()
    
    # variance reduction
    if params['reduce_variance']:
        xi_tilde = None; full_g = None
        vr_min_iter = 0
    else:
        xi_tilde = None; full_g = None; this_iter_vr = False
    
            
    hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
    if verbose:
        print(hdr_fmt % ("iter", "obj (x_t)", "f(x_t)", "phi(x_t)", "alpha_t", "batch size", "eta"))
    
    #########################################################
    ## Main loop
    #########################################################
    for iter_t in np.arange(params['max_iter']):
        
        start = time.time()
            
        if eta <= tol:
            status = 'optimal'
            break
                
        X_old = X_t.copy()
        
        # sample and update
        S = sampler(f.N, batch_size[iter_t], replace = True)
        
        # variance reduction boolean
        reduce_variance = params['reduce_variance'] and (iter_t > vr_min_iter)
        
        #########################################################
        ## Solve subproblem
        #########################################################
        X_t, xi, this_ssn = solve_subproblem(f, phi, X_t, xi, alpha_t, A, S, \
                                             newton_params = params['newton_params'],\
                                             reduce_variance = reduce_variance, xi_tilde = xi_tilde, full_g = full_g,\
                                             verbose = verbose)
                                             
        #########################################################
        ## Variance reduction
        #########################################################
        # if params['reduce_variance']:
        #     this_iter_vr = iter_t % params['m_iter'] == 0 and iter_t >= vr_min_iter
        #     if this_iter_vr:
                
        #         xi_tilde = compute_full_xi(f, X_t, is_easy)
        #         full_g = (1/f.N) * (A.T @ xi_tilde)
                
        #         # update xi
        #         if f.convex:
        #             xi = xi_tilde.copy()
        #         else:
        #             gammas = f.weak_conv(np.arange(f.N))
        #             xi = xi_tilde + gammas*(A@X_t)
        #########################################################               
       
                    
        #stop criterion
        eta = stop_scikit_saga(X_t, X_old)
        
        ssn_info.append(this_ssn)
        if store_hist:
            X_hist.append(X_t)
            
        # we only measure runtime of the iteration, excluding computation of the objective
        end = time.time()
        runtime.append(end-start)
        num_eval.append(this_ssn['evaluations'].sum() + int(this_iter_vr) * f.N)

        if measure:
            f_t = f.eval(X_t.astype('float64')) 
            phi_t = phi.eval(X_t)
            obj.append(f_t+phi_t)
        
        step_sizes.append(alpha_t)
        S_hist.append(S)
        xi_hist.append(xi.copy())
        
          
        if verbose and measure:
            print(out_fmt % (iter_t, obj[-1], f_t, phi_t, alpha_t, len(S), eta))
        
        #########################################################
        ## Step size adaption
        #########################################################
        # if reduce_variance, use constant step size, else use decreasing step size
        # set new alpha_t, +1 for next iter and +1 as indexing starts at 0
        if f.convex and not params['reduce_variance']:
             alpha_t = params['alpha']/(iter_t + 2)**(0.51)
        
        
    if eta > tol:
        status = 'max iterations reached'    
    
    if verbose:   
        print(f"Stochastic ProxPoint terminated after {iter_t} iterations with accuracy {eta} and status {status}.")
    
    info = {'objective': np.array(obj), 'iterates': np.vstack(X_hist), \
            'xi_hist': xi_hist,\
            'step_sizes': np.array(step_sizes), 'samples' : S_hist, \
            'ssn_info': ssn_info, 'runtime': np.array(runtime),\
            'evaluations': np.array(num_eval)
            }
        
    
    return X_t, info

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
    check_newton_params(newton_params)
      
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
    num_eval = list()
    
    # compute var. reduction term
    if reduce_variance:
        hat_d = (alpha/sample_size) * (subA.T @ xi_tilde[S]) - alpha * full_g    
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
        # 2 from Hstar, gstar, rest from fstar during Armijo
        num_eval.append((2+ np.log(beta)/np.log(newton_params['rho'])) * sample_size )
        
    # step 4: update xi
        xi_sub += beta * d         
        sub_iter += 1
        
    if not converged and verbose:
        print(f"WARNING: reached max. iter in semismooth Newton with residual {residual[-1]}")
    
    
    # update xi variable
    xi[S] = xi_sub.copy()
    new_X = phi.prox(Z, alpha)
    
    info = {'residual': np.array(residual), 'direction' : norm_dir, 'step_size': step_sz, \
            'objective': np.array(obj), 'evaluations': np.array(num_eval)}
    
    return new_X, xi, info