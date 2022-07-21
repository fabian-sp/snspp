"""
author: Fabian Schaipp
"""

import numpy as np
from ..helper.utils import block_diag, stop_scikit_saga
from ..helper.utils import compute_full_xi, derive_L
from .spp_easy import solve_subproblem_easy

from scipy.sparse.linalg import cg
from scipy.sparse.csr import csr_matrix
import time
import warnings

#%% functions for creating the samples S_k

def sampler(N, size, replace = False):
    """
    samples a subset of {1,..,N} with/without replacement
    """
    if not replace:
        S = np.random.choice(a = np.arange(N).astype('int'), p = (1/N)*np.ones(N), \
                         size = int(size), replace=False)
    else:
        S = np.random.randint(low = 0, high = N, size = size)
    
    # sort S in order to avoid problems with indexing later on
    S = np.sort(S)
    
    return S

def batch_size_constructor(t, a, b, M, cutoff = 18):
    """
    a: batch size at t=0
    b: batch size at t=M
    """
    if M > cutoff:
        M1 = cutoff
        c1 = np.log(b/a)/M1
    else:
        c1 = np.log(b/a)/M
        
    c2 = np.log(a)   
    y = np.exp(c1* np.minimum(t, cutoff) +c2).astype(int)
    
    return np.maximum(y, 1).astype(int)

def cyclic_batch(N, batch_size, t):
    """
    returns array of samples for cyclic sampling at iteration t, with vector of target batch sizes batch_size
    """
    C = batch_size.cumsum() % N
    if t > 0:
        a = C[t-1]
        b = C[t]
        if a <= b:
            S = np.arange(a,b)
        else:
            S = np.hstack((np.arange(0,b),  np.arange(a, N)))
    else:
        S = np.arange(0, C[t])
    
    np.sort(S)
    return S

#%%

def snspp_theoretical_step_size(f, A, b, m, eta = 0.5):
    """
    see paper for details
    
    should not be used when A is large!
    """  
    normA =  np.linalg.norm(A, axis = 1)**2
    
    if not f.convex:
        M = f.weak_conv(np.arange(f.N)).max() * normA.max()
    else:
        M = 0
    
    L_i = derive_L(f)
    L = L_i * normA.mean()
    tildeL = L_i * normA.max()
    
    term1 = 2*L+M
    term2 = (1+m/np.sqrt(2*b))*tildeL + max(M, L)
    
    a = (1/eta) * max(term1,term2)
    return 1/a


def get_xi_start_point(f, is_easy):
    if is_easy:        
        if f.name == 'logistic':
            xi =  -.5 * np.ones(f.N)
        elif f.name == 'tstudent':
            xi = np.ones(f.N)
        else: 
            xi = np.zeros(f.N)
    else:
        xi = dict(zip(np.arange(f.N), [np.zeros(f.m[i]) for i in np.arange(f.N)]))
    return xi
    
#%% functions for parameter handling

def get_default_spp_params(f, A):
    b = max(int(f.N*0.005),1)
    m = 10
    a = 1. # snspp_theoretical_step_size(f, A, b, m, 0.5)
    
    p = {'alpha': a, 'max_iter': 100, 'batch_size': b, 'sample_style': 'constant', 'reduce_variance': False,\
        'm_iter': m, 'tol_sub': 1e-1, 'newton_params': get_default_newton_params(),\
        'vr_skip': 0, 
        'measure_freq': 1}
    
    return p

def get_default_newton_params():
    
    params = {'tau': .9, 'eta' : 1e-5, 'rho': .5, 'mu': .4, \
              'cg_max_iter': 12, 'max_iter': 20}
    
    return params

def check_newton_params(newton_params):
    
    assert newton_params['mu'] > 0 and newton_params['mu'] < .5
    assert newton_params['eta'] > 0 and newton_params['eta'] < 1
    assert newton_params['tau'] > 0 and newton_params['tau'] <= 1
    assert newton_params['rho'] > 0 and newton_params['rho'] < 1
    
    return


#%% main functions

def stochastic_prox_point(f, phi, A, x0, xi = None, tol = 1e-3, params = dict(), verbose = False, measure = False, store_hist = False):
    """
    This implements the semismooth Newton stochastic proximal point method (SNSPP) for solving 
    
    .. math::
        \min{x} f(x) + \phi(x)
        
    where 
    
    .. math::
        f(x) = \frac{1}{N} \sum_{i=1}^{N} f_i(A_i x)
    
    For the case that each :math:`f_i` maps to :math:`\mathbb{R}`, the implementation becomes much easier and efficient.
    In this case, the dual variables :math:`\\xi` wil be an array of length :math:`N`.
    In the more general case, :math:`\\xi` is a dictionary with keys ``1,..,N``.
    
    SNSPP is executed with constant step sizes if variance reduction is allowed, i.e. ``params['reduce_variance']==True``.
    If variance reduction is disabled, the step sizes are chosen as 
    
    .. math::
        \alpha_k = \frac{\alpha}{0.51^k}
    
    where ::math::`\alpha` can be specified in ``params``.
    
    Parameters
    ----------
    f : loss function object
        This object describes the function :math:`f(x)`. 
        See ``snspp/helper/loss1.py`` for an example.
    phi : regularization function object
        This object describes the function :math:`\phi(x)`.
        See ``snspp/helper/regz.py`` for an example.
    A : matrix for f
    x0 : array of shape (n,)
        Starting point. If no better guess is available, use zero.
    xi : array or dict, optional
        Starting values for the dual variables :math:`\\xi`. The default is None.
    tol : float, optional
        Tolerance for the stopping criterion (sup-norm of relative change in the coefficients). The default is 1e-3.
    params : dict, optional
        Dictionary with all parameters. Possible keys:
            
            * ``max_iter``: int, maximal number of iteartions.
            * ``batch_size``: int, batch size.
            * ``alpha``: float, step size of the algorithm.
            * ``reduce_variance``: boolean. Whether to use VR or not (default = True).
            * ``m_iter``: int, number of iteration for inner loop (if VR is used). The default is 10.
            * ``tol_sub``: float, tolerance for subproblems. The default is 1e-3.
            * ``newton_params``: parameters for the semismooth Newton method for the subproblem. See ``get_default_newton_params`` for the default values.
            * ``sample_style``: str, can be 'constant' or 'fast_increasing'. THe latter will increase the batch size over the first iterations which can be more efficient.
            
    verbose : boolean, optional
        Verbosity. The default is False.
    measure : boolean, optional
        Whether to evaluate the objective after each itearion. The default is False.
        For the experiments, needs to be set to ``True``, for actual computation it is recommended to set this to ``False``.
    store_hist : boolean, optional
        Whether to store iterate history. The default is False.


    Returns
    -------
    x_t : array of shape (n,)
        Final iterate.
    x_mean : None
        Potentially, we could return the mean of the iterates. This is currently disabled.
    info : dict
        Information on objective, runtime and subproblem convergence.
    
    """    
    
    n = len(x0)
    assert n == A.shape[1], f"Starting point has wrong dimension {n} while matrices A_i have dimension {A.shape[1]}."
    
    # boolean to check whether we are in a simple setting --> faster computations (see file spp_easy.py)
    is_easy = (f.m.max() == 1) and callable(getattr(f, "fstar_vec", None))
    if verbose:
        print("Have easy version of the subproblem?", is_easy)
    
    # check whether A is in sparse format
    if isinstance(A, csr_matrix):
        sparse_format = True
    else:
        sparse_format = False
        
    
    x_t = x0.copy()
    
    status = 'not optimal'
    eta = np.inf
    
    #########################################################
    ## Set parameters
    #########################################################
    params_def = get_default_spp_params(f, A)
    params.update({k:v for k,v in params_def.items() if k not in params.keys()})
    
    # initialize step size
    alpha_t = params['alpha']
    
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
        xi = get_xi_start_point(f, is_easy)
    
    # for easy problems, xi is an array (and not a dict), because we can index it faster
    if is_easy:
        assert xi.shape== (f.N,)
    
    x_hist = list(); xi_hist = list()
    step_sizes = list()
    obj = list()
    fnat = list(); _fnat = 1.
    ssn_info = list();
    runtime = list(); num_eval = list()
    sub_runtime = list()
    
    # variance reduction
    if params['reduce_variance']:
        xi_tilde = None; full_g = None
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
            
        if eta <= tol:
            status = 'optimal'
            break
                
        x_old = x_t.copy()
        
        start = time.time()
        # sample and update
        S = sampler(f.N, batch_size[iter_t], replace = True)
        #S = cyclic_batch(f.N, batch_size, iter_t)
        
        # variance reduction boolean
        reduce_variance = params['reduce_variance'] and (iter_t >= params['vr_skip'])
        
        #########################################################
        ## Variance reduction
        #########################################################
        if params['reduce_variance']:
            this_iter_vr = iter_t % params['m_iter'] == params['vr_skip']
            
            # recompute full gradient
            if this_iter_vr:
                z_t = A@x_t
                xi_tilde = compute_full_xi(f, z_t, is_easy)
                full_g = (1/f.N) * (A.T @ xi_tilde)
                
                _fnat = 1. #np.linalg.norm(x_t - phi.prox(x_t-full_g, 1.))
                
                # update xi
                if f.convex:
                    xi = xi_tilde.copy()
                else:
                    if is_easy:
                        gammas = f.weak_conv(np.arange(f.N))
                        xi = xi_tilde + gammas*z_t
                    else:
                        raise KeyError("Variance reduction for nonconvex problems is only available if all m_i=1.")
        
        #########################################################             
        #########################################################
        ## Solve subproblem
        #########################################################
        # if params['reduce_variance']:
        #     _tol = min(params['tol_sub'], 1e-3)
        # else:
        #     _tol = max(min(params['tol_sub']*_fnat, 1e-3), 1e-6)
        
        _tol = 1e-3            
            
        sub_start = time.time()
        if not is_easy:
            x_t, xi, this_ssn = solve_subproblem(f, phi, x_t, xi, alpha_t, A, f.m, S, \
                                             tol = _tol, newton_params = params['newton_params'],\
                                             reduce_variance = reduce_variance, xi_tilde = xi_tilde,\
                                             verbose = verbose)
        else:
            x_t, xi, this_ssn = solve_subproblem_easy(f, phi, x_t, xi, alpha_t, A, S, \
                                             tol = _tol, newton_params = params['newton_params'],\
                                             reduce_variance = reduce_variance, xi_tilde = xi_tilde, full_g = full_g,\
                                             verbose = verbose)
        
        sub_end = time.time()                                     
        # we only measure runtime of the iteration, excluding computation of the objective
        end = time.time()
        
        #stop criterion
        eta = stop_scikit_saga(x_t, x_old)
        ssn_info.append(this_ssn)
            
        runtime.append(end-start)
        sub_runtime.append(sub_end-sub_start)
        num_eval.append(this_ssn['evaluations'].sum() + int(this_iter_vr) * f.N)

        if measure:
            # recompute objective every <measure_freq> iter
            if iter_t % params['measure_freq'] == 0:  
                f_t = f.eval(A@x_t) 
                phi_t = phi.eval(x_t)
            
            obj.append(f_t+phi_t)
            fnat.append(_fnat)
        
        step_sizes.append(alpha_t)
        
        if store_hist:
            xi_hist.append(xi.copy())
            x_hist.append(x_t)
        
          
        if verbose and measure:
            print(out_fmt % (iter_t, obj[-1], f_t, phi_t, alpha_t, len(S), eta))
        
        #########################################################
        ## Step size adaption
        #########################################################
        # if reduce_variance, use constant step size, else use decreasing step size
        # set new alpha_t, +1 for next iter and +1 as indexing starts at 0
        if f.convex and not params['reduce_variance']:
             alpha_t = params['alpha']/(iter_t + 2)**(0.51)
        
    assert np.all(np.array(runtime)>=0), "Measure negative runtime"
    assert np.all(np.array(runtime)-np.array(sub_runtime)>=0), "Measure negative runtime"
        
    if eta > tol:
        status = 'max iterations reached'    
    
    if verbose:   
        print(f"Stochastic ProxPoint terminated after {iter_t} iterations with accuracy {eta}")
        print(f"Stochastic ProxPoint status: {status}")
    
        
    info = {'objective': np.array(obj),
            'fnat': np.array(fnat),
            'step_sizes': np.array(step_sizes),
            'ssn_info': ssn_info, 
            'runtime': np.array(runtime),
            'sub_runtime': np.array(sub_runtime),
            'evaluations': np.array(num_eval)/f.N
            }
    
    if store_hist:
        info['iterates'] = np.vstack(x_hist)
        info['xi_hist'] = xi_hist
         
    return x_t, info



#%% 
# SOLVE SUBPROBLEM
###################### 
#
# The below functions are only used in the most general case. For most real-world applications, we have that each f_i maps to R and thus we can simplify compuatations.
# For the simpler case, the analogous functions are in ./spp_easy.py.
#
######################

def Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA, hat_d):
    """
    This functions evaluates the objective of the subproblem :math:`\mathcal{U}` at the point ``xi_sub``.
    """
    sample_size = len(S)
    
    z = x - (alpha/sample_size) * (subA.T @ xi_stack) + hat_d
    term2 = .5 * np.linalg.norm(z)**2 - phi.moreau(z, alpha)
    
    if f.m.max() == 1:
        term1 = sum([f.fstar(xi_stack[[l]], S[l]) for l in range(sample_size)])
    else:
        term1 = sum([f.fstar(xi_stack[sub_dims == l], S[l]) for l in range(sample_size)])
    
    res = term1 + (sample_size/alpha) * term2
    
    return res.squeeze()


def solve_subproblem(f, phi, x, xi, alpha, A, m, S, tol = 1e-3, newton_params = None, reduce_variance = False, xi_tilde = None, verbose = True):
    """
    see ``solve_subproblem_easy()`` in ``./spp_easy.py`` for a documentation.
    """
    if newton_params is None:
        newton_params = get_default_newton_params()

    
    check_newton_params(newton_params)
    assert alpha > 0 , "step sizes are not positive"
      
    N = len(m)
    
    # creates a vector with nrows like A in order to index the relevant A_i from A
    dims = np.repeat(np.arange(N),m)
    
    sample_size = len(S)
    assert np.all(S == np.sort(S)), "S is not sorted!"
    # dimension of the problem induced by S
    M = m[S].sum()
    
    # IMPORTANT: subA is ordered, i.e. it is in the order as np.arange(N) and NOT of S --> breaks if S not sorted 
    # if m_i = 1 for all i, we cann speed things up
    if m.max() == 1:
        subA = A[S,:]
    else:
        subA = np.vstack([A[dims == i,:] for i in S])
    
    assert subA.shape[0] == M
    assert np.all(list(xi.keys()) == np.arange(N)), "xi has wrong keys"
    
    # sub_dims is helper array to index xi_stack wrt to the elements of S
    sub_dims = np.repeat(range(sample_size), m[S])
    xi_stack = np.hstack([xi[i] for i in S])
    
    assert np.all([np.all(xi[S[l]] == xi_stack[sub_dims == l]) for l in range(sample_size)]), "Something went wrong in the sorting/stacking of xi"
    assert len(xi_stack) == M
    
    sub_iter = 0
    converged = False
    
    residual = list()
    norm_dir = list()
    step_sz = list()
    obj = list()
    num_eval = list()
    
    # compute var. reduction term
    if reduce_variance:
        xi_stack_old = np.hstack([xi_tilde[i] for i in S])
        xi_full_old = np.hstack([xi_tilde[i] for i in range(f.N)])
        hat_d =  (alpha/sample_size) * (subA.T @ xi_stack_old) - (alpha/f.N) * (A.T @ xi_full_old)      
    else:
        hat_d = 0.
    
    #compute term coming from weak convexity
    if not f.convex:
        gamma_i = f.weak_conv(S)
        gamma_i = np.repeat(gamma_i, m[S])
        hat_d += (alpha/sample_size) * (gamma_i.reshape(1,-1) * subA.T @ (subA @ x))
    
    while sub_iter < newton_params['max_iter']:
        
    # step 1: construct Newton matrix and RHS
        
        z = x - (alpha/sample_size) * (subA.T @ xi_stack)  + hat_d
        rhs = -1. * (np.hstack([f.gstar(xi[i], i) for i in S]) - subA @ phi.prox(z, alpha))
        
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
        
        if m.max() == 1:
            tmp_d = np.hstack([f.Hstar(xi[i], i) for i in S])
            tmp = np.diag(tmp_d + eps_reg)           
        else:
            tmp = block_diag([f.Hstar(xi[i], i) for i in S])
            tmp += eps_reg * np.eye(tmp.shape[0])

        W = tmp + tmp2
        assert not np.isnan(W).any(), "The Newton matrix contains NA entries, check f.Hstar."
    # step2: solve Newton system
            
        cg_tol = min(newton_params['eta'], np.linalg.norm(rhs)**(1+ newton_params['tau']))
        
        if m.max() == 1:
            precond = np.diag(1/tmp_d)
        else:
            precond = None
        
        d, cg_status = cg(W, rhs, tol = cg_tol, maxiter = newton_params['cg_max_iter'], M = precond)
        
        if not d@rhs > -1e-8:
            warnings.warn(f"No descent direction, {d@rhs}")
        #assert cg_status == 0, f"CG method did not converge, exited with status {cg_status}"
        norm_dir.append(np.linalg.norm(d))
        
    # step 3: backtracking line search
        U_old = Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA, hat_d)
        beta = 1.
        U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA, hat_d)
           
        while U_new > U_old + newton_params['mu'] * beta * (d @ -rhs):
            beta *= newton_params['rho']
            U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA, hat_d)
            
        step_sz.append(beta)
        obj.append(U_new)
        # 2 from Hstar, gstar, rest from fstar during Armijo
        num_eval.append((2+ np.log(beta)/np.log(newton_params['rho'])) * sample_size )
       
        
    # step 4: update xi
        xi_stack += beta * d
        
        if m.max() == 1:
            # double bracket/ reshape because xi have to be arrays (not scalars!)
            xi.update(dict(zip(S, xi_stack.reshape(-1,1))))
        else:
            for l in range(sample_size):
                xi[S[l]] = xi_stack[sub_dims == l].copy()
                
        sub_iter += 1
        
    if not converged and verbose:
        warnings.warn(f"Reached max. iter in semismooth Newton with residual {residual[-1]}")
    
    # update primal iterate
    z = x - (alpha/sample_size) * (subA.T @ xi_stack) + hat_d
    new_x = phi.prox(z, alpha)
    
    info = {'residual': np.array(residual), 'direction' : norm_dir, 'step_size': step_sz, \
            'objective': np.array(obj), 'evaluations': np.array(num_eval)}
    
    
    return new_x, xi, info
