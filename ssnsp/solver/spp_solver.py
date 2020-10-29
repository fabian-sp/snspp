"""
author: Fabian Schaipp
"""

import numpy as np
from ..helper.utils import block_diag, compute_x_mean, stop_scikit_saga
from ..helper.utils import compute_full_xi, compute_x_mean_hist
from .spp_easy import solve_subproblem_easy
from scipy.sparse.linalg import cg
import time

#%% functions for creating the samples S_k
def sampler(N, size, replace = False):
    """
    samples a subset of {1,..,N} with/without replacement
    """
    assert size <= N, "specified a bigger sample size than N"
    S = np.random.choice(a = np.arange(N).astype('int'), p = (1/N) * np.ones(N), \
                         size = int(size), replace = replace)
    
    S = S.astype('int')
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
    
    return y

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

def determine_alpha(f, batch_size, m_iter):
    
    v1 = 1
    v2 = 1
    v3 = 1
    v4 = 1
    v5 = 1e-4
    theta = 1
    
    gbar = f.weak_conv(np.arange(f.N)).max()
    normA = np.apply_along_axis(np.linalg.norm, axis = 1, arr = f.A)
    
    if f.name == 'squared':
        L_i = 2.  
    elif f.name == 'logistic':
        L_i = .25 
    elif f.name == 'tstudent':
        L_i =  (2/f.v)
    else:
        raise ValueError("Unknown function, cannot determine Lipschitz constant!")
    
    L = (1/f.N) * np.sum(normA**2 * L_i)
    Lb = (normA**2).max() * L_i
    Cb = gbar*(normA**2).max()
    
    s1 = 2*(theta+L) + v3*Cb
    s2 = v5*Lb**2*m_iter*(m_iter+1)/(2*batch_size) + v4*Lb + L
    s3 = Lb/v4 + 1/v5 + Cb/v3 + 1/v1 + 1/v2
    
    print(s1,s2,s3)
    
    return 1/max(s1,s2,s3)
    
    
#%% functions for parameter handling

def get_default_newton_params():
    
    params = {'tau': .9, 'eta' : 1e-5, 'rho': .5, 'mu': .4, 'eps': 1e-3, \
              'cg_max_iter': 12, 'max_iter': 15}
    
    return params

def check_newton_params(newton_params):
    
    assert newton_params['mu'] > 0 and newton_params['mu'] < .5
    assert newton_params['eta'] > 0 and newton_params['eta'] < 1
    assert newton_params['tau'] > 0 and newton_params['tau'] <= 1
    assert newton_params['rho'] > 0 and newton_params['rho'] < 1
    
    assert newton_params['eps'] >= 0
    
    return


#%% main functions

def Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA, hat_d):
    
    sample_size = len(S)
    
    z = x - (alpha/sample_size) * (subA.T @ xi_stack) + hat_d
    term2 = .5 * np.linalg.norm(z)**2 - phi.moreau(z, alpha)
    
    if f.m.max() == 1:
        term1 = sum([f.fstar(xi_stack[[l]], S[l]) for l in range(sample_size)])
    else:
        term1 = sum([f.fstar(xi_stack[sub_dims == l], S[l]) for l in range(sample_size)])
    
    res = term1 + (sample_size/alpha) * term2
    
    return res.squeeze()
    
def solve_subproblem(f, phi, x, xi, alpha, A, m, S, newton_params = None, reduce_variance = False, xi_tilde = None, verbose = False):
    """
    m: vector with all dimensions m_i, i = 1,..,N
    
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
    
    #assert np.all([np.all(xi[S[l]] == xi_stack[sub_dims == l]) for l in range(sample_size)]), "Something went wrong in the sorting/stacking of xi"
    assert len(xi_stack) == M
    
    sub_iter = 0
    converged = False
    
    residual = list()
    norm_dir = list()
    step_sz = list()
    
    # compute var. reduction term
    if reduce_variance:
        xi_stack_old = np.hstack([xi_tilde[i] for i in S])
        xi_full_old = np.hstack([xi_tilde[i] for i in range(f.N)])
        hat_d =  (alpha/sample_size) * (subA.T @ xi_stack_old) - (alpha/f.N) * (f.A.T @ xi_full_old)      
    else:
        hat_d = 0.
    
    #compute term coming from weak convexity
    if not f.convex:
        gamma_i = f.weak_conv(S)
        gamma_i = np.repeat(gamma_i, m[S])
        hat_d += (alpha/sample_size) * (gamma_i.reshape(1,-1) * subA.T @ (subA @ x))
    
    while sub_iter < newton_params['max_iter']:
        
    # step 1: construct Newton matrix and RHS
        if verbose:
            print("Construct")
        
        z = x - (alpha/sample_size) * (subA.T @ xi_stack)  + hat_d
        rhs = -1. * (np.hstack([f.gstar(xi[i], i) for i in S]) - subA @ phi.prox(z, alpha))
        
        residual.append(np.linalg.norm(rhs))
        if np.linalg.norm(rhs) <= newton_params['eps']:
            converged = True
            break
        
        if verbose:
            print("Construct2")
        
        U = phi.jacobian_prox(z, alpha)
        
        if phi.name == '1norm':
            # U is 1d array with only 1 or 0 --> speedup by not constructing 2d diagonal array
            bool_d = U.astype(bool)
            
            subA_d = subA[:, bool_d].astype('float32')
            tmp2 = (alpha/sample_size) * subA_d @ subA_d.T
        else:
            tmp2 = (alpha/sample_size) * subA @ U @ subA.T
        
        
        if verbose:
            print("Construct3")
            
        eps_reg = 1e-4
        
        if m.max() == 1:
            tmp_d = np.hstack([f.Hstar(xi[i], i) for i in S])
            tmp = np.diag(tmp_d + eps_reg)           
        else:
            tmp = block_diag([f.Hstar(xi[i], i) for i in S])
            tmp += eps_reg * np.eye(tmp.shape[0])

        W = tmp + tmp2
        assert not np.isnan(W).any(), "Something went wrong during construction of the Hessian"
    # step2: solve Newton system
        if verbose:
            print("Start CG method")
            
        #start = time.time()
        cg_tol = min(newton_params['eta'], np.linalg.norm(rhs)**(1+ newton_params['tau']))
        
        if m.max() == 1:
            precond = np.diag(1/tmp_d)
            #precond = None
        else:
            precond = None
        
        d, cg_status = cg(W, rhs, tol = cg_tol, maxiter = newton_params['cg_max_iter'], M = precond)
        #end = time.time(); print("CG", end-start)
        
        assert d@rhs > -1e-8 , f"No descent direction, {d@rhs}"
        #assert cg_status == 0, f"CG method did not converge, exited with status {cg_status}"
        norm_dir.append(np.linalg.norm(d))
        
    # step 3: backtracking line search
        if verbose:
            print("Start Line search")
        U_old = Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA, hat_d)
        beta = 1.
        U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA, hat_d)
           
        while U_new > U_old + newton_params['mu'] * beta * (d @ -rhs):
            beta *= newton_params['rho']
            U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA, hat_d)
            
        step_sz.append(beta)
        
    # step 4: update xi
        if verbose:
            print("Update xi variables")
        xi_stack += beta * d
        
        if m.max() == 1:
            # double bracket/ reshape because xi have to be arrays (not scalars!)
            xi.update(dict(zip(S, xi_stack.reshape(-1,1))))
        else:
            for l in range(sample_size):
                xi[S[l]] = xi_stack[sub_dims == l].copy()
                
        sub_iter += 1
        
    if not converged:
        print(f"WARNING: reached maximal iterations in semismooth Newton -- accuracy {residual[-1]}")
    
    # update primal iterate
    z = x - (alpha/sample_size) * (subA.T @ xi_stack) + hat_d
    new_x = phi.prox(z, alpha)
    
    info = {'residual': np.array(residual), 'direction' : norm_dir, 'step_size': step_sz }
    
    
    return new_x, xi, info


    
def stochastic_prox_point(f, phi, x0, xi = None, tol = 1e-3, params = dict(), verbose = False, measure = False):
    
    A = f.A.copy()
    n = len(x0)
    assert n == A.shape[1], "wrong dimensions"
    
    # boolean to check whether we are in a simple setting --> faster computations
    is_easy = (f.m.max() == 1) and callable(getattr(f, "fstar_vec", None))
    if verbose:
        print("EASY PROBLEM???", is_easy)
    
    x_t = x0.copy()
    x_mean = x_t.copy()
    
    status = 'not optimal'
    eta = np.inf
    
    #########################################################
    ## Set parameters
    #########################################################
    if 'alpha_C' not in params.keys():
        C = 1.
    else:
        C = params['alpha_C']       
    alpha_t = C
    
    if 'max_iter' not in params.keys():    
        params['max_iter'] = 100
    else:
        assert type(params['max_iter']) == int, "Max. iter needs to be integer"
        
    if 'sample_size' not in params.keys():    
        params['sample_size'] = max(int(f.N/4), 1)
    
    if 'sample_style' not in params.keys():    
        params['sample_style'] = 'constant'
    
    if 'reduce_variance' not in params.keys():    
        params['reduce_variance'] = False
        
    if 'newton_params' not in params.keys():
        params['newton_params'] = get_default_newton_params()
    
    #########################################################
    ## Sample style
    #########################################################
    if params['sample_style'] == 'increasing':     
        batch_size = batch_size_constructor(np.arange(params['max_iter']), a = params['sample_size']/4, \
                                            b = params['sample_size'], M = params['max_iter']-1)
    elif params['sample_style'] == 'fast_increasing': 
        batch_size = batch_size_constructor(np.arange(params['max_iter']), a = params['sample_size']/4, \
                                            b = params['sample_size'], M = params['max_iter']-1, cutoff = 10)
    else:
        batch_size = params['sample_size'] * np.ones(params['max_iter'], dtype = 'int64')
    
    #########################################################
    ## Initialization
    #########################################################
    if xi is None:
        if f.name == 'logistic':
            xi = dict(zip(np.arange(f.N), [ -.5 * np.ones(f.m[i]) for i in np.arange(f.N)]))
        elif f.name == 'tstudent':
            xi = dict(zip(np.arange(f.N), [ 10*np.ones(f.m[i]) for i in np.arange(f.N)]))
        else:
            xi = dict(zip(np.arange(f.N), [np.zeros(f.m[i]) for i in np.arange(f.N)]))
    
    # for easy problems, xi is an array (and not a dict), because we can index it faster
    if is_easy:
        xi = np.hstack(list(xi.values()))
        assert xi.shape== (f.N,)
    
    x_hist = list(); xi_hist = list()
    step_sizes = list()
    obj = list(); obj2 = list()
    ssn_info = list(); S_hist = list()
    runtime = list()
    
    # variance reduction
    if params['reduce_variance']:
        #counter = batch_size.cumsum() % f.N
        #xi_tilde_update = (np.diff(counter, prepend = f.N) < 0)
        xi_tilde = None
        vr_min_iter = 0
        m_iter = 10
    else:
        xi_tilde = None
    
            
    hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
    if verbose:
        print(hdr_fmt % ("iter", "obj (x_t)", "obj(x_mean)", "alpha_t", "batch size", "eta"))
    
    #########################################################
    ## Main loop
    #########################################################
    for iter_t in np.arange(params['max_iter']):
        
        start = time.time()
            
        if eta <= tol:
            status = 'optimal'
            break
                
        x_old = x_t.copy()
        
        # sample and update
        S = sampler(f.N, batch_size[iter_t])
        #S = cyclic_batch(f.N, batch_size, iter_t)
        
        #params['newton_params']['eps'] =  min(1e-3, 1e-1/(iter_t+1)**(1.1))
        params['newton_params']['eps'] =  5e-3
        # variance reduction boolean
        reduce_variance = params['reduce_variance'] and (iter_t > vr_min_iter)
                
        if not is_easy:
            x_t, xi, this_ssn = solve_subproblem(f, phi, x_t, xi, alpha_t, A, f.m, S, \
                                             newton_params = params['newton_params'],\
                                             reduce_variance = reduce_variance, xi_tilde = xi_tilde,\
                                             verbose = False)
        else:
            x_t, xi, this_ssn = solve_subproblem_easy(f, phi, x_t, xi, alpha_t, A, S, \
                                             newton_params = params['newton_params'],\
                                             reduce_variance = reduce_variance, xi_tilde = xi_tilde,\
                                             verbose = False)
                                             
        # xi_tilde gets updated
        
        if params['reduce_variance']:
            if iter_t % m_iter == 0 and iter_t >= vr_min_iter:
                xi_tilde = compute_full_xi(f, x_t, is_easy)
                
                # when f convex, update xi
                if f.convex:
                    xi = xi_tilde.copy()
                else:
                    if is_easy:
                        gammas = f.weak_conv(np.arange(f.N))
                        xi = xi_tilde + gammas*(A@x_t)
                   
        #stop criterion
        eta = stop_scikit_saga(x_t, x_old)
        
        # we only measure runtime of the iteration, excluding computation of the diagnostics
        end = time.time()
        runtime.append(end-start)

        
        # save all diagnostics
        ssn_info.append(this_ssn)
        x_hist.append(x_t)
        
        if measure:
            obj.append(f.eval(x_t.astype('float64')) + phi.eval(x_t))
        
        step_sizes.append(alpha_t)
        S_hist.append(S)
        xi_hist.append(xi.copy())
        
        #calc x_mean 
        x_mean = compute_x_mean(x_hist, step_sizes = None)
        if measure:
            obj2.append(f.eval(x_mean.astype('float64')) + phi.eval(x_mean))
          
        if verbose and measure:
            print(out_fmt % (iter_t, obj[-1], obj2[-1] , alpha_t, len(S), eta))
        
        # set new alpha_t, +1 for next iter and +1 as indexing starts at 0
        if f.convex:
        #if True:
            alpha_t = C/(iter_t + 2)**(0.51)
        
    if eta > tol:
        status = 'max iterations reached'    
    
       
    print(f"Stochastic ProxPoint terminated after {iter_t} iterations with accuracy {eta}")
    print(f"Stochastic ProxPoint status: {status}")
    
    info = {'objective': np.array(obj), 'objective_mean': np.array(obj2), 'iterates': np.vstack(x_hist), \
            'mean_hist': compute_x_mean_hist(np.vstack(x_hist)), 'xi_hist': xi_hist,\
            'step_sizes': np.array(step_sizes), 'samples' : S_hist, \
            'ssn_info': ssn_info, 'runtime': np.array(runtime)}
    
    return x_t, x_mean, info