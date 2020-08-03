"""
author: Fabian Schaipp
"""

import numpy as np
from ..helper.utils import block_diag, compute_x_mean, stop_mean_objective, stop_optimal, stop_scikit_saga
from ..helper.utils import compute_gradient_table, compute_full_xi
from scipy.sparse.linalg import cg
import time

def sampler(N, size):
    """
    samples a subset of {1,..,N} without replacement
    """
    assert size <= N, "specified a bigger sample size than N"
    S = np.random.choice(a = np.arange(N).astype('int'), p = (1/N) * np.ones(N), size = int(size), replace = False)
    
    S = S.astype('int')
    # sort S in order to avoid problems with indexing later on
    S = np.sort(S)
    
    return S


def Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA):
    
    sample_size = len(S)
    
    z = x - (alpha/sample_size) * (subA.T @ xi_stack)
    term2 = .5 * np.linalg.norm(z)**2 - phi.moreau(z, alpha)
    
    if f.m.max() == 1:
        term1 = sum([f.fstar(xi_stack[[l]], S[l]) for l in range(sample_size)])
    else:
        term1 = sum([f.fstar(xi_stack[sub_dims == l], S[l]) for l in range(sample_size)])
    
    res = term1 + (sample_size/alpha) * term2
    
    return res.squeeze()

def get_default_newton_params():
    
    params = {'tau': .9, 'eta' : 1e-2, 'rho': .5, 'mu': .45, 'eps': 1e-3, \
              'cg_max_iter': 15, 'max_iter': 40}
    
    return params

def check_newton_params(newton_params):
    
    assert newton_params['mu'] > 0 and newton_params['mu'] < .5
    assert newton_params['eta'] > 0 and newton_params['eta'] < 1
    assert newton_params['tau'] > 0 and newton_params['tau'] <= 1
    assert newton_params['rho'] > 0 and newton_params['rho'] < 1
    
    assert newton_params['eps'] >= 0
    #assert newton_params['delta'] >= 0 and newton_params['delta'] < 1
    
    return

def solve_subproblem(f, phi, x, xi, alpha, A, m, S, gradient_table = None, newton_params = None, verbose = False):
    """
    m: vector with all dimensions m_i, i = 1,..,N
    
    """
    if newton_params is None:
        newton_params = get_default_newton_params()
    
    check_newton_params(newton_params)
    assert alpha > 0 , "step sizes are not positive"
      
    xi_old = xi.copy()
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
    
    while sub_iter < newton_params['max_iter']:

    # step 1: construct Newton matrix and RHS
        if verbose:
            print("Construct")
        z = x - (alpha/sample_size) * (subA.T @ xi_stack)
        rhs = -1 * (np.hstack([f.gstar(xi[i], i) for i in S]) - subA @ phi.prox(z, alpha))
        residual.append(np.linalg.norm(rhs))
        
        if np.linalg.norm(rhs) <= newton_params['eps']:
            converged = True
            break
        
        if verbose:
            print("Construct2")
        U = phi.jacobian_prox(z, alpha)
        if phi.name == '1norm':
            # U is diagonal with only 1 or 0 --> speedup
            bool_d = np.diag(U).astype(bool)
            subA_d = subA[:, bool_d].astype('float32')
            tmp2 = (alpha/sample_size) * subA_d @ subA_d.T
        else:
            tmp2 = (alpha/sample_size) * subA @ U @ subA.T
        
        if verbose:
            print("Construct3")
        if m.max() == 1:
            tmp = np.diag(np.hstack([f.Hstar(xi[i], i) for i in S]))
        else:
            tmp = block_diag([f.Hstar(xi[i], i) for i in S])
        
        eps_reg = 1e-4
        W = tmp + tmp2 + eps_reg * np.eye(tmp2.shape[0])
        
        
    # step2: solve Newton system
        if verbose:
            print("Start CG method")
        cg_tol = min(newton_params['eta'], np.linalg.norm(rhs)**(1+ newton_params['tau']))
        d, cg_status = cg(W, rhs, tol = cg_tol, maxiter = newton_params['cg_max_iter'])
        
        assert d@rhs > -1e-8 , f"No descent direction, {d@rhs}"
        #assert cg_status == 0, f"CG method did not converge, exited with status {cg_status}"
        norm_dir.append(np.linalg.norm(d))
    # step 3: backtracking line search
        if verbose:
            print("Start Line search")
        U_old = Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA)
        #print(f"U_old: {U_old} with residual {np.linalg.norm(rhs)}")
        beta = 1.
        U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA)
        
        #counter = 0
        while U_new > U_old + newton_params['mu'] * beta * (d @ -rhs):
            #print(f"U_new: {U_new} vs . { U_old + newton_params['mu'] * beta * (d @ -rhs)} with beta being {beta}")
            beta *= newton_params['rho']
            U_new = Ueval(xi_stack + beta*d, f, phi, x, alpha, S, sub_dims, subA)
            # reset if getting stuck
            #counter +=1
            # if counter >= 15:
            #     print("Semismooth Newton: reset step size and ignore Armijo")
            #     beta = .8
            #     break
        
        step_sz.append(beta)
    # step 4: update xi
        if verbose:
            print("Update xi variables")
        xi_stack += beta * d
        
        if m.max() == 1:
            # double bracket/ reshape because xi have to be arrays (not scalars!)
            # for l in range(sample_size):
            #    xi[S[l]] = xi_stack[[l]].copy()
            xi.update(dict(zip(S,xi_stack.reshape(-1,1))))
        else:
            for l in range(sample_size):
                xi[S[l]] = xi_stack[sub_dims == l].copy()
                
        sub_iter += 1
        
    if not converged:
        print(f"WARNING: reached maximal iterations in semismooth Newton -- accuracy {residual[-1]}")
    
    if gradient_table is not None:
        #xi_stack_old = np.hstack([xi_old[i] for i in S])
        #xi_full_old = np.hstack([xi_old[i] for i in range(f.N)])
        #correct =  (alpha/sample_size) * (subA.T @ xi_stack_old) - (alpha/f.N) * (f.A.T @ xi_full_old)
        
        #tmp_g = np.vstack([gradient_table[i,:] for i in S])
        #correct = (alpha/sample_size) * tmp_g.sum(axis = 0) - (alpha/f.N) * gradient_table.sum(axis = 0)
        #print(np.linalg.norm(correct))
        correct = 0.      
    else:
        correct = 0.
    
    # update primal iterate
    z = x - (alpha/sample_size) * (subA.T @ xi_stack) + correct
    new_x = phi.prox(z, alpha)
    
    info = {'residual': np.array(residual), 'direction' : norm_dir, 'step_size': step_sz }
    
    #new_z = new_x - (1/sample_size) * (subA.T @ xi_stack)
    #eta = np.linalg.norm(new_x - phi.prox(new_z, alpha = 1))
    
    return new_x, xi, info

def batch_size_constructor(t, a, b, M):
    """
    a: batch size at t=0
    b: batch size at t=M
    """
    c1 = np.log(b/a)/M
    c2 = np.log(a)
    y = np.exp(c1*t+c2).astype(int)
    
    #k = np.log(1e3)/M
    #y = b/(1+np.exp(-k*t))
    
    return y

def stochastic_prox_point(f, phi, x0, xi = None, tol = 1e-3, params = dict(), verbose = False, measure = False):
    
    # initialize all variables
    A = f.A.copy()
    n = len(x0)
    m = f.m.copy()
    assert n == A.shape[1], "wrong dimensions"
    
    x_t = x0.copy()
    x_mean = x_t.copy()
    
    # initialize for stopping criterion
    status = 'not optimal'
    eta = np.inf
    
    if 'alpha_C' not in params.keys():
        C = 1.
    else:
        C = params['alpha_C']
        
    alpha_t = C
    
    if 'max_iter' not in params.keys():    
        params['max_iter'] = 70
    
    if 'sample_size' not in params.keys():    
        params['sample_size'] = max(int(f.N/4), 1)
    
    if 'sample_style' not in params.keys():    
        params['sample_style'] = 'constant'
        
    if 'newton_params' not in params.keys():
        params['newton_params'] = get_default_newton_params()
    
    if params['sample_style'] == 'increasing':     
        batch_size = batch_size_constructor(np.arange(params['max_iter']), a = params['sample_size']/4, \
                                            b = params['sample_size'], M = params['max_iter']-1)
    else:
        batch_size = params['sample_size'] * np.ones(params['max_iter'])
    
    
    # initialize variables + containers
    if xi is None:
        if f.name == 'logistic':
            xi = dict(zip(np.arange(f.N), [ -.5 * np.ones(m[i]) for i in np.arange(f.N)]))
        else:
            xi = dict(zip(np.arange(f.N), [np.zeros(m[i]) for i in np.arange(f.N)]))
    
    x_hist = list()
    step_sizes = list()
    obj = list()
    obj2 = list()
    S_hist = list()
    xi_hist = list()
    ssn_info = list()
    runtime = list()
    
    # variance reduction
    reduce_variance = False
    G = None
    
    hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
    if verbose:
        print(hdr_fmt % ("iter", "obj (x_t)", "obj(x_mean)", "alpha_t", "batch size", "eta"))
    
    for iter_t in np.arange(params['max_iter']):
        
        start = time.time()
            
        if eta <= tol:
            status = 'optimal'
            break
                
        x_old = x_t.copy()
        
        # sample and update
        S = sampler(f.N, batch_size[iter_t])
        
        x_t, xi, this_ssn = solve_subproblem(f, phi, x_t, xi, alpha_t, A, m, S, gradient_table = G, \
                                             newton_params = params['newton_params'], verbose = False)
        
        #stop criterion
        #eta = stop_optimal(x_t, f, phi)
        eta = stop_scikit_saga(x_t, x_old)
        
        # variance reduction
        if reduce_variance:
            G = compute_gradient_table(f, x_t)
            #print("Norm of full gradient", np.linalg.norm(1/f.N * G.sum(axis=0)))
        
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
        if iter_t >= 0:
            alpha_t = C/(iter_t + 2)
        else:
            alpha_t = C
            
    if eta > tol:
        status = 'max iterations reached'    
        
    print(f"Stochastic ProxPoint terminated after {iter_t} iterations with accuracy {eta}")
    print(f"Stochastic ProxPoint status: {status}")
    
    info = {'objective': np.array(obj), 'iterates': np.vstack(x_hist), 'xi_hist': xi_hist, 'step_sizes': np.array(step_sizes), 'samples' : np.array(S_hist), \
            'objective_mean': np.array(obj2), 'ssn_info': ssn_info, 'runtime': np.array(runtime)}
    
    return x_t, x_mean, info