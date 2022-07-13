import numpy as np
from numba import njit
import warnings

# logistic loss gradient norm at zero
#np.linalg.norm(1/(2*f.N)*A.sum(axis=0))

############################################################################################
### Stopping criteria
############################################################################################

@njit()
def stop_scikit_saga(x_t, x_old):
    """
    ||x_t - x_t-1||_inf / ||x_t||_inf
    """
    nom = np.linalg.norm(x_t - x_old, np.inf)
    denom = np.linalg.norm(x_t, np.inf) +1e-8
    
    return nom/denom

    
# def stop_optimal(x, f, phi):
#     """
#     Optimality residual using second prox theorem
#     Computationally expensive if N is large!!
#     """
#     gradf = compute_full_gradient(f,x) 
#     return np.linalg.norm(x - phi.prox( x - gradf, 1.))


############################################################################################
### Useful functions for algorithms
############################################################################################

def derive_L(f):
    """
    Given a loss f_i, calculates L := max L_i where f_i is L_i smooth. Does not account for A_i here!
    """
    
    if f.name == 'squared':
        L = 2    
    elif f.name == 'logistic':
        L = .25 
    elif f.name == 'tstudent':
        L =  (2/f.v) 
    else:
        warnings.warn("For the given loss f, we could not determine the correct Lipschitz smoothness constant. The default step size is maybe too large (divergence) or too small (slow convergence).")
        L = 1.
    
    return L

def compute_full_xi(f, A, x, is_easy = False):
    """
    needed for variance reduction
    
    if is_easy: return an array of size (N,)
    if not is_easy: return dictionary where each value is array of size (m_i,)
    """
    if is_easy:
        xi = compute_xi_inner(f, A, x).squeeze() 
    else:
        dims = np.repeat(np.arange(f.N),f.m)
        vals = list()
        for i in np.arange(f.N):
            A_i =  A[dims == i].copy()
            vals.append(f.g(A_i @ x, i))
                 
        xi  = dict(zip(np.arange(f.N), vals))
    
    return xi 

@njit()
def compute_xi_inner(f, A, x):
    
    vals = np.zeros((f.N,1))
    
    for i in np.arange(f.N):
        A_i = np.ascontiguousarray(A[i,:]).reshape(1,-1)
        vals[i,:] = f.g(A_i @ x, i)
    
    return vals
            

# needed for initializing in SAGA (only used once and outside of main loop)
def compute_gradient_table(f, A, x):
    """
    computes a table of gradients at point x
    returns: array of shape Nxn
    """
    
    dims = np.repeat(np.arange(f.N),f.m)

    # initialize object for storing all gradients
    gradients = list()
    for i in np.arange(f.N):
        if f.m.max() == 1:
            A_i = A[[i],:].copy()
        else:
            A_i =  A[dims == i].copy()
        tmp_i = A_i.T @ f.g( A_i @ x, i)
        gradients.append(tmp_i)
        
    gradients = np.vstack(gradients)
    
    return np.ascontiguousarray(gradients)


# needed for ADAGRAD + SVRG
@njit()
def compute_batch_gradient(f, A, x, S):
    """
    computes a table of gradients at point x with mini batch S
    returns: array of shape n
    """   
    
    # initialize object for storing all gradients
    gradients = np.zeros_like(x)
        
    for i in S:
        # A_i needs shape (1,n)
        A_i =  np.ascontiguousarray(A[i,:]).reshape(1,-1)
        tmp_i = A_i.T @ f.g( A_i @ x, i)
        gradients += tmp_i
    
    g = (1/len(S))*gradients
    
    return g


# needed for mini-batch SAGA
@njit()
def compute_batch_gradient_table(f, A, x, S):
    """
    computes a table of gradients at point x with mini batch S
    returns: array of shape (len(S),n)
    """   
    
    # initialize object for storing all gradients
    gradients = np.zeros((len(S), len(x)))
        
    for j in range(len(S)):
        i = S[j]
        # A_i needs shape (1,n)
        A_i =  np.ascontiguousarray(A[i,:]).reshape(1,-1)
        gradients[j,:] = A_i.T @ f.g( A_i @ x, i)
    
    return gradients

# def compute_x_mean(x_hist, step_sizes = None):
#     """

#     Parameters
#     ----------
#     x_hist : list
#         contains all iterates 
#     step_sizes : list, optional
#         contains all step sizes
#         if None, then no weighting

#     Returns
#     -------
#     x_mean : array of length n
#         mean iterate

#     """
#     if step_sizes is not None:
#         a = np.array(step_sizes)
#         assert np.all(a > 0)
#         assert len(step_sizes) == len(x_hist)
#     else:
#         a = np.ones(len(x_hist))
        
#     X = np.vstack(x_hist)
    
#     if len(X.shape) == 1:
#         x_mean = x_hist.copy()
#     else:
#         x_mean = (1/a.sum()) * X.T @ a 
#         #x_mean = X.mean(axis = 0)
        
#     return x_mean


# def compute_x_mean_hist(iterates):
    
#     scaler = 1/ (np.arange(len(iterates)) + 1)   
#     res = scaler[:,np.newaxis] * iterates.cumsum(axis = 0)
    
#     return res

############################################################################################
### Linear Algebra stuff
############################################################################################

def block_diag(arrs):
    """Create a block diagonal matrix from a list of provided arrays.
    
    This is source coded copied from scipy with slight modification!
    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args) 

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


