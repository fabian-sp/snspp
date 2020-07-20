import numpy as np
from numba import njit

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

def stop_mean_objective(obj, cutoff = True):
    """
    obj: list of the objective function values of the "mean iterate"
    """
    if cutoff:
        if len(obj) >= 6:
            objs = obj[3:]
        else:
            objs = obj.copy()
    else:
        objs = obj.copy()
        
    if len(objs) <= 3:
        return np.inf
    else:
        return abs(objs[-1] - np.mean(objs))
    
def stop_optimal(x, f, phi):
    
    gradf = compute_full_gradient(f,x) 
    return np.linalg.norm(x - phi.prox( x- gradf, 1.))


############################################################################################
### Useful functions for algorithms
############################################################################################

def compute_full_xi(f, x):
    
    dims = np.repeat(np.arange(f.N),f.m)
    vals = list()
    for i in np.arange(f.N):
        A_i =  f.A[dims == i].copy()
        vals.append(f.g(A_i @ x, i))
        
    xi  = dict(zip(np.arange(f.N), vals))
    
    return xi 

def compute_full_gradient(f,x):
    """
    computes the full gradient 1/N * sum (A_i.T @ grad f_i(A_ix))
    NOTE: not storage optimized (needs O(N*n) storage)
    """
    grads = compute_gradient_table(f, x)
    return (1/f.N)*grads.sum(axis = 0)


def compute_gradient_table(f, x):
    """
    computes a table of gradients at point x
    returns: array of shape Nxn
    """
    
    dims = np.repeat(np.arange(f.N),f.m)

    # initialize object for storing all gradients
    gradients = list()
    for i in np.arange(f.N):
        A_i =  f.A[dims == i].copy()
        tmp_i = A_i.T @ f.g( A_i @ x, i)
        gradients.append(tmp_i)
        
    gradients = np.vstack(gradients)
    
    return gradients


def compute_x_mean(x_hist, step_sizes = None):
    """

    Parameters
    ----------
    x_hist : list
        contains all iterates 
    step_sizes : list, optional
        contains all step sizes
        if None, then no weighting

    Returns
    -------
    x_mean : array of length n
        mean iterate

    """
    if step_sizes is not None:
        a = np.array(step_sizes)
        assert np.all(a > 0)
        assert len(step_sizes) == len(x_hist)
    else:
        a = np.ones(len(x_hist))
        
    X = np.vstack(x_hist)
    
    if len(X.shape) == 1:
        x_mean = x_hist.copy()
    else:
        x_mean = (1/a.sum()) * X.T @ a 
        #x_mean = X.mean(axis = 0)
        
    return x_mean

# FOR TESTING
#x_hist = [np.random.rand(20) for i in range(10)]
#step_sizes = [.3]*10


############################################################################################
### Linear ALgebra stuff
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


