import numpy as np

def compute_gradient_table(f, x):
    """
    computes a table of gradients at point x
    returns: array of shape 
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


