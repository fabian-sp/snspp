import numpy as np

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
        #x_mean = x_hist.mean(axis = 0)
        
    return x_mean






def block_diag(arrs):
    """Create a block diagonal matrix from the provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Parameters
    ----------
    A, B, C, ... : array-like, up to 2D
        Input arrays.  A 1D array or array-like sequence with length n is
        treated as a 2D array with shape (1,n).

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.

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


