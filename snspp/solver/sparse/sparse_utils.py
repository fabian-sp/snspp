import numpy as np
from numba import njit
import warnings

from csr import CSR
from tick.linear_model import LogisticRegression as tickLogReg

# creates a csr.CSR object from scipy.CSR which is usable inside of numba functions
def create_csr(A):
    
    nrows = A.shape[0]
    ncols = A.shape[1]
    
    nnz = len(A.data)
    rps = A.indptr
    cis= A.indices
    v = A.data
    
    A_csr = CSR(nrows, ncols, nnz, rps, cis, v)

    return A_csr

# needed for SNSPP and SVRG
@njit()
def sparse_xi_inner(f, z):
    
    vals = np.zeros(f.N,)
    
    for i in np.arange(f.N):
        vals[i] = f.g(z[i], i)
    
    return vals

# needed for initializing in SAGA (only used once and outside of main loop)
def sparse_gradient_table(f, A, x):
    """
    computes a vector of nabla f_i(x)
    returns: array of shape N
    """
    
    # initialize object for storing all gradients
    gradients = list()
    z = A@x
    for i in np.arange(f.N):
        tmp_i = f.g( z[i], i)
        gradients.append(tmp_i)
        
    gradients = np.stack(gradients)
    
    return gradients


# needed for ADAGRAD + SVRG
@njit()
def sparse_batch_gradient(f, A, x, S):
    """
    computes a vector of gradients at point x with mini batch S
    returns: array of shape S
    """   
    
    # initialize object for storing all gradients
    gradients = np.zeros(len(S))
        
    for j, i in enumerate(S):
        z_i = A.row(i) @ x
        tmp_i = f.g(z_i, i)
        gradients[j] = tmp_i
    
    return gradients

# needed for SVRG
@njit()
def compute_AS(A, S):
    A_S = np.zeros((len(S), A.ncols))
    
    for j,i in enumerate(S):
        A_S[j,:] = A.row(i)
    
    return A_S

#%% tick wrapper

def solve_with_tick(f, phi, A, alpha, n_epochs, tol, verbose):
    
    assert f.name == 'logistic'
    assert phi.name == '1norm'
    
    tick = tickLogReg(fit_intercept=False, penalty='l1', C=1/phi.lambda1, solver='svrg', step=alpha,
                      max_iter=n_epochs, tol=tol,
                      verbose=verbose, record_every=1, print_every=1)

    # A = X_train*y_train, y_train in {-1,1}
    y_train = f.b  
    X_train = A.multiply(y_train.reshape(-1,1)).tocsr()
    tick.fit(X_train, y_train)
    
    x_hist = tick._solver_obj.history.values['x']
    x = x_hist[-1]
    
    rt = tick._solver_obj.history.last_values['time']
    
    eta = np.inf
    step_sizes = alpha*np.ones(n_epochs)
    return x, x_hist, step_sizes, eta, rt




