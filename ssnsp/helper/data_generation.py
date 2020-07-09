"""
@author: Fabian Schaipp
"""

import numpy as np

from .lasso import Norm1, lsq, block_lsq, logistic_loss

def lasso_test(N = 10, n = 20, k = 5, lambda1 = .1, block = False):
    if block:
        m = np.random.randint(low = 3, high = 10, size = N)
    else:
        m = np.ones(N, dtype = 'int')
    
    if not block:
        A = np.random.randn(N,n)
    else:
        A = []
        for i in np.arange(N):
            A.append(np.random.randn(m[i], n))    
        A = np.vstack(A)
    
    # standardize
    A = A - A.mean(axis=0)
    A = (1/A.std(axis=0)) * A
    
    assert max(abs(A.mean(axis=0))) <= 1e-5
    assert max(abs(A.std(axis=0) - 1)) <= 1e-5
    
    # create true solution
    x = np.random.randn(k) 
    x = np.concatenate((x, np.zeros(n-k)))
    np.random.shuffle(x)
    
    # create measurements
    b = A @ x
    
    A = A.astype('float64')
    b = b.astype('float64')
    x = x.astype('float64')
    
    phi = Norm1(lambda1)    
    if block:
        f = block_lsq(A, b, m)
    else:
        f = lsq(A, b)
        

    return x, A, b, f, phi

def logreg_test(N = 10, n = 20, k = 5, lambda1 = .1):
    
    np.random.seed(1234)
    
    A = np.random.randn(N,n)
    
    # standardize
    A = A - A.mean(axis=0)
    A = (1/A.std(axis=0)) * A
    
    assert max(abs(A.mean(axis=0))) <= 1e-5
    assert max(abs(A.std(axis=0) - 1)) <= 1e-5
    
    # create true solution
    x = np.random.randn(k) 
    x = np.concatenate((x, np.zeros(n-k)))
    np.random.shuffle(x)
    
    h = np.exp(A@x)
    odds = h/(1+h)
    
    #b = (odds >= .5)*2 -1
    b = np.random.binomial(1,p=odds)*2 - 1
    
    
    A = A.astype('float64')
    b = b.astype('float64')
    x = x.astype('float64')
    
    phi = Norm1(lambda1) 
    f = logistic_loss(A,b)
    
    return x, A, b, f, phi
