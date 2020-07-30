"""
@author: Fabian Schaipp
"""

import numpy as np


from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

#from sklearn.datasets import load_digits
#from scipy.stats import ortho_group

from .lasso import Norm1, lsq, block_lsq, logistic_loss

############################################################################################
### Synthetic data
############################################################################################

def A_target_condition(N, n, smax = 100, smin = 1):
    
    A = np.random.randn(N,n)
    U,_,V = np.linalg.svd(A, full_matrices = False)
    
    d = np.linspace(np.sqrt(smax), np.sqrt(smin), min(n,N))
    
    # if n > N:
    #     D = np.hstack((np.diag(d), np.zeros((N,n-N))))  
    # else:
    #     D = np.vstack((np.diag(d), np.zeros((N-n,n)) ))
    
    D = np.diag(d)
    A = U @ D @ V
    
    return A
 
def lasso_test(N = 10, n = 20, k = 5, lambda1 = .1, block = False, kappa = None):
    if block:
        m = np.random.randint(low = 3, high = 10, size = N)
    else:
        m = np.ones(N, dtype = 'int')
    
    if kappa is None:     
        A = np.random.randn(m.sum(),n)
        
        # standardize
        A = A - A.mean(axis=0)
        A = (1/A.std(axis=0)) * A
        
        assert max(abs(A.mean(axis=0))) <= 1e-5
        assert max(abs(A.std(axis=0) - 1)) <= 1e-5
    
    else:
        assert kappa > 1
        A = A_target_condition(m.sum(), n, smax = kappa)
    
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

def logreg_test(N = 10, n = 20, k = 5, lambda1 = .1, noise = 0, kappa = None):
    """
    creates A, b for logistic regression
    b \in{-1,1}
    noise = probability of flipping b after generation --> the closer noise is to 1, the nosier the problem becomes
    """
    #np.random.seed(1234)
    
    if kappa is None:     
        A = np.random.randn(N,n)
        
        # standardize
        A = A - A.mean(axis=0)
        A = (1/A.std(axis=0)) * A
        
        assert max(abs(A.mean(axis=0))) <= 1e-5
        assert max(abs(A.std(axis=0) - 1)) <= 1e-5
    
    else:
        assert kappa > 1
        A = A_target_condition(N, n, smax = kappa)
        
    # create true solution
    x = np.random.randn(k) 
    x = np.concatenate((x, np.zeros(n-k)))
    np.random.shuffle(x)
    
    h = np.exp(A@x)
    odds = h/(1+h)
    
    b = (odds >= .5)*2 -1
    #b = np.random.binomial(1,p=odds)*2 - 1
    
    if noise > 0:
        assert noise <= 1
        f = np.random.binomial(n=1, p = noise, size = N)
        f = (1 - f * 2)
        
        # flip signs (f in {-1,1})
        b = b * f
    
    
    A = A.astype('float64')
    b = b.astype('float64')
    x = x.astype('float64')
    
    phi = Norm1(lambda1) 
    f = logistic_loss(A,b)
    
    return x, A, b, f, phi

############################################################################################
### Actual data
############################################################################################

def get_mnist(lambda1 = 0.02, train_size = .8, scale = True):

    # Load data from https://www.openml.org/d/554
    X, y0 = fetch_openml('mnist_784', version=1, return_X_y=True)
    y0 = y0.astype('float64')
    y = y0.copy()
    
    set1 = [0,3,6,8,9]
    set2 = [1,2,4,5,7]
    
    y[np.isin(y0, set1)] = 1
    y[np.isin(y0, set2)] = -1
    
    assert np.all(np.isin(y,[-1,1]))
    
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation].astype('float64')
    y = y[permutation].astype('float64')
    #X = X.reshape((X.shape[0], -1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size)
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    phi = Norm1(lambda1) 
    f = logistic_loss(X_train, y_train)

    return f, phi, X_train, y_train, X_test, y_test

def get_rcv1(lambda1 = 0.02, train_size = .8, scale = True):
    
    rcv1 = fetch_rcv1()
    
    X = rcv1.data.astype('float64')
    y = rcv1.target[:,0].astype('float64')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size)
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    phi = Norm1(lambda1) 
    f = logistic_loss(X_train, y_train)
    
    return f, phi, X_train, y_train, X_test, y_test
# def get_mnist_dataset():
    
#     digits = load_digits()
#     A = digits.data.astype('float64')
#     b = (digits.target >= 5)
#     b = (b*2-1).astype('float64')
    
#     return A,b