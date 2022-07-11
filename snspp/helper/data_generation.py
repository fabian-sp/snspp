"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

from sklearn.datasets import make_low_rank_matrix
from sklearn.preprocessing import PolynomialFeatures

import scipy.special as sp
from scipy.stats import ortho_group

from .loss1 import lsq, block_lsq, logistic_loss
from .loss2 import huber_loss
from .regz import L1Norm, Zero, Ridge
from .tstudent import tstudent_loss

from ..matopt.nuclear import NuclearNorm
from ..matopt.mat_loss import mat_lsq
from ..matopt.utils import multiple_matdot

############################################################################################
### Synthetic data
############################################################################################

def standardize(A):   
    # standardize columns of A
    M = A - A.mean(axis=0)
    M = (1/(M.std(axis=0))) * M
        
    assert max(abs(M.mean(axis=0))) <= 1e-5
    assert max(abs(M.std(axis=0) - 1)) <= 1e-5
    
    return M

def create_A(N, n, dist = 'ortho', kappa = 1.):
    """
    method for creating a random matrix A
    
    Parameters
    ----------
    N : int
        number of rows.
    n : int
        number of cols.
    dist : str, optional
        * 'ortho': A = QD and columns of A are from random orthogonal group.
        * 'unif': A has uniform entries between -1 and 1.
        The default is 'ortho'.
    kappa : float, optional
        Condition number of A. Only available for dist = 'ortho'. The default is 1.

    Returns
    -------
    A : array of shape (N,n)
    """
    if dist == 'ortho':
        if N >= n:
            Q = ortho_group.rvs(N)
            d = np.linspace(kappa, 1, n)
            D = np.diag(d)
            A = Q[:,:n]@D
        else:
            A = 2*np.random.rand(N,n)-1
            U,S,Vh = np.linalg.svd(A, full_matrices = False)
            d = np.linspace(kappa, 1, min(N,n))
            A = U@np.diag(d)@Vh
                
        #print("Condition number of A:", np.linalg.cond(A) )
        
    elif dist == 'unif':
        A = np.random.rand(N,n)*2 - 1
    else:
        raise KeyError("Choose 'unif' or 'ortho' as input for argument dist.")
    
        
    return A
    

def lasso_test(N = 10, n = 20, k = 5, lambda1 = .1, block = False, noise = 0., kappa = 1., dist = 'ortho', seed = 1234):
    """
    generates data for a LASSO problem with n variables and N samples, where solution has k non-zero entries
    lambda1: regularization parameter of 1-norm
    block: if True, the A_i have different dimensions (>1 rows)
    noise: std. deviation of Gaussian noise added to measurements b
    kappa: if not None, A is created such that is has condition sqrt(kappa)
    """
    np.random.seed(seed)
    
    if block:
        m = np.random.randint(low = 3, high = 10, size = N)
    else:
        m = np.ones(N, dtype = 'int')
    
    A = create_A(m.sum(), n, kappa = kappa, dist = dist)
    
    # create true solution
    x = np.random.randn(k) 
    x = np.concatenate((x, np.zeros(n-k)))
    np.random.shuffle(x)
    
    # create measurements
    b = A @ x + noise*np.random.randn(m.sum())
    
    A = np.ascontiguousarray(A.astype('float64'))
    b = b.astype('float64')
    x = x.astype('float64')
    
    N_test = max(100,int(N*0.1))
    A_test = create_A(N_test, n, kappa = kappa, dist = dist)
    b_test = A_test @ x + noise*np.random.randn(N_test)
    
    
    phi = L1Norm(lambda1)    
    if block:
        f = block_lsq(A, b, m)
    else:
        f = lsq(A, b)
        

    return x, A, b, f, phi, A_test, b_test

def logreg_test(N = 10, n = 20, k = 5, lambda1 = .1, noise = 0, kappa = 1., dist = 'ortho', seed = 1234):
    """
    creates data for l1-regularized logistic regression with n variables and N samples, where solution has k non-zero entries
    lambda1: regularization parameter of 1-norm
    b \in{-1,1}
    noise = probability of flipping b after generation --> the closer noise is to 1, the nosier the problem becomes
    """
    np.random.seed(seed)
    
    N_test = max(100,int(N*0.1))
    A = create_A(N+N_test, n, kappa = kappa, dist = dist)
        
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
        flip = np.random.binomial(n=1, p = noise, size = N+N_test)
        flip = (1 - flip * 2)      
        # flip signs (f in {-1,1})
        b = b * flip
     
    A = np.ascontiguousarray(A.astype('float64'))
    b = b.astype('float64')
    x = x.astype('float64')
    
    phi = L1Norm(lambda1) 
    f = logistic_loss(A[:N,:],b[:N])
    
    ##### TEST SET ############
    A_test = A[N:,:]
    b_test = b[N:]
    
    return x, A[:N,:], b[:N], f, phi, A_test, b_test

def tstudent_test(N = 10, n = 20, k = 5, lambda1 = .1, v = 4., noise = 0.1, poly = 0, kappa = 1., dist = 'ortho', seed = 23456):
    
    np.random.seed(seed)
    
    N_test = max(100,int(N*0.1))
    A = create_A(N+N_test, n, kappa = kappa, dist = dist)
    
    if poly > 0:
        A = poly_expand(A, d=poly)
        k = int(A.shape[1]*k/n)
        n = A.shape[1]
    
    # create true solution
    x = np.random.randn(k)
    x = np.concatenate((x, np.zeros(n-k)))
    np.random.shuffle(x)
           
    b = A@x + noise*np.random.standard_t(v, size = N+N_test)
     
    A = np.ascontiguousarray(A.astype('float64'))
    b = b.astype('float64')
    x = x.astype('float64')
    
    phi = L1Norm(lambda1) 
    f = tstudent_loss(A[:N,:],b[:N],v=v)
        
    A_test = A[N:,:]
    b_test = b[N:]
    
    return x, A[:N,:], b[:N], f, phi, A_test, b_test

def huber_test(N = 10, n = 20, k = 5, lambda1 = .1, mu = 1., noise = 0., kappa = 1., dist = 'ortho', seed = 23456):
    """
    generates data for a Huber regression problem with n variables and N samples, where solution has k non-zero entries
    lambda1: regularization parameter of 1-norm
    mu: parameter for the Huber function
    block: if True, the A_i have different dimensions (>1 rows)
    noise: std. deviation of Gaussian noise added to measurements b
    kappa: if not None, A is created such that is has condition sqrt(kappa)
    """
    np.random.seed(seed)
    
    A = create_A(N, n, kappa = kappa, dist = dist)
    
    # create true solution
    x = np.random.randn(k) 
    x = np.concatenate((x, np.zeros(n-k)))
    np.random.shuffle(x)
    
    # create measurements
    b = A @ x + noise*np.random.randn(N)
    
    A = np.ascontiguousarray(A.astype('float64'))
    b = b.astype('float64')
    x = x.astype('float64')
    
    N_test = max(100,int(N*0.1))
    A_test = create_A(N_test, n, kappa = kappa, dist = dist)
    b_test = A_test @ x + noise*np.random.randn(N_test)
    
    
    phi = L1Norm(lambda1) 
    mu_arr = mu * np.ones(N) 
    f = huber_loss(A, b, mu_arr)
        
    return x, A, b, f, phi, A_test, b_test

def lowrank_test(N = 10, p = 20, q = 30, r = 5, lambda1 = .1, noise = 0., kappa = 1., dist = 'ortho', seed = 23456):
    """
    noise: std. deviation of Gaussian noise added to measurements b
    kappa: if not None, A is created such that is has condition sqrt(kappa)
    """
    np.random.seed(seed)
    
    A = np.zeros((p,q,N))
    for j in np.arange(N):
        A[:,:,j] = create_A(p, q, kappa = kappa, dist = dist)
    
    # create true solution
    X = make_low_rank_matrix(p, q, effective_rank=r, tail_strength=0.5, random_state=23456)
    
    # create measurements
    b = multiple_matdot(A, X) + noise*np.random.randn(N)
    
    A = np.ascontiguousarray(A.astype('float64'))
    b = b.astype('float64')
    
    N_test = max(100,int(N*0.1))
    A_test = np.zeros((p,q,N_test))
    for j in np.arange(N_test):
        A_test[:,:,j] = create_A(p, q, kappa = kappa, dist = dist)
    b_test = multiple_matdot(A_test, X) + noise*np.random.randn(N_test)
    
    phi = NuclearNorm(lambda1) 
    f = mat_lsq(A, b)
        
    return X, A, b, f, phi, A_test, b_test

def poly_expand(A, d = 5):

    n = A.shape[1]
    print("Number of features after polynomial expansion: ", sp.comb(n+d, d, True))
    
    poly = PolynomialFeatures(d)
    return poly.fit_transform(A)

#%%
############################################################################################
### Actual data - CLASSIFICATION
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
    
    X = X.astype('float64')
    y = y.astype('float64')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                        random_state = 12345)
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    phi = L1Norm(lambda1) 
    f = logistic_loss(X_train, y_train)

    return f, phi, X_train, y_train, X_test, y_test

def get_gisette(lambda1 = 0.02, train_size = .8, path_prefix = '../'):
    # download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#gisette
    
    X = np.load(path_prefix + 'data/gisette_X.npy')
    y = np.load(path_prefix + 'data/gisette_y.npy')
    
    assert np.all(np.isin(y, [-1,1]))
    
    X_train = X.astype('float64')
    y_train = y.astype('float64')
    np.nan_to_num(X_train, copy = False)
    
    # use only train set and split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size = train_size,\
                                                        random_state = 1234)
    
    # use original train and test set split (caveat: test set has additional features)
    # X_test = np.load(path_prefix + 'data/gisette_test_X.npy')
    # y_test = np.load(path_prefix + 'data/gisette_test_y.npy')
    
    # assert np.all(np.isin(y_test, [-1,1]))
    
    # X_test = X_test.astype('float64')
    # y_test = y_test.astype('float64')
    # np.nan_to_num(X_test, copy = False)
    
    
    phi = L1Norm(lambda1) 
    f = logistic_loss(X_train, y_train)
        
    return f, phi, X_train, y_train, X_test, y_test

def get_sido(lambda1 = 0.02, train_size = .8, scale = False, path_prefix = '../'):
    # download from http://www.causality.inf.ethz.ch/challenge.php?page=datasets
    
    X = np.loadtxt(path_prefix + 'data/sido0/sido0_train.data')
    y = np.loadtxt(path_prefix + 'data/sido0/sido0_train.targets')
    
    assert np.all(np.isin(y,[-1,1]))
    
    X = X.astype('float64')
    y = y.astype('float64')
    
    np.nan_to_num(X, copy = False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                        random_state = 1234)
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    phi = L1Norm(lambda1) 
    f = logistic_loss(X_train, y_train)
        
    return f, phi, X_train, y_train, X_test, y_test

def get_w8a(lambda1 = 0.01, train_size = .8, scale = False, path_prefix = '../'):
    X, y = load_svmlight_file(path_prefix + 'data/libsvm/w8a')
    assert np.all(np.isin(y,[-1,1]))
    
    X = X.toarray().astype('float64') # sparse to dense
    y = y.astype('float64')
    
    np.nan_to_num(X, copy = False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                        random_state = 1234)
    
    # is already scaled from -1 to 1 so not really needed
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    phi = L1Norm(lambda1) 
    f = logistic_loss(X_train, y_train)
    
    return f, phi, X_train, y_train, X_test, y_test

# dataset with only two features, serves well for visualization
def get_fourclass(lambda1 = 0.001, reg = None, train_size = .99, path_prefix = '../'):
    # download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/fourclass_scale
    
    X, y = load_svmlight_file(path_prefix + 'data/libsvm/fourclass_scale')
    assert np.all(np.isin(y,[-1,1]))
    
    X = X.toarray().astype('float64') # sparse to dense
    y = y.astype('float64')
    
    np.nan_to_num(X, copy = False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                        random_state = 1234)
    
    if reg is None:
        phi = Zero() 
    elif reg == 'l1':
        phi = L1Norm(lambda1)
    elif reg == 'l2':
        phi = Ridge(lambda1)
        
    f = logistic_loss(X_train, y_train)
        
    return f, phi, X_train, y_train, X_test, y_test

############################################################################################
################## REGRESSION

def get_triazines(lambda1 = 0.01, train_size = .8, v = 1, poly = 0, noise = 0, path_prefix = '../'):
    # download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/triazines_scale
    assert v > 0
    
    X,y = load_from_txt('triazines', path_prefix) 
    y += noise*np.random.standard_t(v, size = len(y))
    
    X = X.astype('float64')
    y = y.astype('float64')
    
    if poly > 0:
        X = poly_expand(X, d=poly)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                        random_state = 1234)
    
    phi = L1Norm(lambda1) 
    f = tstudent_loss(X_train, y_train, v=v)
    
    return f, phi, X_train, y_train, X_test, y_test
    
#%% for loading libsvm data from .txt-file

def load_from_txt(name, path_prefix = '', as_numpy = True):
    """
    Parameters
    ----------
    name : str
        name of the .txt file, e.g. 'gisette'.

    Returns
    -------
    X : array
        input features.
    y : array
        response/class labels.

    """
    with open(path_prefix + f'data/libsvm/{name}.txt', 'r') as file:
            
        data = []
        labels = []
        for line in file:
            
            tmp = line.split(' ')
            label = float(tmp[0])# use int() for classification datasets
            feat = tmp[1:]
            
            keys = []
            vals = []
            
            for f in feat:
                if f == '\n':
                    continue

                keys.append(f.split(':')[0])
                vals.append(float(f.split(':')[1]))
            
            d = dict(zip(keys,vals))
            data.append(d)
            labels.append(label)
        
    print("Done reading")
                
    y = np.array(labels)
    if as_numpy:
        X = pd.DataFrame(data).values.astype('float64')
    else:
        X = pd.DataFrame(data).astype('float64')
        
    return X,y
        
def prepare_gisette(path_prefix = '', test=False):
    """
    function to create gisette npy files from libsvm download.
    Test set has features which are not in train set --> they get deleted.
    """
    X_train, y_train = load_from_txt('gisette', path_prefix, as_numpy=False)    
    
    if test:
        X_test, y_test = load_from_txt('gisette_test', path_prefix, as_numpy=False)
    
        X_test = X_test.loc[:, X_train.columns]
        assert X_test.shape[1] == X_train.shape[1]
    
    np.save('data/gisette_X.npy', X_train)
    np.save('data/gisette_y.npy', y_train)

    if test:
        np.save('data/gisette_test_X.npy', X_test)
        np.save('data/gisette_test_y.npy', y_test)
    
    return