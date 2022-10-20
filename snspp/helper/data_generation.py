"""
@author: Fabian Schaipp
"""
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

from sklearn.datasets import make_low_rank_matrix
from sklearn.preprocessing import PolynomialFeatures

import scipy.special as sp
from scipy.stats import ortho_group, t
from scipy.sparse.csr import csr_matrix

from .loss1 import lsq, logistic_loss, block_lsq
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

def create_matrix(N, n, dist = 'ortho', kappa = 1.):
    """
    method for creating a random matrix
    
    Parameters
    ----------
    N : int
        number of rows.
    n : int
        number of cols.
    dist : str, optional
        * 'ortho': B = QD and columns of B are from random orthogonal group.
        * 'unif': B has uniform entries between -1 and 1.
        The default is 'ortho'.
    kappa : float, optional
        Condition number of B. Only available for dist = 'ortho'. The default is 1.

    Returns
    -------
    B : array of shape (N,n)
    """
    if dist == 'ortho':
        # if N >= n:
        #     Q = ortho_group.rvs(N)
        #     d = np.linspace(kappa, 1, n)
        #     D = np.diag(d)
        #     B = Q[:,:n]@D
        # else:
        B = 2*np.random.rand(N,n)-1
        U,S,Vh = np.linalg.svd(B, full_matrices = False)
        d = np.linspace(kappa, 1, min(N,n))
        B = U@np.diag(d)@Vh
            
    elif dist == 'unif':
        B = np.random.rand(N,n)*2 - 1
    else:
        raise KeyError("Choose 'unif' or 'ortho' as input for argument dist.")
    
        
    return B
    

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
    
    X_train = create_matrix(m.sum(), n, kappa = kappa, dist = dist)
    
    # create true solution
    beta = np.random.randn(k) 
    beta = np.concatenate((beta, np.zeros(n-k)))
    np.random.shuffle(beta)
    
    # create measurements
    y_train = X_train @ beta + noise*np.random.randn(m.sum())
    
    X_train = np.ascontiguousarray(X_train.astype('float64'))
    y_train = y_train.astype('float64')
    beta = beta.astype('float64')
    
    N_test = max(100,int(N*0.1))
    X_test = create_matrix(N_test, n, kappa = kappa, dist = dist)
    y_test = X_test @ beta + noise*np.random.randn(N_test)
    
    
    phi = L1Norm(lambda1)    
    if block:
        f = block_lsq(y_train, m)
    else:
        f = lsq(y_train)
    A = X_train.copy()

    return f, phi, A, X_train, y_train, X_test, y_test, beta

def logreg_test(N = 10, n = 20, k = 5, lambda1 = .1, noise = 0, kappa = 1., dist = 'ortho', seed = 1234):
    """
    creates data for l1-regularized logistic regression with n variables and N samples, where solution has k non-zero entries
    lambda1: regularization parameter of 1-norm
    b \in{-1,1}
    noise = probability of flipping b after generation --> the closer noise is to 1, the nosier the problem becomes
    """
    np.random.seed(seed)
    
    N_test = max(100,int(N*0.1))
    X = create_matrix(N+N_test, n, kappa = kappa, dist = dist)
        
    # create true solution
    beta = np.random.randn(k) 
    beta = np.concatenate((beta, np.zeros(n-k)))
    np.random.shuffle(beta)
    
    h = np.exp(X@beta)
    odds = h/(1+h)
    
    y = (odds >= .5)*2 -1
    #y = np.random.binomial(1,p=odds)*2 - 1
    
    if noise > 0:
        assert noise <= 1
        flip = np.random.binomial(n=1, p = noise, size = N+N_test)
        flip = (1 - flip * 2)      
        # flip signs (f in {-1,1})
        y = y * flip
     
    X = np.ascontiguousarray(X.astype('float64'))
    y = y.astype('float64')
    beta = beta.astype('float64')
    
    X_train = X[:N,:]
    y_train = y[:N]
    
    phi = L1Norm(lambda1) 
    f = logistic_loss(y_train)
    A = X_train * y_train.reshape(-1,1) # logistic loss has a_i*b_i
    
    ##### TEST SET ############
    X_test = X[N:,:]
    y_test = y[N:]
    
    return f, phi, A, X_train, y_train, X_test, y_test, beta

def tstudent_test(N = 10, n = 20, k = 5, lambda1 = .1, v = 4., noise = 0.1, poly = 0, kappa = 1., dist = 'ortho', seed = 23456):
    
    np.random.seed(seed)
    
    N_test = max(100,int(N*0.1))
    X = create_matrix(N+N_test, n, kappa = kappa, dist = dist)
    
    if poly > 0:
        X = poly_expand(X, d=poly)
        k = int(X.shape[1]*k/n)
        n = X.shape[1]
    
    # create true solution
    beta = np.random.randn(k)
    beta = np.concatenate((beta, np.zeros(n-k)))
    np.random.shuffle(beta)
           
    y = X@beta + noise*np.random.standard_t(v, size = N+N_test)
     
    X = np.ascontiguousarray(X.astype('float64'))
    y = y.astype('float64')
    beta = beta.astype('float64')
    
    X_train = X[:N,:]
    y_train = y[:N]
    
    phi = L1Norm(lambda1) 
    f = tstudent_loss(y_train, v=v)
    A = np.ascontiguousarray(X_train)
    
    ##### TEST SET ############
    X_test = X[N:,:]
    y_test = y[N:]
        
    
    return f, phi, A, X_train, y_train, X_test, y_test, beta

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
    
    X_train = create_matrix(N, n, kappa = kappa, dist = dist)
    
    # create true solution
    beta = np.random.randn(k) 
    beta = np.concatenate((beta, np.zeros(n-k)))
    np.random.shuffle(beta)
    
    # create measurements
    y_train = X_train @ beta + noise*np.random.randn(N)
    
    X_train = np.ascontiguousarray(X_train.astype('float64'))
    y_train = y_train.astype('float64')
    beta = beta.astype('float64')
    
    N_test = max(100,int(N*0.1))
    X_test = create_matrix(N_test, n, kappa = kappa, dist = dist)
    y_test = X_test @ beta + noise*np.random.randn(N_test)
    
    
    phi = L1Norm(lambda1) 
    mu_arr = mu * np.ones(N) 
    f = huber_loss(y_train, mu_arr)
    A = X_train.copy()
    
    return f, phi, A, X_train, y_train, X_test, y_test, beta

def lowrank_test(N = 10, p = 20, q = 30, r = 5, lambda1 = .1, noise = 0., kappa = 1., dist = 'ortho', seed = 23456):
    """
    noise: std. deviation of Gaussian noise added to measurements b
    kappa: if not None, A is created such that is has condition sqrt(kappa)
    """
    np.random.seed(seed)
    
    X = np.zeros((p,q,N))
    for j in np.arange(N):
        X[:,:,j] = create_matrix(p, q, kappa = kappa, dist = dist)
    
    # create true solution
    beta = make_low_rank_matrix(p, q, effective_rank=r, tail_strength=0.5, random_state=23456)
    
    # create measurements
    Y = multiple_matdot(X, beta) + noise*np.random.randn(N)
    
    X_train = np.ascontiguousarray(X.astype('float64'))
    Y_train = Y.astype('float64')
    
    N_test = max(100,int(N*0.1))
    X_test = np.zeros((p,q,N_test))
    for j in np.arange(N_test):
        X_test[:,:,j] = create_matrix(p, q, kappa = kappa, dist = dist)
    Y_test = multiple_matdot(X_test, beta) + noise*np.random.randn(N_test)
    
    phi = NuclearNorm(lambda1) 
    f = mat_lsq(Y_train)
    A = X_train.copy()
        
    return f, phi, A, X_train, Y_train, X_test, Y_test, beta

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
    X, y0 = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
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
    
    A = X_train * y_train.reshape(-1,1) # logistic loss has a_i*b_i
    phi = L1Norm(lambda1) 
    f = logistic_loss(y_train)
    
    return f, phi, A, X_train, y_train, X_test, y_test

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
    
    A = X_train * y_train.reshape(-1,1) # logistic loss has a_i*b_i
    phi = L1Norm(lambda1) 
    f = logistic_loss(y_train)
        
    return f, phi, A, X_train, y_train, X_test, y_test

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
        
    A = X_train * y_train.reshape(-1,1) # logistic loss has a_i*b_i
    phi = L1Norm(lambda1) 
    f = logistic_loss(y_train)
    
        
    return f, phi, A, X_train, y_train, X_test, y_test

def get_higgs(lambda1 = 0.01, train_size = .8, scale = True, path_prefix = '../'):
    # download from https://archive.ics.uci.edu/ml/datasets/HIGGS
    
    warnings.warn("Loading higgs is highly memory intensive.")
    
    df = pd.read_csv(path_prefix + 'data/HIGGS.csv', header=None)
    y = df.iloc[:,0].values
    X = df.iloc[:,1:].values
    
    # labels are in 0,1 format
    y[y==0] = -1.
    
    assert np.all(np.isin(y,[-1,1]))
    
    X = X.astype('float64')
    y = y.astype('float64')
    
    np.nan_to_num(X, copy = False)
    
    if train_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                        random_state = 1234)
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None
        
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)
        
    A = X_train * y_train.reshape(-1,1) # logistic loss has a_i*b_i
    phi = L1Norm(lambda1) 
    f = logistic_loss(y_train)
    
        
    return f, phi, A, X_train, y_train, X_test, y_test

def get_e2006(lambda1 = 0.01, train_size = None, path_prefix = '../'):
    # download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#E2006-tfidf
    # extract to data/libsvm

    # X is in sparse format
    X, y = load_svmlight_file(path_prefix + 'data/libsvm/E2006.train' )
    
    if train_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                            random_state = 1234)
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None
        
    nu_est = np.round(t.fit(y_train)[0], 2) # estimate degrees of freedom
    print("Estimated degrees of freedom: ", nu_est)
    #tmp = y_train/(nu_est+y_train**2)
    #X_train.multiply(tmp.reshape(-1,1)).tocsr().sum(axis=0)
        
    phi = L1Norm(lambda1) 
    f = tstudent_loss(y_train, v=nu_est)
    A = X_train
    
    return f, phi, A, X_train, y_train, X_test, y_test
        

def get_poly(name = 'madelon', lambda1 = 0.01, train_size = None, scale = True, poly = 0, path_prefix = '../'):
    # using libsvm dataset but with polynomial feature expansion
    
    warnings.warn("Doing polynomial expansion might be memory intensive.")   
    X, y = load_svmlight_file(path_prefix + 'data/libsvm/' + libsvm_dict[name])
    
    assert np.all(np.isin(y,[-1,1]))
    
    # to dense (madelon is dense)
    X = X.toarray()
    y = y.astype('float64')    
    np.nan_to_num(X, copy = False)
    
    # train/test
    if train_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                        random_state = 1234)
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None
    
    # scaling
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)
    
    # polynomial expansion (after scaling)
    if poly > 0:
        X_train = poly_expand(X_train, d=poly)
        if train_size is not None:
            X_test = poly_expand(X_test, d=poly)
            
    A = X_train * y_train.reshape(-1,1) # logistic loss has a_i*b_i
    phi = L1Norm(lambda1) 
    f = logistic_loss(y_train)
            
    return f, phi, A, X_train, y_train, X_test, y_test

##############################
## LIBSVM

libsvm_dict = {'rcv1': 'rcv1_train.binary', 'w8a': 'w8a', 
               'fourclass': 'fourclass_scale',
               'covtype': 'covtype.libsvm.binary.scale',
               'news20': 'news20.binary',
               'madelon': 'madelon',
               'ijcnn': 'ijcnn1.tr'}

def get_libsvm(name, lambda1 = 0.01, train_size = .8, scale = False, path_prefix = '../', dense = False):
    
    X, y = load_svmlight_file(path_prefix + 'data/libsvm/' + libsvm_dict[name])
    
    if name == 'covtype':
        y[y==2] = -1
    
    assert np.all(np.isin(y,[-1,1]))
        
    y = y.astype('float64')
    
    # convert to dense (not recommended)
    if dense:
        warnings.warn("Converting LIBSVM file to dense format. Not recommended if data is very sparse.")
        X = X.toarray()
    
    if train_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size,\
                                                        random_state = 1234)
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None
        
    # is often already scaled from -1 to 1
    # scaling should only be done when in dense format
    if scale and dense:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test = scaler.transform(X_test)
    
    if not dense:
        A = X_train.multiply(y_train.reshape(-1,1)).tocsr() # logistic loss has a_i*b_i
    else:
        A = X_train * y_train.reshape(-1,1)
        
    phi = L1Norm(lambda1) 
    f = logistic_loss(y_train)
    
    return f, phi, A, X_train, y_train, X_test, y_test


############################################################################################
################## REGRESSION

    
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