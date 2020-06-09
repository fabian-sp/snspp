import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LogisticRegression

from lasso import Norm1, lsq, block_lsq, logistic_loss
from opt_problem import problem

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
    
    phi = Norm1(lambda1)    
    if block:
        f = block_lsq(A, b, m)
    else:
        f = lsq(A, b)

    return x, A, b, f, phi

def logreg_test(N = 10, n = 20, k = 5, lambda1 = .1):
    
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
    
    b = (odds >= .5)*2 -1
    
    phi = Norm1(lambda1) 
    f = logistic_loss(A,b)
    
    return x, A, b, f, phi
    
#%% generate data

N = 10000
n = 500
k = 10
l1 = .1

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False)

xsol, A, b, f, phi = logreg_test(N, n, k, l1)


#%% solve with SPP
params = {'max_iter' : 50, 'sample_size': 200, 'alpha_C' : 10.}

P = problem(f, phi, params = params, verbose = True)

P.solve()

P.plot_path()
P.plot_samples()
P.plot_objective()

info = P.info.copy()

#%% compare to scikit

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-5, selection = 'cyclic')
sk = LogisticRegression(penalty = 'l1', C = 1/(N*l1), fit_intercept= False, solver = 'saga', max_iter = 100)


sk.fit(A,b)
x_sk = sk.coef_.copy()

all_x = pd.DataFrame(np.vstack((xsol, P.x, x_sk)).T, columns = ['true', 'spp', 'scikit'])

#%% newton convergence

sub_rsd = P.info['ssn_info']

fig, axs = plt.subplots(5,10)
fig.legend(['residual', 'step_size', 'direction'])

for j in np.arange(50):
    ax = axs.ravel()[j]
    ax.plot(sub_rsd[j]['residual'], 'blue')
    ax2 = ax.twinx()
    ax2.plot(sub_rsd[j]['step_size'], 'orange')
    ax2.plot(sub_rsd[j]['direction'], 'green')
    
    ax.set_yscale('log')
