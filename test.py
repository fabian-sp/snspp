import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lasso import Norm1, lsq
from opt_problem import problem

def lasso_test(N = 10, n = 20, k = 5):
    
    m = np.ones(N, dtype = 'int')

    A = []
    for i in np.arange(N):
        A.append(np.random.rand(m[i], n))
    
    
    A = np.vstack(A)
    
    x = np.random.randn(k) 
    x = np.concatenate((x, np.zeros(n-k)))
    np.random.shuffle(x)
    
    b = A @ x
    
    phi = Norm1(.1)    
    phi.prox(np.ones(3), alpha = 1)
    
    f = lsq(A, b)

    return x, A, b, f, phi
#%%
N = 40
n = 50
k = 5
xsol, A, b, f, phi = lasso_test(N, n, k)

P = problem(f, phi, A, verbose = True)

P.solve()

P.plot_path()

P.plot_samples()
#%%

#m = np.random.randint(low = 3, high = 10, size = N)


np.isin(np.arange(N), P.info['samples'])


