import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lasso import Norm1, lsq
from opt_problem import problem

def lasso_test(N = 10, n = 20, k = 5, lambda1 = .1):
    
    m = np.ones(N, dtype = 'int')

    A = []
    for i in np.arange(N):
        A.append(np.random.rand(m[i], n))
    
    
    A = np.vstack(A)
    
    x = np.random.randn(k) 
    x = np.concatenate((x, np.zeros(n-k)))
    np.random.shuffle(x)
    
    b = A @ x
    
    phi = Norm1(lambda1)    
    #phi.prox(np.ones(3), alpha = 1)
    
    f = lsq(A, b)

    return x, A, b, f, phi
#%%
N = 100
n = 500
k = 5
l1 = .01

xsol, A, b, f, phi = lasso_test(N, n, k, l1)

P = problem(f, phi, verbose = True)

P.solve()

P.plot_path()

P.plot_samples()

tmp = pd.DataFrame(np.vstack((xsol, P.xavg)).T, columns = ['true', 'estimated'])


#%%
sub_rsd = P.info['ssn_info']

fig, axs = plt.subplots(5,10)

for j in np.arange(50):
    ax = axs.ravel()[j]
    ax.plot(sub_rsd[j]['residual'], 'blue')
    ax2 = ax.twinx()
    ax2.plot(sub_rsd[j]['step_size'], 'orange')
    ax2.plot(sub_rsd[j]['direction'], 'green')
    
    ax.set_yscale('log')
    
#fig.legend(['residual', 'step_size', 'direction'])
#%%

#m = np.random.randint(low = 3, high = 10, size = N)


