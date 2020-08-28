import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import Lasso


from ssnsp.helper.data_generation import lasso_test
from ssnsp.solver.opt_problem import problem

N = 10000
n = 1000
k = 100
l1 = .01

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False)


#%% solve with scikt to get true solution
sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-8, selection = 'cyclic', max_iter = 1e6)

start = time.time()
sk.fit(f.A,b)
end = time.time()

print(end-start)

xsol = sk.coef_.copy()

f_star = f.eval(xsol) + phi.eval(xsol)
print("Optimal value: ", f_star)

#%%

alpha_0 = np.array([.1, 1., 10., 100., 1000.])
alpha_0 = np.logspace(-1,3,5)

batch_size = np.array([0.01, 0.05, 0.1, 0.2])

A, B = np.meshgrid(alpha_0, batch_size)

# K ~ len(batch_size), L ~ len(alpha_0)
K,L = A.shape


f_tol = 1e-3 
err = lambda x: np.linalg.norm(x-xsol)


all_time = np.zeros_like(A)
converged = np.zeros_like(A)

for k in np.arange(K):
    for l in np.arange(L):
        
        params = {'sample_style': 'fast_increasing', 'reduce_variance': True}
        
        # target M epochs 
        params["max_iter"] = int(15 *  1/B[k,l])
        params['sample_size'] = max(1, int(B[k,l] * f.N))
        params['alpha_C'] = A[k,l]
        
        print(params)
        
        P = problem(f, phi, tol = 1e-6, params = params, verbose = False, measure = True)
        P.solve(solver = 'ssnsp')
        
        
        xhist = P.info['iterates'].copy()
        obj = P.info['objective'].copy()
        
        print("Last objective value: ", obj[-1])
        
        if np.any(obj <= f_star + f_tol):
            stop = np.where(obj <= f_star + f_tol)[0][0]
            this_time = P.info['runtime'].cumsum()[stop]
            
            converged[k,l] = 1
        else:
            this_time = P.info['runtime'].sum()
            print("NO CONVERGENCE!")
            
        
        all_time[k,l] = this_time
        
converged = converged.astype(bool)       

#%%
fig, ax = plt.subplots()

colors = sns.color_palette("GnBu_d", K)
colors = sns.color_palette("viridis", K)

for k in np.arange(K):
    
    ax.plot(alpha_0, all_time[k,:], c = colors[k], label = rf"batch size  $N_S =  N \cdot$ {batch_size[k]} ")
    
    nc = ~converged[k,:]
    
    c_arr = np.array(colors[k]).reshape(1,-1)
    ax.scatter(alpha_0[nc], all_time[k,:][nc], marker = 'x', c = c_arr)
    ax.scatter(alpha_0[~nc], all_time[k,:][~nc], marker = 'o', c = c_arr, s = 5)


ax.set_xlabel(r"Initial step size $\alpha_0$")    
ax.set_ylabel(r"Runtime until convergence [sec]")    

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()