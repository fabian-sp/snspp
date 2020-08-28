import numpy as np
import matplotlib.pyplot as plt
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
sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-8, selection = 'cyclic')

sk.fit(A,b)
xsol = sk.coef_.copy()

f_star = f.eval(xsol) + phi.eval(xsol)
print("Optimal value: ", f_star)

#%%

alpha_0 = [.1, 1., 10., 100., 1000.]
batch_size = [0.01, 0.05, 0.1]

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
        params["max_iter"] = int(10 *  1/B[k,l])
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
            this_time = P.info['runtime'].cumsum()
            print("NO CONVERGENCE!")
            
        
        all_time[k,l] = this_time
        
        

#%%
fig, ax = plt.subplots()

for k in np.arange(K):
    
    ax.plot(alpha_0, all_time[k,:])
    
ax.set_xscale('log')
