import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Lasso, LogisticRegression


from ssnsp.helper.data_generation import lasso_test, logreg_test
from ssnsp.solver.opt_problem import problem

N = 5000
n = 50
k = 10
l1 = .01

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False)


#%% solve with scikt to get true solution
sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-8, selection = 'cyclic')

sk.fit(A,b)
xsol = sk.coef_.copy()

#%%

params = {'max_iter' : 120, 'alpha_C' : 100.}
err = lambda x: np.linalg.norm(x-xsol)

num_p = 5
sizeS = np.linspace(N/10, N-1, num_p).astype(int)


all_err = list()
all_time = list()

for j in np.arange(num_p):
    params['sample_size'] = sizeS[j]

    P = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True )

    start = time.time()
    P.solve(solver = 'ssnsp')
    end = time.time()
    
    xhist = P.info['iterates'].copy()
    
    print(f"Batch size: {np.round(sizeS[j]/N, 2)} * N")
    print(f"Computing time: {end-start} sec")


    this_err = np.apply_along_axis(err, axis = 1, arr = xhist)
    this_time = P.info['runtime'].cumsum()
    
    all_err.append(this_err)
    all_time.append(this_time)
    
#%%

fig, ax = plt.subplots(1,1)

for j in np.arange(num_p):
    ax.plot( all_time[j],  all_err[j])
    
ax.set_yscale('log')

   
#%%

fig, axs = plt.subplots(1,2)

ax = axs[0]
for j in np.arange(num_p):
    ax.plot(all_err[j])
    
ax.set_yscale('log')


ax = axs[1]
for j in np.arange(num_p):
    ax.plot(all_time[j])
    
ax.set_yscale('log')


