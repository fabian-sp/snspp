"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression
import time

from ssnsp.helper.data_generation import lasso_test, logreg_test
from ssnsp.solver.opt_problem import problem
from ssnsp.experiments.experiment_utils import plot_multiple, initialize_fast_gradient


#%% generate data

N = 6000
n = 10000
k = 100
l1 = .2

kappa = 1e6

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False, kappa = None)
#xsol, A, b, f, phi = logreg_test(N, n, k, l1, noise = .5)

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-9, max_iter = 20000, selection = 'cyclic')
#sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-5, solver = 'saga', max_iter = 700000, verbose = 1)

start = time.time()
sk.fit(A,b)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

psi_star = f.eval(x_sk) +phi.eval(x_sk)
print(psi_star)
initialize_fast_gradient(f, phi)

#%% solve with SAGA

params = {'n_epochs' : 100, 'reg': 1e-4}

Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))


#%% solve with ADAGRAD
if kappa >= 1e5:
    params = {'n_epochs' : 1000, 'batch_size': 100, 'gamma': .1}
else:
    params = {'n_epochs' : 400, 'batch_size': 100, 'gamma': .02}
    
Q1 = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))

#%% solve with SSNSP

params = {'max_iter' : 25, 'sample_size': 1000 ,'sample_style': 'fast_increasing',\
          'alpha_C' : 10., 'reduce_variance': True}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')


#%% solve with SSNSP (multiple times, VR)

params = {'max_iter' : 25, 'sample_size': f.N ,'sample_style': 'fast_increasing',\
          'alpha_C' : 10., 'reduce_variance': True}
    
K = 10
allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-12, params = params, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k.info)
    
#%% solve with SSNSP (multiple times, no VR)

params1 = params.copy()
params1["reduce_variance"] = False

allP1 = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-12, params = params1, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP1.append(P_k.info)
    
#%% solve with CONSTANT SSNSP

params = {'max_iter' : 10, 'sample_size': f.N, 'sample_style': 'constant', 'alpha_C' : 10.}

P1 = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P1.solve(solver = 'ssnsp')

#%%

all_x = pd.DataFrame(np.vstack((xsol, Q.x, Q1.x, P.x, P1.x, x_sk)).T, columns = ['true', 'saga', 'adagrad', 'spp', 'spp_full', 'scikit'])


#%% plotting
save = False

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True}

Q.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
#Q1.plot_objective(ax = ax, ls = '-.', marker = '>', **kwargs)


#plot_multiple(allP, ax = ax , label = "ssnsp", **kwargs)
#plot_multiple(allP1, ax = ax , label = "ssnsp_noVR", name = "ssnsp (no VR)", **kwargs)

P.plot_objective(ax = ax, **kwargs)
P1.plot_objective(ax = ax, label = " constant", marker = "x", **kwargs)


#ax.set_xlim(-.1,20)
ax.legend(fontsize = 10)
#ax.set_yscale('log')

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)


if save:
    fig.savefig(f'data/plots/exp_runtime/lasso_obj_{int(np.log10(kappa))}.pdf', dpi = 300)


#%%







