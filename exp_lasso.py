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
from ssnsp.experiments.experiment_utils import plot_multiple, initialize_fast_gradient, adagrad_step_size_tuner


#%% generate data

N = 6000
n = 10000
k = 100
l1 = .2

kappa = 1e6

xsol, A, b, f, phi, A_test, b_test = lasso_test(N, n, k, l1, block = False, kappa = kappa, scale = 10.)

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-9, max_iter = 20000, selection = 'cyclic')

start = time.time()
sk.fit(A,b)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

psi_star = f.eval(x_sk) +phi.eval(x_sk)
print(psi_star)
initialize_fast_gradient(f, phi)


#%% solve with SAGA
params_saga = {'n_epochs' : 200, 'gamma' : 4.}
Q = problem(f, phi, tol = 1e-9, params = params_saga, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))


#%% solve with ADAGRAD
params_adagrad = {'n_epochs' : 300, 'batch_size': int(f.N*0.05), 'gamma': 0.002}

#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = None)
opt_gamma = .01

params = {'n_epochs' : 200, 'batch_size': int(f.N*0.05), 'gamma': opt_gamma}

Q1 = problem(f, phi, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))

#%% solve with SSNSP

params_ssnsp = {'max_iter' : 100, 'sample_size': 100, 'sample_style': 'constant', 'alpha_C' : 0.1, 'reduce_variance': True}

P = problem(f, phi, tol = 1e-7, params = params_ssnsp, verbose = True, measure = True)

P.solve(solver = 'ssnsp')


#%% solve with SSNSP (multiple times, VR)

K = 10
allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-12, params = params_ssnsp, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k)
    
#%% solve with SSNSP (multiple times, no VR)

params1 = params_ssnsp.copy()
params1["reduce_variance"] = False

allP1 = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-12, params = params1, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP1.append(P_k)
    
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
Q1.plot_objective(ax = ax, ls = '-.', marker = '>', **kwargs)


plot_multiple(allP, ax = ax , label = "ssnsp", **kwargs)
#plot_multiple(allP1, ax = ax , label = "ssnsp_noVR", name = "ssnsp (no VR)", **kwargs)

#P.plot_objective(ax = ax, **kwargs)
#P1.plot_objective(ax = ax, label = " constant", marker = "x", **kwargs)


#ax.set_xlim(-.1,20)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_lasso/lasso_obj_{int(np.log10(kappa))}.pdf', dpi = 300)


#%%







