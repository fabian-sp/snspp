"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LogisticRegression
import time

from ssnsp.helper.data_generation import lasso_test, logreg_test
from ssnsp.solver.opt_problem import problem


#%% generate data

N = 2000
n = 5000
k = 100
l1 = .2

kappa = 1e6

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False, kappa = kappa)
#xsol, A, b, f, phi = logreg_test(N, n, k, l1, noise = .5)

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-9, max_iter = 20000, selection = 'cyclic')
#sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-5, solver = 'saga', max_iter = 700000, verbose = 1)

start = time.time()
sk.fit(A,b)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

print(f.eval(x_sk) +phi.eval(x_sk))


#%% solve with SAGA

params = {'n_epochs' : 300, 'reg': 1e-4}

Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))
#Q.plot_path()


#%% solve with ADAGRAD

params = {'n_epochs' : 100, 'batch_size': 10, 'gamma': .1}

Q1 = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))
#Q1.plot_path()

#%% solve with SSNSP

params = {'max_iter' : 10, 'sample_size': f.N, 'sample_style': 'fast_increasing', 'alpha_C' : 10., 'n_epochs': 5}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')

#P.plot_path()

#%% solve with FULL SSNSP

params = {'max_iter' : 15, 'sample_size': f.N/3, 'sample_style': 'constant', 'alpha_C' : 10.}

P1 = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P1.solve(solver = 'ssnsp')

#P1.plot_path()

#%%
all_x = pd.DataFrame(np.vstack((xsol, Q.x, Q1.x, P.x, P1.x, x_sk)).T, columns = ['true', 'saga', 'adagrad', 'spp', 'spp_full', 'scikit'])


#%% plotting
save = False

fig,ax = plt.subplots(figsize = (6,4))
Q.plot_objective(ax = ax, ls = '--', marker = '<')
Q1.plot_objective(ax = ax, ls = '--', marker = '<')
P.plot_objective(ax = ax)

P1.plot_objective(ax = ax, label = 'ssnsp_constant')

ax.set_yscale('log')
#fig.suptitle('Lasso problem - objective')

if save:
    fig.savefig(f'data/plots/exp_runtime/lasso_obj_{kappa}.png', dpi = 300)


#%%







