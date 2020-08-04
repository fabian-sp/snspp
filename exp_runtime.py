"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LogisticRegression

from ssnsp.helper.data_generation import lasso_test, logreg_test
from ssnsp.solver.opt_problem import problem


#%% generate data

N = 2000
n = 5000
k = 200
l1 = .2

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False, kappa = 1e6)
#xsol, A, b, f, phi = logreg_test(N, n, k, l1, noise = .5)

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-9, max_iter = 20000, selection = 'cyclic')
#sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-5, solver = 'saga', max_iter = 700000, verbose = 1)

sk.fit(A,b)
x_sk = sk.coef_.copy().squeeze()

#%% solve with SAGA

params = {'n_epochs' : 300, 'reg': 1e-4}

Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))
#Q.plot_path()


#%% solve with ADAGRAD

params = {'n_epochs' : 100, 'batch_size': 10}

Q1 = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))
#Q1.plot_path()

#%% solve with SSNSP

params = {'max_iter' : 15, 'sample_size': f.N, 'sample_style': 'increasing', 'alpha_C' : 10., 'n_epochs': 5}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')

#P.plot_path()

#%% solve with FULL SSNSP

params = {'max_iter' : 5, 'sample_size': f.N, 'sample_style': 'constant', 'alpha_C' : 20., 'n_epochs': 5}

P1 = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P1.solve(solver = 'ssnsp')

#P1.plot_path()

#%%
all_x = pd.DataFrame(np.vstack((xsol, Q.x, P.x, P1.x, x_sk)).T, columns = ['true', 'saga', 'spp', 'spp_full', 'scikit'])


#%% plotting
fig,ax = plt.subplots()
Q.plot_objective(ax = ax)
Q1.plot_objective(ax = ax)
P.plot_objective(ax = ax)

P1.plot_objective(ax = ax, label = 'full ssnsp')

ax.set_yscale('log')