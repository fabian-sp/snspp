"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression


from snspp.helper.data_generation import logreg_test, lasso_test
from snspp.solver.opt_problem import problem

#%% generate data

N = 200
n = 100
k = 10
l1 = .01

xsol, A, b, f, phi, A_test, b_test = lasso_test(N, n, k, l1, noise = 0.1, kappa = 50., dist = 'ortho')


# sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
#                         solver = 'saga', max_iter = 200, verbose = 0)

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-9, max_iter = 20000, selection = 'cyclic')

sk.fit(A, b)

x_sk = sk.coef_.copy().squeeze()

psi_star = f.eval(x_sk) + phi.eval(x_sk)
print("Optimal value: ", psi_star)

#%% solve with truncated SGD

params_sgd = {'n_epochs': 300, 'batch_size': 10, 'alpha': 20., 'truncate': True}

P = problem(f, phi, tol = 1e-5, params = params_sgd, verbose = True, measure = True)
P.solve(solver = 'sgd')


#P.plot_path()

print("Objective value: ", f.eval(P.x) + phi.eval(P.x))

#%% solve with truncated SGD (COMPOSITE)

params_sgd = {'n_epochs': 300, 'batch_size': 10, 'alpha': 20., 'truncate': False}

P1 = problem(f, phi, tol = 1e-5, params = params_sgd, verbose = True, measure = True)
P1.solve(solver = 'sgd')

P1.plot_objective(runtime = False, psi_star = psi_star, log_scale = True)
#P1.plot_path()

print("Objective value: ", f.eval(P1.x) + phi.eval(P1.x))
#%%
fig, ax = plt.subplots()

P.plot_objective(ax = ax, runtime = False, psi_star = psi_star, log_scale = True, marker = '<', label = "truncated")
P1.plot_objective(ax = ax, runtime = False, psi_star = psi_star, log_scale = True, label = "truncated composite")

#%%

all_x = pd.DataFrame(np.vstack((xsol, P1.x, P.x, x_sk)).T, columns = ['true', 'truncated', 'truncated_composite', 'scikit'])


#%%

step_sizes = P1.info['step_sizes']

plt.figure()
plt.plot(P.info['step_sizes'])
plt.plot(P1.info['step_sizes'])
plt.yscale('log')