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

#%% solve with vanilla SGD

params_sgd = {'n_epochs': 600, 'batch_size': 50, 'alpha': 1e-1, 'beta': 0.6, 'style': 'vanilla'}

P = problem(f, phi, tol = 1e-5, params = params_sgd, verbose = True, measure = True)
P.solve(solver = 'sgd')

#P.plot_path()

print("Objective value: ", f.eval(P.x) + phi.eval(P.x))

#%% solve with SNSPP

params = {'max_iter' : 300, 'batch_size': 80, 'sample_style': 'fast_increasing', \
          'alpha' : 4., 'reduce_variance': False}
    
P2 = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)
P2.solve(solver = 'snspp')

#P.plot_path()

print("Objective value: ", f.eval(P2.x) + phi.eval(P2.x))

#%% solve with truncated SGD (COMPOSITE)

params_sgd = {'n_epochs': 600, 'batch_size': 50, 'alpha': 1., 'style': 'polyak'}

P1 = problem(f, phi, tol = 1e-5, params = params_sgd, verbose = True, measure = True)
P1.solve(solver = 'sgd')

#P1.plot_path()

print("Objective value: ", f.eval(P1.x) + phi.eval(P1.x))

#%%
fig, ax = plt.subplots()

P.plot_objective(ax = ax, runtime = False, psi_star = psi_star, log_scale = True, label = "")
P1.plot_objective(ax = ax, runtime = False, psi_star = psi_star, log_scale = True, ls = ':',label = " Polyak")
P2.plot_objective(ax = ax, runtime = False, psi_star = psi_star, log_scale = True)

#%%

all_x = pd.DataFrame(np.vstack((xsol, P.x, P1.x, x_sk)).T, columns = ['true', 'sgd', 'polyak', 'scikit'])


#%%

step_sizes = P1.info['step_sizes']

plt.figure()
plt.plot(P.info['step_sizes'])
plt.plot(P1.info['step_sizes'])
plt.yscale('log')