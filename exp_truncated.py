"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


from snspp.helper.data_generation import lasso_test
from snspp.solver.opt_problem import problem

#%% generate data

N = 1000
n = 50
k = 5
l1 = .01

xsol, A, b, f, phi, A_test, b_test = lasso_test(N, n, k, l1, block = False, noise = 0.1, kappa = 10., dist = 'ortho')


sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-9, max_iter = 20000, selection = 'cyclic')
sk.fit(A,b)

x_sk = sk.coef_.copy().squeeze()
print(np.abs(x_sk).max())

psi_star = f.eval(x_sk) + phi.eval(x_sk)

#%% solve with SGD

params_sgd = {'n_epochs': 300, 'batch_size': 10, 'alpha': 100, 'truncate': False, 'normed': False}


P = problem(f, phi, tol = 1e-5, params = params_sgd, verbose = True, measure = True)
P.solve(solver = 'sgd')

P.plot_objective(runtime = False, psi_star = psi_star, log_scale = True)
#P.plot_path()

#%% solve with SGD

params_sgd = {'n_epochs': 300, 'batch_size': 10, 'alpha': 100, 'truncate': True, 'normed': False}

P1 = problem(f, phi, tol = 1e-5, params = params_sgd, verbose = True, measure = True)
P1.solve(solver = 'sgd')

P1.plot_objective(runtime = False, psi_star = psi_star, log_scale = True)
#P1.plot_path()

#%%

all_x = pd.DataFrame(np.vstack((xsol, P1.x, x_sk)).T, columns = ['true', 'sgd', 'scikit'])


#%%

step_sizes = P1.info['step_sizes']

plt.figure()
plt.plot(P.info['step_sizes'])
plt.plot(P1.info['step_sizes'])
plt.yscale('log')