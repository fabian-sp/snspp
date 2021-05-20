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

N = 3000
n = 3000
k = 100
l1 = .01

xsol, A, b, f, phi, A_test, b_test = lasso_test(N, n, k, l1, block = False, kappa = None, noise = 0.1)

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-9, max_iter = 20000, selection = 'cyclic')
sk.fit(A,b)

x_sk = sk.coef_.copy().squeeze()

psi_star = f.eval(x_sk) +phi.eval(x_sk)

#%% solve with SGD

params_sgd = {'n_epochs': 20, 'batch_size': 10, 'gamma': 1, 'truncate': False, 'normed': False}


P = problem(f, phi, tol = 1e-5, params = params_sgd, verbose = True, measure = True)


P.plot_objective(psi_star = psi_star)