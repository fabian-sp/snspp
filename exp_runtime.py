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
k = 100
l1 = 1

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False, kappa = 1e3)
#xsol, A, b, f, phi = logreg_test(N, n, k, l1, kappa = None)


#%% solve with SAGA

params = {'n_epochs' : 120}

Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga_fast')

Q.plot_path()

#%% solve with SSNSP

params = {'max_iter' : 15, 'sample_size': 600, 'alpha_C' : 10., 'n_epochs': 5}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'warm_ssnsp')

P.plot_path()

#%% plotting
fig,ax = plt.subplots()

x = Q.info['runtime'].cumsum()
y = Q.info['objective']

ax.plot(x,y, '-o')

x = P.info['runtime'].cumsum()
y = P.info['objective']

ax.plot(x,y, '-o')

ax.set_xlabel('Runtime')
ax.set_ylabel('Objective')