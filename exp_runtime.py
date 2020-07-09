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

N = 20000
n = 100
k = 10
l1 = .01

#xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = True)
xsol, A, b, f, phi = logreg_test(N, n, k, l1)


#%% solve with SAGA


params = {'n_epochs' : 70}

Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga_fast')


#%% plotting

x = Q.info['runtime'].cumsum()
y = Q.info['objective']

plt.plot(x,y, '-o')

#%% solve with SSNSP

params = {'max_iter' : 5, 'sample_size': 2000, 'alpha_C' : 1., 'n_epochs': 4}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'warm_ssnsp')


x = P.info['runtime'].cumsum()
y = P.info['objective']

plt.plot(x,y, '-o')

#%%

