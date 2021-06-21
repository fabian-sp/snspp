"""
author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from sklearn.linear_model import Lasso, LogisticRegression

from snspp.helper.data_generation import lasso_test, logreg_test, tstudent_test
from snspp.solver.opt_problem import problem

#%% generate data

N = 1000
n = 50
k = 5
l1 = .01

xsol, A, b, f, phi, A_test, b_test = lasso_test(N, n, k, l1, block = False, noise = 0.1, kappa = 10., dist = 'ortho')

#xsol, A, b, f, phi, A_test, b_test = logreg_test(N, n, k, l1, noise = 0.1, kappa = 10., dist = 'ortho')

#x, A, b, f, phi, A_test, b_test = tstudent_test(N, n, k, l1, v = 4, noise = 0.1, poly = 2, kappa = 10., dist = 'ortho')

#%% solve with SSNSP

params = {'max_iter' : 50, 'batch_size': 1000, 'sample_style': 'fast_increasing', \
          'alpha' : 10., 'reduce_variance': False}

P = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

start = time.time()
P.solve(solver = 'snspp')
end = time.time()

print(f"Computing time: {end-start} sec")

P.plot_path()
P.plot_objective()

info = P.info.copy()

#%% compare to scikit

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-6, max_iter = 10000, selection = 'cyclic')

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-5, solver = 'saga', max_iter = 700000, verbose = 1)


start = time.time()
sk.fit(A,b)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

f.eval(x_sk) + phi.eval(x_sk)

#%% compare to SAGA/ADAGRAD

params = {'n_epochs' : 200, 'alpha': 1}

Q = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

start = time.time()
Q.solve(solver = 'svrg')
end = time.time()

print(f"Computing time: {end-start} sec")

Q.plot_path()
Q.plot_objective()

info2 = Q.info.copy()

#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((xsol, P.x, x_sk)).T, columns = ['true', 'spp', 'scikit'])

all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x)).T, columns = ['true', 'spp', 'saga'])

#%% convergence of the xi variables
import seaborn as sns

info = P.info.copy()
#xis = [np.hstack(list(i.values())) for i in info['xi_hist']]
xis = info['xi_hist']

xis = np.vstack(xis)

plt.figure()
sns.heatmap(xis, cmap = 'coolwarm', vmin = -1, vmax = 1)

plt.figure()
#sns.distplot(np.hstack(info['xi_hist'][-1].values()))
sns.distplot(xis[-1,:])


