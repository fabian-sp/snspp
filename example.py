"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from sklearn.linear_model import Lasso, LogisticRegression

from snspp.helper.data_generation import logreg_test, get_libsvm
from snspp.solver.opt_problem import problem
from snspp.helper.regz import Zero

#%% generate data

N = 1000 # number of samples
n = 50 # dimension
k = 5 # oracle nonzero elements
l1 = .01 # l1 penalty

f, phi, A, X_train, y_train, _, _, beta = logreg_test(N, n, k, l1, noise = 0.1, kappa = 10., dist = 'ortho')
f, phi, A, X_train, y_train, _, _ = get_libsvm(name = "news20", lambda1 = 1e-5, train_size = .8, path_prefix = '')

# for unregularized case:
#phi = Zero()

#%% solve with SSNSP (run twice to compile numba)

params = {'max_iter' : 50, 'batch_size': 250, 'sample_style': 'constant', \
          'alpha' : 2500., 'reduce_variance': True, 'vr_skip': 1}

P = problem(f, phi, A, tol = 1e-5, params = params, verbose = True, measure = True)

P.solve(solver = 'snspp')

P.plot_path()
P.plot_objective()

info = P.info.copy()

#%% solve with SAGA (run twice to compile numba)

params = {'n_epochs' : 10, 'alpha': 1e-3}

Q = problem(f, phi, A, tol = 1e-5, params = params, verbose = True, measure = True)
Q.solve(solver = 'saga')

Q.plot_path()
Q.plot_objective()

info2 = Q.info.copy()

#%% solve with SVRG (run twice to compile numba)

params = {'n_epochs' : 10, 'alpha': 4., 'batch_size': 300}

Q = problem(f, phi, A, tol = 1e-50, params = params, verbose = True, measure = True)
Q.solve(solver = 'tick-svrg')

Q.plot_path()
Q.plot_objective()

info2 = Q.info.copy()

#%% compare to scikit

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-10, solver = 'saga', max_iter = 10, verbose = 1)
sk.fit(X_train, y_train)

x_sk = sk.coef_.copy().squeeze()

f.eval(A@x_sk) + phi.eval(x_sk)

#%% compare solutions

all_x = pd.DataFrame(np.vstack((beta, P.x, Q.x, x_sk)).T, columns = ['true', 'spp', 'saga', 'scikit'])


#%% convergence of the xi variables
import seaborn as sns

info = P.info.copy()
xis = info['xi_hist']

xis = np.vstack(xis)

plt.figure()
sns.heatmap(xis, cmap = 'coolwarm', vmin = -1, vmax = 1)

plt.figure()
sns.distplot(xis[-1,:])


