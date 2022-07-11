"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from sklearn.linear_model import Lasso, LogisticRegression

from snspp.helper.data_generation import logreg_test, get_rcv1
from snspp.solver.opt_problem import problem
from snspp.helper.regz import Zero

#%% generate data

N = 1000 # number of samples
n = 50 # dimension
k = 5 # oracle nonzero elements
l1 = .01 # l1 penalty

xsol, A, b, f, phi, A_test, b_test = logreg_test(N, n, k, l1, noise = 0.1, kappa = 10., dist = 'ortho')
#f, phi, A, b, _, _ = get_rcv1(lambda1 = 0.001, train_size = .8, path_prefix = '')

# for unregularized case:
#phi = Zero()

#%% solve with SSNSP (run twice to compile numba)

params = {'max_iter' : 50, 'batch_size': 100, 'sample_style': 'constant', \
          'alpha' : 1., 'reduce_variance': True}

P = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

P.solve(solver = 'snspp')

P.plot_path()
P.plot_objective()

info = P.info.copy()

#%% solve with SAGA (run twice to compile numba)

params = {'n_epochs' : 100, 'alpha': 1.}

Q = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)
Q.solve(solver = 'svrg')

Q.plot_path()
Q.plot_objective()

info2 = Q.info.copy()

#%% compare to scikit

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-5, solver = 'saga', max_iter = 100, verbose = 1)
sk.fit(A,b)

x_sk = sk.coef_.copy().squeeze()

f.eval(x_sk) + phi.eval(x_sk)

#%% compare solutions

all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, x_sk)).T, columns = ['true', 'spp', 'saga', 'scikit'])


#%% convergence of the xi variables
import seaborn as sns

info = P.info.copy()
xis = info['xi_hist']

xis = np.vstack(xis)

plt.figure()
sns.heatmap(xis, cmap = 'coolwarm', vmin = -1, vmax = 1)

plt.figure()
sns.distplot(xis[-1,:])


