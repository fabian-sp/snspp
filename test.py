"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import Lasso, LogisticRegression

from ssnsp.helper.data_generation import lasso_test, logreg_test
from ssnsp.solver.opt_problem import problem


#%% generate data

N = 10000
n = 1000
k = 200
l1 = .01

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False, kappa = None)

xsol, A, b, f, phi = logreg_test(N, n, k, l1, noise = .1)


#%% solve with SPP
params = {'max_iter' : 10, 'sample_size': 1000, 'sample_style': 'constant', 'alpha_C' : 10.}

params = {'n_epochs' : 5}

P = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

start = time.time()
P.solve(solver = 'ssnsp')
end = time.time()

print(f"Computing time: {end-start} sec")

P.plot_path()
P.plot_objective()
P.plot_samples()

info = P.info.copy()

#%% compare to scikit

sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-6, max_iter = 10000, selection = 'cyclic')

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-5, solver = 'saga', max_iter = 700000, verbose = 1)


start = time.time()
sk.fit(A,b)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

#f.eval(x_sk) + phi.eval(x_sk)

all_x = pd.DataFrame(np.vstack((xsol, P.x, x_sk)).T, columns = ['true', 'spp', 'scikit'])


#%% plot error over iterations

true_x = x_sk.copy()

err_l2 = np.linalg.norm(P.info['iterates'] - x_sk, 2, axis = 1)
err_linf = np.linalg.norm(P.info['iterates'] - x_sk, np.inf, axis = 1)


#(P.info['iterates'] * P.info['step_sizes'][:,np.newaxis])
tmp = P.info['iterates'].cumsum(axis = 0)

scale = (1 / (np.arange(P.info['iterates'].shape[0]) + 1))[:,np.newaxis]
xmean_hist = scale * tmp 

err_l2_mean = np.linalg.norm(xmean_hist - x_sk, 2, axis = 1)




plt.figure()
plt.plot(err_l2)
plt.plot(err_linf)
plt.plot(err_l2_mean)

plt.legend(labels = ['error xk (l2)', 'error xk(linf)', 'error xmean (l2)'])

#%% convergence of the xi variables

info = P.info.copy()
xis = [np.hstack(list(i.values())) for i in info['xi_hist']]
xis = np.vstack(xis)

plt.figure()
sns.heatmap(xis, cmap = 'coolwarm', vmin = -1, vmax = 1)

#%% newton convergence

sub_rsd = P.info['ssn_info']

fig, axs = plt.subplots(5,5)
fig.legend(['residual', 'step_size', 'direction'])

for j in np.arange(25):
    ax = axs.ravel()[j]
    ax.plot(sub_rsd[j]['residual'], 'blue')
    ax2 = ax.twinx()
    ax2.plot(sub_rsd[j]['step_size'], 'orange')
    ax2.plot(sub_rsd[j]['direction'], 'green')
    
    ax.set_yscale('log')
    ax2.set_yticks([])
