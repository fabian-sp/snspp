"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from snspp.helper.data_generation import get_cpusmall
from snspp.solver.opt_problem import problem
from snspp.experiments.experiment_utils import params_tuner, plot_multiple, initialize_solvers, eval_test_set, plot_test_error, plot_multiple_error

#%% load data

l1 = 0.01
v = 1.
poly = 4

f, phi, X_train, y_train, X_test, y_test = get_cpusmall(lambda1 = l1, train_size = .99, v = v, poly = 0, noise = 0.)
n = X_train.shape[1]

initialize_solvers(f, phi)

print("psi(0) = ", f.eval(np.zeros(n)))

#%% parameter setup

params_saga = {'n_epochs' : 200, 'alpha' : 6}
params_svrg = {'n_epochs' : 200, 'batch_size': 10, 'alpha': 38.}
params_adagrad = {'n_epochs' : 500, 'batch_size': 30, 'alpha': 0.004}
params_snspp = {'max_iter' : 300, 'batch_size': 300, 'sample_style': 'fast_increasing', 'alpha' : 1000, 'reduce_variance': True}

#params_tuner(f, phi, solver = "saga", alpha_range = np.linspace(4,8, 10))
#params_tuner(f, phi, solver = "svrg", alpha_range = np.linspace(15, 50, 7), batch_range = np.array([10,20]))
#params_tuner(f, phi, solver = "adagrad", alpha_range = np.logspace(-3,-2, 6), batch_range = np.array([30, 50]))
#params_tuner(f, phi, solver = "snspp", alpha_range = np.logspace(1,3,10), batch_range = np.array([100,500]), n_iter = 100)

#%% solve with SAGA
Q = problem(f, phi, tol = 1e-6, params = params_saga, verbose = True, measure = True)

Q.solve(solver = 'saga')
print("f(x_t) = ", f.eval(Q.x))
print("phi(x_t) = ", phi.eval(Q.x))
print("psi(x_t) = ", f.eval(Q.x) + phi.eval(Q.x))


#%% solve with SVRG

Q2 = problem(f, phi, tol = 1e-6, params = params_svrg, verbose = True, measure = True)
Q2.solve(solver = 'svrg')

print("f(x_t) = ", f.eval(Q2.x))
print("phi(x_t) = ", phi.eval(Q2.x))
print("psi(x_t) = ", f.eval(Q2.x) + phi.eval(Q2.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, tol = 1e-6, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print("f(x_t) = ", f.eval(Q1.x))
print("phi(x_t) = ", phi.eval(Q1.x))
print("psi(x_t) = ", f.eval(Q1.x) + phi.eval(Q1.x))

#%% solve with SNSPP

P = problem(f, phi, tol = 1e-6, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

print("f(x_t) = ", f.eval(P.x))
print("phi(x_t) = ", phi.eval(P.x))
print("psi(x_t) = ", f.eval(P.x) + phi.eval(P.x))


P.plot_objective()
P.plot_path()

#%% plot objective
save = False

# use the last objective of SAGA as surrogate optimal value / plot only psi(x^k)
#psi_star = f.eval(Q.x)+phi.eval(Q.x)
psi_star = 0


fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True, "lw": 0.4, "markersize": 1}

Q.plot_objective(ax = ax, ls = '--', marker = '<',  **kwargs)
#Q1.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
Q2.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
#P.plot_objective(ax = ax, **kwargs)

#plot_multiple(allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
#plot_multiple(allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
#plot_multiple(allQ2, ax = ax , label = "svrg", ls = '--', marker = '>', **kwargs)
#plot_multiple(allP, ax = ax , label = "snspp", **kwargs)

ax.set_xlim(0,10)
#ax.set_ylim(0.19,0.3)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.21,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_cpusmall/obj.pdf', dpi = 300)