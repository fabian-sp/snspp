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

f, phi, X_train, y_train, X_test, y_test = get_cpusmall(lambda1 = l1, train_size = .99, v = v, poly = poly, noise = 0.)
n = X_train.shape[1]

initialize_solvers(f, phi)

print("psi(0) = ", f.eval(np.zeros(n)))

#%% parameter setup

params_saga = {'n_epochs' : 200, 'alpha' : 4.5}
params_svrg = {'n_epochs' : 200, 'batch_size': 10, 'alpha': 38.}
params_adagrad = {'n_epochs' : 500, 'batch_size': 30, 'alpha': 0.004}
params_snspp = {'max_iter' : 200, 'batch_size': 50, 'sample_style': 'fast_increasing', 'alpha' : .7, 'reduce_variance': True}

#params_tuner(f, phi, solver = "saga", alpha_range = np.linspace(4,8, 10))
#params_tuner(f, phi, solver = "svrg", alpha_range = np.linspace(15, 50, 7), batch_range = np.array([10,20]))
#params_tuner(f, phi, solver = "adagrad", alpha_range = np.logspace(-3,-2, 6), batch_range = np.array([30, 50]))
#params_tuner(f, phi, solver = "snspp", alpha_range = np.linspace(0.2,0.8,10), batch_range = np.array([50,70]), n_iter = 100)


#%% solve with SNSPP

P = problem(f, phi, tol = 1e-6, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

print("f(x_t) = ", f.eval(P.x))
print("phi(x_t) = ", phi.eval(P.x))
print("psi(x_t) = ", f.eval(P.x) + phi.eval(P.x))


P.plot_objective()
P.plot_path()
