import time
import matplotlib.pyplot as plt
import numpy as np


from ssnsp.solver.opt_problem import problem
from ssnsp.helper.data_generation import get_mnist


f, phi, X_train, y_train, X_test, y_test = get_mnist()


#%% solve with SAGA

params = {'n_epochs' : 50}

Q = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga_fast')

Q.plot_path()


#%% solve with SSNSP

params = {'max_iter' : 20, 'sample_size': f.N/4, 'sample_style': 'increasing', 'alpha_C' : .05, 'n_epochs': 5}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')

P.plot_path()


#%% solve with FULL SSNSP

params = {'max_iter' : 6, 'sample_size': .7*f.N, 'sample_style': 'constant', 'alpha_C' : 1., 'n_epochs': 5}

P1 = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P1.solve(solver = 'warm_ssnsp')

#%% plotting
fig,ax = plt.subplots()

x = Q.info['runtime'].cumsum()
y = Q.info['objective']

ax.plot(x,y, '-o', label = 'SAGA')

x = P.info['runtime'].cumsum()
y = P.info['objective']

ax.plot(x,y, '-o', label = 'SSNSP')

ax.legend()
ax.set_xlabel('Runtime')
ax.set_ylabel('Objective')

ax.set_yscale('log')
