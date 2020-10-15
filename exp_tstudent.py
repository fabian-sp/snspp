"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from ssnsp.helper.data_generation import tstudent_test
from ssnsp.solver.opt_problem import problem
from ssnsp.experiments.experiment_utils import plot_multiple, initialize_fast_gradient, adagrad_step_size_tuner


#%% generate data

N = 5000
n = 1000
k = 100
l1 = .01

xsol, A, b, f, phi = tstudent_test(N, n, k, l1, v = 4)

initialize_fast_gradient(f, phi)

psi_star = 0.7387611791660234
#%% solve with SAGA
params = {'n_epochs' : 100}

Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))

#%% solve with BATCH-SAGA

params = {'n_epochs' : 100}

Q2 = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q2.solve(solver = 'batch saga')

print(f.eval(Q2.x) +phi.eval(Q2.x))

#%% solve with ADAGRAD

#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = None)
opt_gamma = 0.06579332246575682

params = {'n_epochs' : 200, 'batch_size': int(f.N*0.05), 'gamma': opt_gamma}

Q1 = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))
#%% solve with SSNSP

params = {'max_iter' : 100, 'sample_size': 150 ,'sample_style': 'constant',\
          'alpha_C' : 6., 'reduce_variance': True}

P = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')


#%% solve with SSNSP (multiple times, VR)

params = {'max_iter' : 100, 'sample_size': 100 ,'sample_style': 'fast_increasing',\
          'alpha_C' : 5., 'reduce_variance': True}
    
K = 20
allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-9, params = params, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k.info)
  
#%%
save = False

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True}

#Q.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
Q1.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
P.plot_objective(ax = ax, **kwargs)

#plot_multiple(allP, ax = ax , label = "ssnsp", **kwargs)

#ax.set_xlim(-.1,20)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)
