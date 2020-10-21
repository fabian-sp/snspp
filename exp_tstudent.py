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


def sample_loss(A_test, b_test, x, v):
    z = A_test@x - b_test
    return 1/A_test.shape[0] * np.log(1+ z**2/v).sum()
    

#%% generate data

N = 5000
n = 3000
k = 100
l1 = 1e-2

xsol, A, b, f, phi, A_test, b_test = tstudent_test(N, n, k, l1, v = 4)

initialize_fast_gradient(f, phi)

#x0 = np.linalg.pinv(f.A)@b
#x0[np.abs(x0) <= 0.1] = 0

print(f.eval(xsol) +phi.eval(xsol))

psi_star = f.eval(xsol) +phi.eval(xsol)
#%% solve with SAGA
params = {'n_epochs' : 200}

Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))

#%% solve with BATCH-SAGA

params = {'n_epochs' : 200}

Q2 = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q2.solve(solver = 'batch saga')

print(f.eval(Q2.x) +phi.eval(Q2.x))

#%% solve with ADAGRAD

#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = None)
opt_gamma = 0.06579332246575682

params = {'n_epochs' : 350, 'batch_size': int(f.N*0.05), 'gamma': opt_gamma}

Q1 = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))
#%% solve with SSNSP

params = {'max_iter' : 120, 'sample_size': 70, 'sample_style': 'constant',\
          'alpha_C' : 8., 'reduce_variance': True}

P = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')


#%% solve with SSNSP (multiple times, VR)

params = {'max_iter' : 10, 'sample_size': 1000 ,'sample_style': 'fast_increasing',\
          'alpha_C' : 10., 'reduce_variance': True}
    
K = 20
allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-9, params = params, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k.info)
 
#%%
all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, Q1.x)).T, columns = ['true', 'spp', 'saga', 'adagrad'])

#all_x = pd.DataFrame(np.vstack((xsol, x0, P.x, Q.x, Q1.x)).T, columns = ['true', 'x0','spp', 'saga', 'adagrad'])

#%%
save = False

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": 0, "log_scale": True}

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

#%% coefficent plot

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
P.plot_path(ax = ax[1,0])
P.plot_path(ax = ax[1,1], mean = True, ylabel = False)

for a in ax.ravel():
    a.set_ylim(-2., 2.)
    
plt.subplots_adjust(hspace = 0.33)


#%%
sample_loss(A_test, b_test, xsol, f.v)

sample_loss(A_test, b_test, Q.x, f.v)

sample_loss(A_test, b_test, Q1.x, f.v)

sample_loss(A_test, b_test, P.x, f.v)
