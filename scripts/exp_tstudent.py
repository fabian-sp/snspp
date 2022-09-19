"""
@author: Fabian Schaipp
"""

import sys

if len(sys.argv) > 2:
    save = sys.argv[1]
    setup = float(sys.argv[2])
else:
    save = False
    setup = 2

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from snspp.helper.data_generation import tstudent_test
from snspp.solver.opt_problem import problem

from snspp.experiments.experiment_utils import params_tuner,  initialize_solvers, eval_test_set,\
                                                tstudent_loss

from snspp.experiments.container import Experiment

#%% load data

if setup == 2:
    l1 = 0.001
    v = 1.
    poly = 0
    n = 5000; N = 4000; k = 20
    noise = 0.1
    
elif setup == 3:
    l1 = 0.01
    v = 1.
    poly = 0
    n = 10000; N = 1000; k = 20
    noise = 0.1


#%%

f, phi, A, X_train, y_train, X_test, y_test, xsol = tstudent_test(N = N, n = n, k = k, lambda1 = l1, v = v,\
                                                                  noise = noise, poly = poly, kappa = 15., dist = 'ortho')

print("psi(0) = ", f.eval(np.zeros(f.N)))
initialize_solvers(f, phi, A)

#%% parameter setup

if setup == 2:
    params_saga = {'n_epochs' : 50, 'alpha' : 0.00498}
    params_svrg = {'n_epochs' : 70, 'batch_size': 20, 'alpha': 0.19950}
    params_adagrad = {'n_epochs' : 150, 'batch_size': 20, 'alpha': 0.03}   
    params_snspp = {'max_iter' : 320, 'batch_size': 20, 'sample_style': 'constant', 'alpha' : 3., 'reduce_variance': True}

elif setup == 3:
    params_saga = {'n_epochs' : 25, 'alpha' : 0.07730}
    params_svrg = {'n_epochs' : 10, 'batch_size': 10, 'alpha': 2.31905}
    params_adagrad = {'n_epochs' : 300, 'batch_size': 100, 'alpha': 0.06}   
    params_snspp = {'max_iter' : 250, 'batch_size': 10, 'sample_style': 'constant', 'alpha' : 12.5, 'reduce_variance': True}


#%% solve with SAGA

Q = problem(f, phi, A, tol = 1e-6, params = params_saga, verbose = True, measure = True)
Q.solve(solver = 'saga')

print("psi(x_t) = ", f.eval(A@Q.x) + phi.eval(Q.x))

# use the last objective of SAGA as surrogate optimal value / plot only psi(x^k)
psi_star = f.eval(A@Q.x)+phi.eval(Q.x)
#psi_star = 0

#%% solve with SVRG

Q2 = problem(f, phi, A, tol = 1e-6, params = params_svrg, verbose = True, measure = True)
Q2.solve(solver = 'svrg')

print("psi(x_t) = ", f.eval(A@Q2.x) + phi.eval(Q2.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, A, tol = 1e-6, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print("psi(x_t) = ", f.eval(A@Q1.x) + phi.eval(Q1.x))

#%% solve with SNSPP

P = problem(f, phi, A, tol = 1e-6, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

print("psi(x_t) = ", f.eval(A@P.x) + phi.eval(P.x))

#fig = P.plot_subproblem(start = 0)

#%%

###########################################################################
# multiple execution
############################################################################

K = 20

kwargs2 = {"A": X_test, "b": y_test, "v": f.v}
loss = [tstudent_loss]
names = ['test_loss']

Cont = Experiment(name = f'exp_tstudent_N_{N}_n_{n}')

Cont.params = {'saga':params_saga, 'svrg': params_svrg, 'adagrad':params_adagrad, 'snspp':params_snspp}
Cont.psi_star = psi_star


#%% solve with SAGA (multiple times)

allQ = list()
for k in range(K):
    
    Q_k = problem(f, phi, A, tol = 1e-19, params = params_saga, verbose = True, measure = True)
    Q_k.solve(solver = 'saga')
    
    Cont.store(Q_k, k)
    err_k = eval_test_set(X = Q_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q_k.solver, k = k)
    
    allQ.append(Q_k)

#%% solve with ADAGRAD (multiple times)

allQ1 = list()
for k in range(K):
    
    Q1_k = problem(f, phi, A, tol = 1e-19, params = params_adagrad, verbose = True, measure = True)
    Q1_k.solve(solver = 'adagrad')
    
    Cont.store(Q1_k, k)
    err_k = eval_test_set(X = Q1_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q1_k.solver, k = k)
    
    allQ1.append(Q1_k)

#%% solve with SVRG (multiple times)

allQ2 = list()
for k in range(K):
    
    Q2_k = problem(f, phi, A, tol = 1e-19, params = params_svrg, verbose = True, measure = True)
    Q2_k.solve(solver = 'svrg')
    
    Cont.store(Q2_k, k)
    err_k = eval_test_set(X = Q2_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q2_k.solver, k = k)
    
    allQ2.append(Q2_k)
    
#%% solve with SSNSP (multiple times, VR)

allP = list()
for k in range(K):
    
    P_k = problem(f, phi, A, tol = 1e-19, params = params_snspp, verbose = False, measure = True)
    P_k.solve(solver = 'snspp')
    
    Cont.store(P_k, k)
    err_k = eval_test_set(X = P_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = P_k.solver, k = k)
    
    allP.append(P_k)


#%% store

all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['sol', 'spp', 'saga', 'adagrad', 'svrg'])

Cont.save_to_disk(path = '../data/output/')

#%%

###########################################################################
# plotting
############################################################################

if setup == 3:
    xlim = (0, 0.5)
else:
    xlim = (0, 1.5)

#%% plot objective

fig,ax = plt.subplots(figsize = (4.5, 3.5))
kwargs = {"psi_star": psi_star, "log_scale": True, "lw": 1., "markersize": 2.5}

#Q.plot_objective(ax = ax, **kwargs)
#Q1.plot_objective(ax = ax, **kwargs)
#Q2.plot_objective(ax = ax, **kwargs)
#P.plot_objective(ax = ax, **kwargs)

Cont.plot_objective(ax = ax, median = False, **kwargs) 

ax.set_xlim(xlim)
ax.set_ylim(1e-7,1e-1)
ax.legend(fontsize = 10, loc = 'upper right')

fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
if save:
    fig.savefig(f'../data/plots/exp_tstudent/obj_N_{N}_n_{n}.pdf', dpi = 300)

#%% plot error
    
fig,ax = plt.subplots(figsize = (4.5, 3.5))
kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}

Cont.plot_error(error_key = 'test_loss', ax = ax, median = True, ylabel = 'Test loss', **kwargs) 

ax.set_xlim(xlim)
ax.legend(fontsize = 10)
ax.set_yscale('log')

if setup == 3:    
    ax.set_ylim(0.224, 0.26)
else:
    ax.set_ylim(0.2, 0.4)
 
fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)

if save:
    fig.savefig(f'../data/plots/exp_tstudent/error_N_{N}_n_{n}.pdf', dpi = 300)

#%% coeffcient plot

fig,ax = plt.subplots(2, 2, figsize = (7,5))

Q_k.plot_path(ax = ax[0,0], xlabel = False)
Q1_k.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
Q2_k.plot_path(ax = ax[1,0])
P_k.plot_path(ax = ax[1,1], ylabel = False)

for a in ax.ravel():
    a.set_ylim(-2., 2.)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'../data/plots/exp_tstudent/coeff_N_{N}_n_{n}.pdf', dpi = 300)
    
