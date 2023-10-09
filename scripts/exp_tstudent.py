"""
@author: Fabian Schaipp
"""

import sys
    
if len(sys.argv) > 1:
    _save = bool(int(sys.argv[1]))
    _run = bool(int(sys.argv[2]))
    _plot = bool(int(sys.argv[3]))
    setup = int(sys.argv[4])
else:
    _save = False
    _run = True
    _plot = True
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

if setup == 1:
    l1 = 0.001
    v = 2.
    poly = 0
    n = 5000; N = 4000; k = 20
    noise = 0.1
    
elif setup == 2:
    l1 = 0.001
    v = 1.
    poly = 0
    n = 5000; N = 4000; k = 20
    noise = 0.1

elif setup == 4:
    l1 = 0.001
    v = 0.5
    poly = 0
    n = 5000; N = 4000; k = 20
    noise = 0.1

#%%

f, phi, A, X_train, y_train, X_test, y_test, xsol = tstudent_test(N = N, n = n, k = k, lambda1 = l1, v = v,\
                                                                  noise = noise, poly = poly, kappa = 15., dist = 'ortho')

print("psi(0) = ", f.eval(np.zeros(f.N)))
initialize_solvers(f, phi, A)

#%% parameter setup

if setup == 1:
    #params_saga = {'n_epochs' : 50, 'alpha' : 0.015} # best setting with b=1
    params_saga = {'n_epochs' : 50, 'batch_size': 20, 'alpha' : 0.2}
    params_svrg = {'n_epochs' : 20, 'batch_size': 20, 'alpha': 0.4}
    params_adagrad = {'n_epochs' : 150, 'batch_size': 40, 'alpha': 0.032}   
    params_snspp = {'max_iter' : 150, 'batch_size': 20, 'sample_style': 'constant', 'alpha' : 7., 'reduce_variance': True}

elif setup == 2:
    #params_saga = {'n_epochs' : 50, 'alpha' : 0.0049} # best setting with b=1
    params_saga = {'n_epochs' : 50, 'batch_size': 4, 'alpha' : 0.015}
    params_svrg = {'n_epochs' : 70, 'batch_size': 20, 'alpha': 0.199}
    params_adagrad = {'n_epochs' : 150, 'batch_size': 20, 'alpha': 0.03}   
    params_snspp = {'max_iter' : 320, 'batch_size': 20, 'sample_style': 'constant', 'alpha' : 3., 'reduce_variance': True}

elif setup == 4:
    params_saga = {'n_epochs' : 100, 'alpha' : 0.0025}
    params_svrg = {'n_epochs' : 100, 'batch_size': 40, 'alpha': 0.14}
    params_adagrad = {'n_epochs' : 300, 'batch_size': 40, 'alpha': 0.0316}   
    params_snspp = {'max_iter' : 800, 'batch_size': 20, 'sample_style': 'constant', 'alpha' : 1.05, 'reduce_variance': True}

#params_tuner(f, phi, A, solver = "adagrad", batch_range = np.array([20,40,80]))

#%% solve with SAGA

Q = problem(f, phi, A, tol = 1e-16, params = params_saga, verbose = True, measure = True)
Q.solve(solver = 'saga')

print("psi(x_t) = ", f.eval(A@Q.x) + phi.eval(Q.x))

# use the last objective of SAGA as surrogate optimal value / plot only psi(x^k)
psi_star = f.eval(A@Q.x)+phi.eval(Q.x)

#%% solve with SVRG

Q2 = problem(f, phi, A, tol = 1e-16, params = params_svrg, verbose = True, measure = True)
Q2.solve(solver = 'svrg')

print("psi(x_t) = ", f.eval(A@Q2.x) + phi.eval(Q2.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, A, tol = 1e-16, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print("psi(x_t) = ", f.eval(A@Q1.x) + phi.eval(Q1.x))

#%% solve with SNSPP

P = problem(f, phi, A, tol = 1e-16, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

print("psi(x_t) = ", f.eval(A@P.x) + phi.eval(P.x))

#fig = P.plot_subproblem(start = 0)

#%%

all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['sol', 'spp', 'saga', 'adagrad', 'svrg'])


print("SAGA-SNSPP: ", np.linalg.norm(Q.x-P.x)/np.linalg.norm(P.x))
print("SVRG-SNSPP: ", np.linalg.norm(Q2.x-P.x)/np.linalg.norm(P.x))
print("Adagrad-SNSPP: ", np.linalg.norm(Q1.x-P.x)/np.linalg.norm(P.x))

###########################################################################
# multiple execution
############################################################################

Cont = Experiment(name = f'exp_tstudent_setup{setup}')

if not _run:
    Cont.load_from_disk(path='../data/output/')
else:
    K = 20
    
    kwargs2 = {"A": X_test, "b": y_test, "v": f.v}
    loss = [tstudent_loss]
    names = ['test_loss']
       
    Cont.params = {'saga':params_saga, 'svrg': params_svrg, 'adagrad':params_adagrad, 'snspp':params_snspp}
    Cont.psi_star = psi_star
       
    #%% solve with SAGA (multiple times)
    
    allQ = list()
    for k in range(K):
        
        Q_k = problem(f, phi, A, tol = 1e-30, params = params_saga, verbose = True, measure = True)
        Q_k.solve(solver = 'saga')
        
        Cont.store(Q_k, k)
        err_k = eval_test_set(X = Q_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q_k.solver, k = k)
        
        allQ.append(Q_k)
    
    #%% solve with ADAGRAD (multiple times)
    
    allQ1 = list()
    for k in range(K):
        
        Q1_k = problem(f, phi, A, tol = 1e-30, params = params_adagrad, verbose = True, measure = True)
        Q1_k.solve(solver = 'adagrad')
        
        Cont.store(Q1_k, k)
        err_k = eval_test_set(X = Q1_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q1_k.solver, k = k)
        
        allQ1.append(Q1_k)
    
    #%% solve with SVRG (multiple times)
    
    allQ2 = list()
    for k in range(K):
        
        Q2_k = problem(f, phi, A, tol = 1e-30, params = params_svrg, verbose = True, measure = True)
        Q2_k.solve(solver = 'svrg')
        
        Cont.store(Q2_k, k)
        err_k = eval_test_set(X = Q2_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q2_k.solver, k = k)
        
        allQ2.append(Q2_k)
        
    #%% solve with SSNSP (multiple times, VR)
    
    allP = list()
    for k in range(K):
        
        P_k = problem(f, phi, A, tol = 1e-30, params = params_snspp, verbose = False, measure = True)
        P_k.solve(solver = 'snspp')
        
        Cont.store(P_k, k)
        err_k = eval_test_set(X = P_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = P_k.solver, k = k)
        
        allP.append(P_k)

#%% store

if _run:
    Cont.save_to_disk(path = '../data/output/')

#%%

###########################################################################
# plotting
############################################################################

if setup == 1:
    xlim = (0, 0.7)
elif setup == 2:
   xlim = (0, 1.5)
elif setup == 4:
    xlim = (0, 3.5)
   
if _plot:
    #%% plot objective
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"psi_star": Cont.psi_star, "log_scale": True, "lw": 1., "markersize": 2.5}
    
    # Q.plot_objective(ax = ax, **kwargs)
    # Q1.plot_objective(ax = ax, **kwargs)
    # Q2.plot_objective(ax = ax, **kwargs)
    # P.plot_objective(ax = ax, **kwargs)
    
    Cont.plot_objective(ax = ax, median = False, **kwargs) 
    
    ax.set_xlim(xlim)
    ax.set_ylim(1e-7,1e-1)
    ax.legend(fontsize = 10, loc = 'upper right')
    
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    if _save:
        fig.savefig(f'../data/plots/exp_tstudent/setup{setup}/obj.pdf', dpi = 300)
    
    #%% plot error
        
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}
    
    Cont.plot_error(error_key = 'test_loss', ax = ax, median = True, ylabel = 'Test loss', **kwargs) 
    
    ax.set_xlim(xlim)
    ax.legend(fontsize = 10)
    #ax.set_yscale('log')
    
    if setup ==2:
        ax.set_ylim(0.2, 0.4)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.96,bottom=0.14,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig(f'../data/plots/exp_tstudent/setup{setup}/error.pdf', dpi = 300)
    
    #%% plot fnat
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": True, "lw": 1., "markersize": 2.5}
    
    Cont.plot_fnat(ax = ax, median = False, **kwargs) 
    
    ax.set_xlim(xlim)
    ax.set_ylim(1e-7,1e-1)
    ax.legend(fontsize = 10)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.96,bottom=0.14,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig(f'../data/plots/exp_tstudent/setup{setup}/fnat.pdf', dpi = 300)

    #%% coeffcient plot
    
    fig,ax = plt.subplots(2, 2, figsize = (7,5))
    
    Q.plot_path(ax = ax[0,0], xlabel = False)
    Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
    Q2.plot_path(ax = ax[1,0])
    P.plot_path(ax = ax[1,1], ylabel = False)
    
    for a in ax.ravel():
        a.set_ylim(-2., 2.)
        
    plt.subplots_adjust(hspace = 0.33)
    
    if _save:
        fig.savefig(f'../data/plots/exp_tstudent/setup{setup}/coeff.pdf', dpi = 300)
    
