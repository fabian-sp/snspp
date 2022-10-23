"""
@author: Fabian Schaipp
"""

import sys
    
if len(sys.argv) > 1:
    _save = bool(int(sys.argv[1]))
    _run = bool(int(sys.argv[2]))
    _plot = bool(int(sys.argv[3]))
else:
    _save = False
    _run = True
    _plot = True

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from snspp.helper.data_generation import get_e2006
from snspp.solver.opt_problem import problem

from snspp.experiments.experiment_utils import params_tuner,  initialize_solvers, eval_test_set,\
                                                tstudent_loss

from snspp.experiments.container import Experiment

#%% load data

f, phi, A, X_train, y_train, X_test, y_test = get_e2006(lambda1 = 1e-6, train_size = 0.8)

initialize_solvers(f, phi, A)

orP = problem(f, phi, A, tol = 1e-20, params = {'n_epochs': 200}, verbose = True, measure = False)
orP.solve(solver = 'saga')
xsol = orP.x.copy()

#%% parameter setup

params_saga = {'n_epochs' : 10, 'alpha' : 0.00025, 'measure_freq': 10}
params_svrg = {'n_epochs' : 10, 'batch_size': 80, 'alpha': 0.08, 'measure_freq': 10}
params_adagrad = {'n_epochs' : 15, 'batch_size': 100, 'alpha': 0.25}   
params_snspp = {'max_iter' : 300, 'batch_size': 80, 'sample_style': 'constant', 'alpha' : 3., 'reduce_variance': True}

#params_tuner(f, phi, A, solver = "adagrad", batch_range = np.array([100, 500]), alpha_range=np.logspace(-3, 1, 6), n_iter=40)

#%% solve with SAGA

Q = problem(f, phi, A, tol = 1e-30, params = params_saga, verbose = True, measure = True)
Q.solve(solver = 'saga')

print("psi(x_t) = ", f.eval(A@Q.x) + phi.eval(Q.x))

# use the last objective of SAGA as surrogate optimal value / plot only psi(x^k)
psi_star = f.eval(A@Q.x)+phi.eval(Q.x)

#%% solve with SVRG

Q2 = problem(f, phi, A, tol = 1e-30, params = params_svrg, verbose = True, measure = True)
Q2.solve(solver = 'svrg')

print("psi(x_t) = ", f.eval(A@Q2.x) + phi.eval(Q2.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, A, tol = 1e-30, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print("psi(x_t) = ", f.eval(A@Q1.x) + phi.eval(Q1.x))

#%% solve with SNSPP

P = problem(f, phi, A, tol = 1e-30, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

print("psi(x_t) = ", f.eval(A@P.x) + phi.eval(P.x))

#fig = P.plot_subproblem(start = 0)

#%%

all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['sol', 'spp', 'saga', 'adagrad', 'svrg'])

###########################################################################
# multiple execution
############################################################################

Cont = Experiment(name = f'exp_eosix')

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

xlim = (0, 60.)
   
if _plot:
    #%% plot objective
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"psi_star": Cont.psi_star, "log_scale": True, "lw": 1., "markersize": 2.5}
    
    # Q.plot_objective(ax = ax, **kwargs)
    # Q1.plot_objective(ax = ax, **kwargs)
    # Q2.plot_objective(ax = ax, **kwargs)
    # P.plot_objective(ax = ax, **kwargs)
    
    mk_every_dict = {'saga': 10, 'svrg': 10} # mark every epoch/outer iter
    Cont.plot_objective(ax = ax, median = False, markevery_dict = mk_every_dict, **kwargs) 
    
    ax.set_xlim(xlim)
    ax.set_ylim(1e-7, 1e-1)
    ax.legend(fontsize = 10, loc = 'upper right')
    
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    if _save:
        fig.savefig('../data/plots/exp_eosix/obj.pdf', dpi = 300)
    
    #%% plot error
        
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}
    
    Cont.plot_error(error_key = 'test_loss', ax = ax, median = True, ylabel = 'Test loss', markevery_dict = mk_every_dict, **kwargs) 
    
    ax.set_xlim(xlim)
    ax.legend(fontsize = 10)
    #ax.set_yscale('log')
    
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.96,bottom=0.14,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig('../data/plots/exp_eosix/error.pdf', dpi = 300)
    
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
        fig.savefig('../data/plots/exp_eosix/coeff.pdf', dpi = 300)
    
