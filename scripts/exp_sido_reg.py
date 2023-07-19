"""
@author: Fabian Schaipp

This runs the L1-Student-Regression experiment on the Sido0 dataset.
For running this, complete the following steps:

1) Download Sido0 dataset from http://www.causality.inf.ethz.ch/challenge.php?page=datasets#cont
2) Extract the files to the directory ../data/sido0 (relative to the path of this file)
    
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
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from snspp.solver.opt_problem import problem
from snspp.helper.data_generation import get_sido_reg
from snspp.experiments.experiment_utils import params_tuner,  initialize_solvers, eval_test_set,\
                                                tstudent_loss

from snspp.experiments.container import Experiment

l1 = 0.01
f, phi, A, X_train, y_train, X_test, y_test = get_sido_reg(lambda1 = l1, v = 2, k = 50, scale = False, train_size = 0.8)

#%% solve with SAGA
initialize_solvers(f, phi, A)

# compute starting point
refQ = problem(f, phi, A, tol = 1e-20, params = {'n_epochs': 1}, verbose = False, measure = False)
refQ.solve(solver = 'saga')
x0 = refQ.x.copy()


refP = problem(f, phi, A, x0 = x0, tol = 1e-20, params = {'n_epochs': 500}, verbose = True, measure = False)
refP.solve(solver = 'saga')
xsol = refP.x.copy()

psi_star = f.eval(A@xsol) + phi.eval(xsol)
print("Optimal value: ", psi_star)
print("Nonzeros: ", np.count_nonzero(xsol))

#%% params 

params_saga = {'n_epochs' : 40, 'alpha': 0.004}
    
params_svrg = {'n_epochs' : 100, 'batch_size': 10, 'alpha': 0.0145}

params_adagrad = {'n_epochs' : 200, 'batch_size': 100, 'alpha': 0.015}

params_snspp = {'max_iter' : 500, 'batch_size': 200, 'sample_style': 'constant', 'alpha' : 5.5 ,\
                "reduce_variance": True}
    
# params_tuner(f, phi, A, x0 = x0, solver = "adagrad", alpha_range = np.logspace(-2, 0, 7), batch_range = np.array([20, 100, 200]), n_iter = 100)    
#%% solve with SAGA

Q = problem(f, phi, A, x0 = x0, tol = 1e-20, params = params_saga, verbose = True, measure = True)
Q.solve(solver = 'saga')

print(f.eval(A@Q.x) +phi.eval(Q.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, A, x0 = x0, tol = 1e-20, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print(f.eval(A@Q1.x)+phi.eval(Q1.x))

#%% solve with SVRG

Q2 = problem(f, phi, A, x0 = x0, tol = 1e-20, params = params_svrg, verbose = True, measure = True)
Q2.solve(solver = 'svrg')

print(f.eval(A@Q2.x)+phi.eval(Q2.x))

#%% solve with SNSPP

P = problem(f, phi, A, x0 = x0, tol = 1e-20, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')


#%%

all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['sol', 'spp', 'saga', 'adagrad', 'svrg'])
print("SAGA-SNSPP: ", np.linalg.norm(Q.x-P.x)/np.linalg.norm(P.x))
print("SVRG-SNSPP: ", np.linalg.norm(Q2.x-P.x)/np.linalg.norm(P.x))
print("Adagrad-SNSPP: ", np.linalg.norm(Q1.x-P.x)/np.linalg.norm(P.x))

###########################################################################
# multiple execution
############################################################################

Cont = Experiment(name = 'exp_sido_reg')

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
        
        Q_k = problem(f, phi, A, x0 = x0, tol = 1e-20, params = params_saga, verbose = True, measure = True)
        Q_k.solve(solver = 'saga')
        
        Cont.store(Q_k, k)
        err_k = eval_test_set(X = Q_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q_k.solver, k = k)
        
        allQ.append(Q_k)
    
    #%% solve with ADAGRAD (multiple times)
    
    allQ1 = list()
    for k in range(K):
        
        Q1_k = problem(f, phi, A, x0 = x0, tol = 1e-20, params = params_adagrad, verbose = True, measure = True)
        Q1_k.solve(solver = 'adagrad')
        
        Cont.store(Q1_k, k)
        err_k = eval_test_set(X = Q1_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q1_k.solver, k = k)
        
        allQ1.append(Q1_k)
    
    #%% solve with SVRG (multiple times)
    
    allQ2 = list()
    for k in range(K):
        
        Q2_k = problem(f, phi, A, x0 = x0, tol = 1e-20, params = params_svrg, verbose = True, measure = True)
        Q2_k.solve(solver = 'svrg')
        
        Cont.store(Q2_k, k)
        err_k = eval_test_set(X = Q2_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q2_k.solver, k = k)
        
        allQ2.append(Q2_k)
        
    #%% solve with SSNSP (multiple times, VR)
    
    allP = list()
    for k in range(K):
        
        P_k = problem(f, phi, A, x0 = x0, tol = 1e-20, params = params_snspp, verbose = False, measure = True)
        P_k.solve(solver = 'snspp')
        
        Cont.store(P_k, k)
        err_k = eval_test_set(X = P_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = P_k.solver, k = k)
        
        allP.append(P_k)
    
#%% 

if _run:
    Cont.save_to_disk(path = '../data/output/')

#%%

###########################################################################
# plotting
############################################################################

xlim = (0,8)


if _plot:    
    #%% objective plot    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"psi_star": Cont.psi_star, "log_scale": True, "lw": 1., "markersize": 2.5}
    
    #Q.plot_objective(ax = ax, **kwargs)
    #Q1.plot_objective(ax = ax, **kwargs)
    #Q2.plot_objective(ax = ax, **kwargs)
    #P.plot_objective(ax = ax, **kwargs)
    
    Cont.plot_objective(ax = ax, median = False, **kwargs) 
    
    ax.set_xlim(xlim)
    ax.set_ylim(1e-7,1e-1)
    
    ax.legend(fontsize = 10, loc = 'lower left')
        
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig(f'../data/plots/exp_sido_reg/obj.pdf', dpi = 300)
       
    #%% test loss
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}
    
    Cont.plot_error(error_key = 'test_loss', ax = ax, median = True, ylabel = 'Test loss', **kwargs) 
    
    ax.set_xlim(xlim)
    
    ax.set_ylim(0.074, 0.2)
        
    ax.legend(fontsize = 10)
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig(f'../data/plots/exp_sido_reg/error.pdf', dpi = 300)
    
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
        fig.savefig(f'../data/plots/exp_sido_reg/fnat.pdf', dpi = 300)

    #%% coeffcient plot
    fig,ax = plt.subplots(2, 2, figsize = (7,5))
    
    Q.plot_path(ax = ax[0,0], xlabel = False)
    Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
    Q2.plot_path(ax = ax[1,0])
    P.plot_path(ax = ax[1,1], ylabel = False)
    
    for a in ax.ravel():
        a.set_ylim(-0.5, 1)
        
    plt.subplots_adjust(hspace = 0.33)
    
    if _save:
        fig.savefig(f'../data/plots/exp_sido_reg/coeff.pdf', dpi = 300)