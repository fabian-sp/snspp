"""
@author: Fabian Schaipp

This runs the L1-Logistic Regression experiment on the Sido0 dataset.
For running this, complete the following steps:

1) Download Sido0 dataset from http://www.causality.inf.ethz.ch/challenge.php?page=datasets#cont
2) Extract the files to the directory ../data/sido0 (relative to the path of this file)
    
"""
import sys
    
if len(sys.argv) > 1:
    _save = bool(int(sys.argv[1]))
    _run = bool(int(sys.argv[2]))
    _plot = bool(int(sys.argv[3]))
    l1 = float(sys.argv[4])
else:
    _save = False
    _run = True
    _plot = True
    l1 = 1e-2
    
#%%
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from snspp.solver.opt_problem import problem
from snspp.helper.data_generation import get_sido
from snspp.experiments.experiment_utils import params_tuner,  initialize_solvers, eval_test_set,\
                                                logreg_loss, logreg_accuracy

from snspp.experiments.container import Experiment


from sklearn.linear_model import LogisticRegression


f, phi, A, X_train, y_train, X_test, y_test = get_sido(lambda1 = l1, scale = False)

print("Regularization parameter lambda:", phi.lambda1)

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                        solver = 'saga', max_iter = 150, verbose = 0)

start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()
#(np.sign(predict(X_test, x_sk)) == np.sign(y_test)).sum() / len(y_test)

psi_star = f.eval(A@x_sk) + phi.eval(x_sk)
print("l1, psi(x*) = ", l1, psi_star)

initialize_solvers(f, phi, A)

# compute starting point
sk0 = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                         solver = 'saga', max_iter = 1, verbose = 0).fit(X_train,y_train)
x0 = sk0.coef_.squeeze()
print("x0 max", x0.max())

#%% params 

if l1 == 1e-3:
    params_saga = {'n_epochs' : 30, 'alpha': 0.0102}
    
    params_svrg = {'n_epochs' : 30, 'batch_size': 50, 'alpha': 0.2702}
    
    params_adagrad = {'n_epochs' : 60, 'batch_size': 20, 'alpha': 0.008}
    
    params_snspp = {'max_iter' : 320, 'batch_size': 120, 'sample_style': 'constant', 'alpha' : 2.5 ,\
                    "reduce_variance": True}
        
    # params_tuner(f, phi, A, solver = "adagrad", batch_range = np.array([20, 200, 500]), x0 = x0)
    # params_tuner(f, phi, A, solver = "snspp", alpha_range = np.linspace(1., 4., 8), batch_range = np.array([60, 120]), n_iter = 200, x0 = x0)

elif l1 == 1e-2:
    
    #params_saga = {'n_epochs' : 30, 'alpha': 0.0140} # best setting with b=1
    params_saga = {'n_epochs' : 30, 'alpha': 0.2, 'batch_size': 10}
    
    params_svrg = {'n_epochs' : 30, 'batch_size': 50, 'alpha': 0.733}
    
    params_adagrad = {'n_epochs' : 60, 'batch_size': 200, 'alpha': 0.015}
    
    params_snspp = {'max_iter' : 400, 'batch_size': 50, 'sample_style': 'constant', 'alpha' : 30.,\
                    "reduce_variance": True}

    # params_tuner(f, phi, A, solver = "adagrad", batch_range = np.array([20, 200, 500]), x0 = x0)
    # params_tuner(f, phi, A, solver = "snspp", alpha_range = np.linspace(5, 15, 8), batch_range = np.array([50, 100]), n_iter = 150, x0 = x0)
    
else:
    raise KeyError("Parameters not tuned for this value of l1.")
    
#%% solve with SAGA

Q = problem(f, phi, A, x0 = x0, tol = 1e-9, params = params_saga, verbose = True, measure = True)
Q.solve(solver = 'saga')

print(f.eval(A@Q.x) +phi.eval(Q.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, A, x0 = x0, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print(f.eval(A@Q1.x)+phi.eval(Q1.x))

#%% solve with SVRG

Q2 = problem(f, phi, A, x0 = x0, tol = 1e-9, params = params_svrg, verbose = True, measure = True)
Q2.solve(solver = 'svrg')

print(f.eval(A@Q2.x)+phi.eval(Q2.x))

#%% solve with SNSPP

P = problem(f, phi, A, x0 = x0, tol = 1e-9, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')


#%%

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad', 'svrg'])

###########################################################################
# multiple execution
############################################################################

Cont = Experiment(name = 'exp_sido')

if not _run:
    Cont.load_from_disk(path='../data/output/', path_suffix = f'_l1_{l1}')
else:   
    K = 20
    
    kwargs2 = {"A": X_test, "b": y_test}
    loss = [logreg_loss, logreg_accuracy]
    names = ['test_loss', 'test_accuracy']
    
    Cont.params = {'saga':params_saga, 'svrg': params_svrg, 'adagrad':params_adagrad, 'snspp':params_snspp}
    Cont.psi_star = psi_star
    
    #%% solve with SAGA (multiple times)
    
    allQ = list()
    for k in range(K):
        
        Q_k = problem(f, phi, A, x0 = x0, tol = 1e-9, params = params_saga, verbose = True, measure = True)
        Q_k.solve(solver = 'saga')
        
        Cont.store(Q_k, k)
        err_k = eval_test_set(X = Q_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q_k.solver, k = k)
        
        allQ.append(Q_k)
    
    #%% solve with ADAGRAD (multiple times)
    
    allQ1 = list()
    for k in range(K):
        
        Q1_k = problem(f, phi, A, x0 = x0, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)
        Q1_k.solve(solver = 'adagrad')
        
        Cont.store(Q1_k, k)
        err_k = eval_test_set(X = Q1_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q1_k.solver, k = k)
        
        allQ1.append(Q1_k)
    
    #%% solve with SVRG (multiple times)
    
    allQ2 = list()
    for k in range(K):
        
        Q2_k = problem(f, phi, A, x0 = x0, tol = 1e-9, params = params_svrg, verbose = True, measure = True)
        Q2_k.solve(solver = 'svrg')
        
        Cont.store(Q2_k, k)
        err_k = eval_test_set(X = Q2_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q2_k.solver, k = k)
        
        allQ2.append(Q2_k)
        
    #%% solve with SSNSP (multiple times, VR)
    
    allP = list()
    for k in range(K):
        
        P_k = problem(f, phi, A, x0 = x0, tol = 1e-9, params = params_snspp, verbose = False, measure = True)
        P_k.solve(solver = 'snspp')
        
        Cont.store(P_k, k)
        err_k = eval_test_set(X = P_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = P_k.solver, k = k)
        
        allP.append(P_k)
    
#%% 

if _run:
    Cont.save_to_disk(path = '../data/output/', path_suffix = f'_l1_{l1}')

#%%

###########################################################################
# plotting
############################################################################

if l1 == 0.01:
    xlim = (0,3)
else:
    xlim = (0,3)


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
    
    if l1 == 0.01:
        ax.legend(fontsize = 10, loc = 'upper right')
    else:
        ax.legend(fontsize = 10, loc = 'lower left')
        
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig(f'../data/plots/exp_sido/l1_{l1}/obj.pdf', dpi = 300)
       
    #%% test loss
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}
    
    Cont.plot_error(error_key = 'test_loss', ax = ax, median = True, ylabel = 'Test loss', **kwargs) 
    
    ax.set_xlim(xlim)
    
    if l1 == 0.01:
        ax.set_ylim(0.104, 0.11)
    else:
        ax.set_ylim(0.085, 0.095)
        
    ax.legend(fontsize = 10)
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig(f'../data/plots/exp_sido/l1_{l1}/error.pdf', dpi = 300)
    
    #%% test accuracy
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}
    
    Cont.plot_error(error_key = 'test_accuracy', ax = ax, median = True, ylabel = 'Test accuracy', **kwargs) 
    
    ax.set_xlim(xlim)
    ax.set_ylim(0.6, 1.)
    ax.legend(fontsize = 10)
    
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig(f'../data/plots/exp_sido/l1_{l1}/accuracy.pdf', dpi = 300)
        
    #%% coeffcient plot
    fig,ax = plt.subplots(2, 2, figsize = (7,5))
    
    Q.plot_path(ax = ax[0,0], xlabel = False)
    Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
    Q2.plot_path(ax = ax[1,0])
    P.plot_path(ax = ax[1,1], ylabel = False)
    
    for a in ax.ravel():
        a.set_ylim(-1.5, 1.5)
        
    plt.subplots_adjust(hspace = 0.33)
    
    if _save:
        fig.savefig(f'../data/plots/exp_sido/l1_{l1}/coeff.pdf', dpi = 300)