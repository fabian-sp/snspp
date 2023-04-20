"""
@author: Fabian Schaipp

This runs the L1-Logistic Regression experiment on the Covertype dataset.
For running this, complete the following steps:

1) Download covtype.binary dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary
2) Save the file under ../data/libsvm (relative to the path of this file)

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
from snspp.helper.data_generation import get_libsvm
from snspp.experiments.experiment_utils import params_tuner,  initialize_solvers, eval_test_set,\
                                                logreg_loss, logreg_accuracy

from snspp.experiments.container import Experiment

from sklearn.linear_model import LogisticRegression


f, phi, A, X_train, y_train, X_test, y_test = get_libsvm(name = 'covtype', lambda1 = 0.005, train_size = 0.8)


print("Regularization parameter lambda:", phi.lambda1)

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-20, \
                        solver = 'saga', max_iter = 200, verbose = 0)

start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

psi_star = f.eval(A@x_sk) + phi.eval(x_sk)
print("psi(x*) = ", psi_star)
initialize_solvers(f, phi, A)

#%% params

params_saga = {'n_epochs' : 20, 'alpha': 0.01, 'measure_freq': 10}

params_svrg = {'n_epochs' : 10, 'batch_size': 50, 'alpha': 0.35, 'measure_freq': 10}

params_adagrad = {'n_epochs' : 100, 'batch_size': 250, 'alpha': 0.1}

params_snspp = {'max_iter' : 200, 'batch_size': 50, 'sample_style': 'constant', 'alpha' : 50.,\
                "reduce_variance": True}

#params_tuner(f, phi, A, solver = "adagrad", batch_range = np.array([50, 250, 500]))

#%% solve with SAGA

Q = problem(f, phi, A, tol = 1e-9, params = params_saga, verbose = True, measure = True)
Q.solve(solver = 'saga')

#%% solve with ADAGRAD

Q1 = problem(f, phi, A, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

#%% solve with SVRG

Q2 = problem(f, phi, A, tol = 1e-9, params = params_svrg, verbose = True, measure = True)
Q2.solve(solver = 'svrg')

#%% solve with SSNSP

P = problem(f, phi, A, tol = 1e-9, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

#%%

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad', 'svrg'])

###########################################################################
# multiple execution
############################################################################

Cont = Experiment(name = 'exp_covtype')

if not _run:
    Cont.load_from_disk(path='../data/output/')

    # adjust for measuring multiple times per epoch
    for k in range(len(Cont.results['saga'])):
        Cont.results['saga'][k]['evaluations'] *= 1/params_saga['measure_freq']

    for k in range(len(Cont.results['svrg'])):
        Cont.results['svrg'][k]['evaluations'] *= 1/params_svrg['measure_freq'] 


else:
    K = 20

    kwargs2 = {"A": X_test, "b": y_test}
    loss = [logreg_loss, logreg_accuracy]
    names = ['test_loss', 'test_accuracy']
        
    Cont.params = {'saga': params_saga, 'svrg': params_svrg, 'adagrad': params_adagrad, 'snspp': params_snspp}
    Cont.psi_star = psi_star

    #%% solve with SAGA (multiple times)
    
    allQ = list()
    for k in range(K):
        
        Q_k = problem(f, phi, A, tol = 1e-20, params = params_saga, verbose = True, measure = True)
        Q_k.solve(solver = 'saga')
        
        Cont.store(Q_k, k)
        err_k = eval_test_set(X = Q_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q_k.solver, k = k)
        
        allQ.append(Q_k)
    
    #%% solve with ADAGRAD (multiple times)
    
    allQ1 = list()
    for k in range(K):
        
        Q1_k = problem(f, phi, A, tol = 1e-20, params = params_adagrad, verbose = True, measure = True)
        Q1_k.solve(solver = 'adagrad')
        
        Cont.store(Q1_k, k)
        err_k = eval_test_set(X = Q1_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q1_k.solver, k = k)
        
        allQ1.append(Q1_k)
    
    #%% solve with SVRG (multiple times)
    
    allQ2 = list()
    for k in range(K):
        
        Q2_k = problem(f, phi, A, tol = 0., params = params_svrg, verbose = True, measure = True)
        Q2_k.solve(solver = 'svrg')
        
        Cont.store(Q2_k, k)
        err_k = eval_test_set(X = Q2_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        Cont.store_by_key(res = err_k, label = Q2_k.solver, k = k)
        
        allQ2.append(Q2_k)
        
    #%% solve with SSNSP (multiple times, VR)
    
    allP = list()
    for k in range(K):
        
        P_k = problem(f, phi, A, tol = 1e-20, params = params_snspp, verbose = False, measure = True)
        P_k.solve(solver = 'snspp')
        
        del P_k.info['xi_hist'] #memory heavy and not needed here
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

xlim = (0, 0.7)

if _plot:

    #%% objective plot
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"psi_star": Cont.psi_star, "log_scale": True, "lw": 1., "markersize": 2.5}
    
    mk_every_dict = {'saga': 10, 'svrg': 10} # mark every epoch/outer iter
    
    # Q.plot_objective(ax = ax, markevery = 10, **kwargs)
    # Q1.plot_objective(ax = ax, **kwargs)
    # Q2.plot_objective(ax = ax, markevery = 10, **kwargs)
    # P.plot_objective(ax = ax, **kwargs)
    
    Cont.plot_objective(ax = ax, runtime = False, median = False, markevery_dict = mk_every_dict, **kwargs) 
    
    ax.set_xlim(0, 6)
    ax.set_ylim(1e-7,1e-1)
    ax.legend(fontsize = 10, loc = 'upper right')
    
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig('../data/plots/exp_covtype/obj2.pdf', dpi = 300)
    
    #%% test loss
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}
    
    Cont.plot_error(error_key = 'test_loss', ax = ax, median = True, ylabel = 'Test loss', markevery_dict = mk_every_dict, **kwargs) 
    
    ax.set_xlim(xlim)
    ax.set_ylim(0.58,)
    ax.legend(fontsize = 10)
    
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig('../data/plots/exp_covtype/error.pdf', dpi = 300)
    
    #%% test accuracy
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}
    
    Cont.plot_error(error_key = 'test_accuracy', ax = ax, median = True, ylabel = 'Test accuracy', markevery_dict = mk_every_dict, **kwargs) 
    
    ax.set_xlim(xlim)
    ax.set_ylim(0.58, )
    ax.legend(fontsize = 10)
    
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig('../data/plots/exp_covtype/accuracy.pdf', dpi = 300)
    
    #%% coeffcient plot
    
    fig,ax = plt.subplots(2, 2, figsize = (7,5))
    
    Q.plot_path(ax = ax[0,0], xlabel = False)
    Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
    Q2.plot_path(ax = ax[1,0])
    P.plot_path(ax = ax[1,1], ylabel = False)
    
    for a in ax.ravel():
        a.set_ylim(-2., 2.5)
        
    plt.subplots_adjust(hspace = 0.33)
    
    if _save:
        fig.savefig('../data/plots/exp_covtype/coeff.pdf', dpi = 300)



