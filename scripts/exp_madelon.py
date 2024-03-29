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
from snspp.helper.data_generation import get_poly
from snspp.experiments.experiment_utils import params_tuner, initialize_solvers, logreg_loss, logreg_accuracy

from snspp.experiments.container import Experiment

from sklearn.linear_model import LogisticRegression


f, phi, A, X_train, y_train, _, _ = get_poly(name = 'madelon', lambda1 = 0.02, train_size=None,\
                                                     scale=True, poly=2)

print("Regularization parameter lambda:", phi.lambda1)

#%% solve with scikit (SAGA)

# sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-9, \
#                         solver = 'saga', max_iter = 300, verbose = 1)


# start = time.time()
# sk.fit(X_train, y_train)
# end = time.time()
# print(f"Computing time: {end-start} sec")

# x_sk = sk.coef_.copy().squeeze()
# psi_star = f.eval(A@x_sk) + phi.eval(x_sk)

psi_star = 0.5682952130847875 # used from stability experiment
print("psi(x*) = ", psi_star)
print("Shape of A: ", A.shape)

initialize_solvers(f, phi, A)

#%% params

params_saga = {'n_epochs': 600, 'alpha': 0.001, 'batch_size': 10}

params_svrg = {'n_epochs': 120, 'batch_size': 20, 'alpha': 0.005}

params_adagrad = {'n_epochs' : 600, 'batch_size': 100, 'alpha': 0.00075}

params_snspp = {'max_iter' : 700, 'batch_size': 100, 'sample_style': 'constant', \
                'alpha' : 0.08, 'reduce_variance': True}
        
#params_tuner(f, phi, A, solver = "adagrad", batch_range = np.array([100, 1000, 3000]))

#%% solve with SAGA

Q = problem(f, phi, A, tol = 1e-19, params = params_saga, verbose = True, measure = True)
Q.solve(solver = 'saga')

print(f.eval(A@Q.x)+phi.eval(Q.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, A, tol = 1e-19, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print(f.eval(A@Q1.x)+phi.eval(Q1.x))

#%% solve with SVRG

Q2 = problem(f, phi, A, tol = 1e-19, params = params_svrg, verbose = True, measure = True)
Q2.solve(solver = 'svrg')

print(f.eval(A@Q2.x)+phi.eval(Q2.x))

#%% solve with SSNSP

P = problem(f, phi, A, tol = 1e-19, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

print(f.eval(A@P.x)+phi.eval(P.x))

#%%

#all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad', 'svrg'])

###########################################################################
# multiple execution
############################################################################

K = 20

#kwargs2 = {"A": X_test, "b": y_test}
#loss = [logreg_loss, logreg_accuracy]
#names = ['test_loss', 'test_accuracy']

Cont = Experiment(name = 'exp_madelon')

if not _run:
    Cont.load_from_disk(path='../data/output/')
else:
    Cont.params = {'saga':params_saga, 'svrg': params_svrg, 'adagrad': params_adagrad, 'snspp':params_snspp}   
    Cont.psi_star = psi_star


    #%% solve with SAGA (multiple times)
    
    #allQ = list()
    for k in range(K):
        
        Q_k = problem(f, phi, A, tol = 1e-19, params = params_saga, verbose = True, measure = True)
        Q_k.solve(solver = 'saga')
        
        Cont.store(Q_k, k)
        #err_k = eval_test_set(X = Q_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        #Cont.store_by_key(res = err_k, label = Q_k.solver, k = k)
        
        #allQ.append(Q_k)
    
    #%% solve with ADAGRAD (multiple times)
    
    allQ1 = list()
    for k in range(K):
        
        Q1_k = problem(f, phi, A, tol = 1e-19, params = params_adagrad, verbose = True, measure = True)
        Q1_k.solve(solver = 'adagrad')
        
        Cont.store(Q1_k, k)
    #     err_k = eval_test_set(X = Q1_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    #     Cont.store_by_key(res = err_k, label = Q1_k.solver, k = k)
        
    #     allQ1.append(Q1_k)
    
    #%% solve with SVRG (multiple times)
    
    #allQ2 = list()
    for k in range(K):
        
        Q2_k = problem(f, phi, A, tol = 1e-19, params = params_svrg, verbose = True, measure = True)
        Q2_k.solve(solver = 'svrg')
        
        Cont.store(Q2_k, k)
        #err_k = eval_test_set(X = Q2_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        #Cont.store_by_key(res = err_k, label = Q2_k.solver, k = k)
        
        #allQ2.append(Q2_k)
        
    #%% solve with SSNSP (multiple times, VR)
    
    #allP = list()
    for k in range(K):
        
        P_k = problem(f, phi, A, tol = 1e-19, params = params_snspp, verbose = False, measure = True)
        P_k.solve(solver = 'snspp')
        
        Cont.store(P_k, k)
        #err_k = eval_test_set(X = P_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
        #Cont.store_by_key(res = err_k, label = P_k.solver, k = k)
        
        #allP.append(P_k)

#%% 

if _run:
    Cont.save_to_disk(path = '../data/output/')

#%%

###########################################################################
# plotting
############################################################################

xlim = (0, 300)

if _plot:
    #%% objective plot
    
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
        fig.savefig('../data/plots/exp_madelon/obj.pdf', dpi = 300)
    
    
    #%% fnat
    
    fig,ax = plt.subplots(figsize = (4.5, 3.5))
    kwargs = {"log_scale": True, "lw": 1., "markersize": 2.5}
    
    Cont.plot_fnat(ax = ax, median = False, **kwargs) 
    
    ax.set_xlim(xlim)
    ax.set_ylim(1e-3,1e-1)
    ax.legend(fontsize = 10)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.96,bottom=0.14,right=0.965,hspace=0.2,wspace=0.2)
    
    if _save:
        fig.savefig(f'../data/plots/exp_madelon/fnat.pdf', dpi = 300)


