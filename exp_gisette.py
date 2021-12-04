"""
@author: Fabian Schaipp

This runs the L1-Logistic Regression experiment on the Gisette dataset.
For this to run, download the dataset. Then convert the .txt file to a .npy file by running the function (see snspp.data_generation.py)

X, y = load_from_txt('gisette')

The returned arrays need to be saved in the directory 'data/gisette_X.npy' and 'data/gisette_y.npy' using

np.save('data/gisette_X.npy', X)

"""
import sys

if len(sys.argv) > 1:
    save = sys.argv[1]
else:
    save = False

#%%

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from snspp.solver.opt_problem import problem
from snspp.helper.data_generation import get_gisette
from snspp.experiments.experiment_utils import params_tuner, plot_multiple, initialize_solvers, eval_test_set,\
                                                convert_to_dict, logreg_loss, logreg_accuracy

from snspp.experiments.container import Experiment

from sklearn.linear_model import LogisticRegression


f, phi, X_train, y_train, X_test, y_test = get_gisette(lambda1 = 0.05)


print("Regularization parameter lambda:", phi.lambda1)

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                        solver = 'saga', max_iter = 200, verbose = 0)

start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

psi_star = f.eval(x_sk) + phi.eval(x_sk)
print("psi(x*) = ", psi_star)
initialize_solvers(f, phi)

#%% params

params_saga = {'n_epochs' : 50, 'alpha': 5.5}

params_svrg = {'n_epochs' : 50, 'batch_size': 50, 'alpha': 60.}

params_adagrad = {'n_epochs' : 200, 'batch_size': 240, 'alpha': 0.028}

params_snspp = {'max_iter' : 60, 'batch_size': 400, 'sample_style': 'fast_increasing', 'alpha' : 7.,\
          "reduce_variance": True}

#params_tuner(f, phi, solver = "saga", alpha_range = np.linspace(4,8, 10))
#params_tuner(f, phi, solver = "svrg", alpha_range = np.linspace(50, 80, 8), batch_range = np.array([50]))
#params_tuner(f, phi, solver = "svrg", alpha_range = np.linspace(100, 400, 8), batch_range = np.array([300, 400]))
#params_tuner(f, phi, solver = "adagrad", batch_range = np.array([50, 250, 500]))
#params_tuner(f, phi, solver = "snspp", alpha_range = np.linspace(4,9, 10), batch_range = np.array([200, 400]))

#%% solve with SAGA

Q = problem(f, phi, tol = 1e-9, params = params_saga, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x)+phi.eval(Q1.x))

#%% solve with SVRG

Q2 = problem(f, phi, tol = 1e-9, params = params_svrg, verbose = True, measure = True)

Q2.solve(solver = 'svrg')

print(f.eval(Q2.x)+phi.eval(Q2.x))

#%% solve with SSNSP

P = problem(f, phi, tol = 1e-9, params = params_snspp, verbose = True, measure = True)

P.solve(solver = 'snspp')

#fig = P.plot_subproblem(M=20)
#fig.savefig(f'data/plots/exp_gisette/subprob.pdf', dpi = 300)

#%%

###########################################################################
# multiple execution and plotting
############################################################################

K = 20

kwargs2 = {"A": X_test, "b": y_test}
loss = [logreg_loss, logreg_accuracy]
names = ['test_loss', 'test_accuracy']

Cont = Experiment(name = 'exp_gisette')

#%% solve with SAGA (multiple times)

allQ = list()
for k in range(K):
    
    Q_k = problem(f, phi, tol = 1e-9, params = params_saga, verbose = True, measure = True)
    Q_k.solve(solver = 'saga')
    
    Cont.store(Q_k, k)
    err_k = eval_test_set(X = Q_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q_k.solver, k = k)
    
    allQ.append(Q_k)

#%% solve with ADAGRAD (multiple times)

allQ1 = list()
for k in range(K):
    
    Q1_k = problem(f, phi, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)
    Q1_k.solve(solver = 'adagrad')
    
    Cont.store(Q1_k, k)
    err_k = eval_test_set(X = Q1_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q1_k.solver, k = k)
    
    allQ1.append(Q1_k)

#%% solve with SVRG (multiple times)

allQ2 = list()
for k in range(K):
    
    Q2_k = problem(f, phi, tol = 1e-9, params = params_svrg, verbose = True, measure = True)
    Q2_k.solve(solver = 'svrg')
    
    Cont.store(Q2_k, k)
    err_k = eval_test_set(X = Q2_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q2_k.solver, k = k)
    
    allQ2.append(Q2_k)
    
#%% solve with SSNSP (multiple times, VR)

allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-9, params = params_snspp, verbose = False, measure = True)
    P_k.solve(solver = 'snspp')
    
    Cont.store(P_k, k)
    err_k = eval_test_set(X = P_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = P_k.solver, k = k)
    
    allP.append(P_k)


#%% eval test set loss

# for P in allP: P.info['test_error'] = eval_test_set(X = P.info["iterates"], loss = logreg_loss, **kwargs2)
# for Q in allQ: Q.info['test_error'] = eval_test_set(X = Q.info["iterates"], loss = logreg_loss, **kwargs2)
# for Q in allQ1: Q.info['test_error'] = eval_test_set(X = Q.info["iterates"], loss = logreg_loss, **kwargs2)
# for Q in allQ2: Q.info['test_error'] = eval_test_set(X = Q.info["iterates"], loss = logreg_loss, **kwargs2)
    

#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad', 'svrg'])

Cont.save_to_disk(path = 'data/output/')

#%% objective plot

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True, "lw": 0.4, "markersize": 3}

#Q.plot_objective(ax = ax, ls = '--', **kwargs)
#Q1.plot_objective(ax = ax, ls = '-.', **kwargs)
#Q2.plot_objective(ax = ax, ls = '-.', **kwargs)
#P.plot_objective(ax = ax, **kwargs)


plot_multiple(allQ, ax = ax , label = "saga", ls = '--', **kwargs)
plot_multiple(allQ1, ax = ax , label = "adagrad", ls = '--', **kwargs)
plot_multiple(allQ2, ax = ax , label = "svrg", ls = '--', **kwargs)
plot_multiple(allP, ax = ax , label = "snspp", **kwargs)


ax.set_xlim(-.1, 6)
ax.set_ylim(1e-7,)

ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_gisette/obj.pdf', dpi = 300)

#%% coeffcient plot

P = allP[-1]

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
allQ[0].plot_path(ax = ax[0,0], xlabel = False)
allQ1[0].plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
allQ2[0].plot_path(ax = ax[1,0])
allP[0].plot_path(ax = ax[1,1], ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.5,.3)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_gisette/coeff.pdf', dpi = 300)


#%%
fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"log_scale": False, "lw": 0.7, "markersize": 3, 'ls': '-'}

# plot_multiple_error(allQ, ax = ax , label = "saga", ls = '--', **kwargs)
# plot_multiple_error(allQ1, ax = ax , label = "adagrad", ls = '--', **kwargs)
# plot_multiple_error(allQ2, ax = ax , label = "svrg", ls = '--', **kwargs)
# plot_multiple_error(allP, ax = ax , label = "snspp", **kwargs)

Cont.plot_error(error_key = 'test_loss', ax = ax, ylabel = 'Test loss', **kwargs) 

ax.set_xlim(-.1, 6)
ax.set_ylim(0.32, 0.42)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_gisette/error.pdf', dpi = 300)




