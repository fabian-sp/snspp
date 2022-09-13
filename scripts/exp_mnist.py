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
from snspp.helper.data_generation import get_mnist
from snspp.experiments.experiment_utils import params_tuner, initialize_solvers, eval_test_set,\
                                                logreg_loss, logreg_accuracy

from snspp.experiments.container import Experiment

from sklearn.linear_model import LogisticRegression


f, phi, A, X_train, y_train, X_test, y_test = get_mnist()

#plt.imshow(X_train[119,:].reshape(28,28))

print("Regularization parameter lambda:", phi.lambda1)

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-9, \
                        solver = 'saga', max_iter = 300, verbose = 1)


start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

psi_star = f.eval(A@x_sk) + phi.eval(x_sk)
print("psi(x*) = ", psi_star)

initialize_solvers(f, phi, A)

#%% params

params_saga = {'n_epochs': 20, 'alpha': 0.00045}

params_svrg = {'n_epochs': 15, 'batch_size': 650, 'alpha': 0.45583}

params_adagrad = {'n_epochs' : 100, 'batch_size': int(f.N*0.05), 'alpha': 0.03}

params_snspp = {'max_iter' : 120, 'batch_size': 280, 'sample_style': 'constant', \
                'alpha' : 3., 'reduce_variance': True}
        
#params_tuner(f, phi, A, solver = "adagrad", batch_range = np.array([100, 1000, 3000]))

#%% solve with SAGA

Q = problem(f, phi, A, tol = 1e-9, params = params_saga, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(A@Q.x)+phi.eval(Q.x))

#%% solve with ADAGRAD

Q1 = problem(f, phi, A, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(A@Q1.x)+phi.eval(Q1.x))

#%% solve with SVRG

Q2 = problem(f, phi, A, tol = 1e-9, params = params_svrg, verbose = True, measure = True)

Q2.solve(solver = 'svrg')

print(f.eval(A@Q2.x)+phi.eval(Q2.x))

#%% solve with SSNSP

P = problem(f, phi, A, tol = 1e-9, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

print(f.eval(A@P.x)+phi.eval(P.x))

#%%

###########################################################################
# multiple execution
############################################################################

K = 20

kwargs2 = {"A": X_test, "b": y_test}
loss = [logreg_loss, logreg_accuracy]
names = ['test_loss', 'test_accuracy']

Cont = Experiment(name = 'exp_mnist')

Cont.params = {'saga':params_saga, 'svrg': params_svrg, 'adagrad':params_adagrad, 'snspp':params_snspp}
Cont.psi_star = psi_star

#Cont.load_from_disk(path='../data/output/')

#%% solve with SAGA (multiple times)

allQ = list()
for k in range(K):
    
    Q_k = problem(f, phi, A, tol = 1e-9, params = params_saga, verbose = True, measure = True)
    Q_k.solve(solver = 'saga')
    
    Cont.store(Q_k, k)
    err_k = eval_test_set(X = Q_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q_k.solver, k = k)
    
    allQ.append(Q_k)

#%% solve with ADAGRAD (multiple times)

allQ1 = list()
for k in range(K):
    
    Q1_k = problem(f, phi, A, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)
    Q1_k.solve(solver = 'adagrad')
    
    Cont.store(Q1_k, k)
    err_k = eval_test_set(X = Q1_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q1_k.solver, k = k)
    
    allQ1.append(Q1_k)

#%% solve with SVRG (multiple times)

allQ2 = list()
for k in range(K):
    
    Q2_k = problem(f, phi, A, tol = 1e-9, params = params_svrg, verbose = True, measure = True)
    Q2_k.solve(solver = 'svrg')
    
    Cont.store(Q2_k, k)
    err_k = eval_test_set(X = Q2_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = Q2_k.solver, k = k)
    
    allQ2.append(Q2_k)
    
#%% solve with SSNSP (multiple times, VR)

allP = list()
for k in range(K):
    
    P_k = problem(f, phi, A, tol = 1e-9, params = params_snspp, verbose = False, measure = True)
    P_k.solve(solver = 'snspp')
    
    Cont.store(P_k, k)
    err_k = eval_test_set(X = P_k.info["iterates"], loss = loss, names = names, kwargs = kwargs2)
    Cont.store_by_key(res = err_k, label = P_k.solver, k = k)
    
    allP.append(P_k)

#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x, Q2.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad', 'svrg'])

Cont.save_to_disk(path = '../data/output/')

#%%

###########################################################################
# plotting
############################################################################

xlim = (0, 3)

#%% objective plot

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True, "lw": 1., "markersize": 2.5}

# Q.plot_objective(ax = ax, **kwargs)
# Q1.plot_objective(ax = ax, **kwargs)
# Q2.plot_objective(ax = ax, **kwargs)
# P.plot_objective(ax = ax, **kwargs)

Cont.plot_objective(ax = ax, median = False, **kwargs) 

ax.set_xlim(xlim)
ax.set_ylim(1e-7,1e-1)

ax.legend(fontsize = 10, loc = 'upper right')

fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)

if save:
    fig.savefig(f'../data/plots/exp_mnist/obj.pdf', dpi = 300)


#%% test loss

fig,ax = plt.subplots(figsize = (4.5, 3.5))
kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}

Cont.plot_error(error_key = 'test_loss', ax = ax, median = True, ylabel = 'Test loss', **kwargs) 

ax.set_xlim(xlim)
ax.set_ylim(0.45, 0.5)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)

if save:
    fig.savefig(f'../data/plots/exp_mnist/error.pdf', dpi = 300)

#%% test accuracy

fig,ax = plt.subplots(figsize = (4.5, 3.5))
kwargs = {"log_scale": False, "lw": 1., "markersize": 1.5, 'ls': '-'}

Cont.plot_error(error_key = 'test_accuracy', ax = ax, median = True, ylabel = 'Test accuracy', **kwargs) 

ax.set_xlim(xlim)
ax.set_ylim(0.6, 1.)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)

if save:
    fig.savefig(f'../data/plots/exp_mnist/accuracy.pdf', dpi = 300)


#%% coefficent plot

fig,ax = plt.subplots(2, 2, figsize = (7,5))

Q_k.plot_path(ax = ax[0,0], xlabel = False)
Q1_k.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
Q2_k.plot_path(ax = ax[1,0])
P_k.plot_path(ax = ax[1,1], ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.25,.25)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'../data/plots/exp_mnist/coeff.pdf', dpi = 300)



