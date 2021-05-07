"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from snspp.helper.data_generation import tstudent_test, get_triazines
from snspp.solver.opt_problem import problem
from snspp.experiments.experiment_utils import params_tuner, plot_multiple, initialize_solvers, adagrad_step_size_tuner, eval_test_set, plot_test_error, plot_multiple_error

#%% load data

l1 = 1e-3
v = 1.
poly = 4

f, phi, A, b, A_test, b_test = get_triazines(lambda1 = l1, train_size = .8, v = v, poly = poly, noise = 0.)
n = A.shape[1]

initialize_solvers(f, phi)

print("psi(0) = ", f.eval(np.zeros(n)))

#%% old param setup for v=0.25
# params_saga = {'n_epochs' : 200, 'gamma' : 4.}
# params_svrg = {'n_epochs' : 200, 'batch_size': 1, 'gamma': 8.}
# params_adagrad = {'n_epochs' : 500, 'batch_size': 15, 'gamma': 0.002}
# params_snspp = {'max_iter' : 1000, 'batch_size': 15, 'sample_style': 'constant', 'alpha_C' : 0.008, 'reduce_variance': True}

#%% parameter setup

params_saga = {'n_epochs' : 200, 'gamma' : 4.5}
params_svrg = {'n_epochs' : 200, 'batch_size': 10, 'gamma': 38.}
params_adagrad = {'n_epochs' : 500, 'batch_size': 30, 'gamma': 0.004}
params_snspp = {'max_iter' : 1200, 'batch_size': 15, 'sample_style': 'constant', 'alpha_C' : .03, 'reduce_variance': True}

#params_tuner(f, phi, solver = "saga", gamma_range = np.linspace(4,8, 10))
#params_tuner(f, phi, solver = "svrg", gamma_range = np.linspace(15, 50, 7), batch_range = np.array([10,20]))
#params_tuner(f, phi, solver = "adagrad", gamma_range = np.logspace(-3,-2, 6), batch_range = np.array([30, 50]))
#params_tuner(f, phi, solver = "snspp", gamma_range = np.linspace(0.02,0.05,10), batch_range = np.array([15,30]))

#%% determine psi_star

# params_ref = {'max_iter' : 500, 'batch_size': f.N, 'sample_style': 'constant', 'alpha_C' : 10., 'reduce_variance': True}
# ref = problem(f, phi, tol = 1e-6, params = params_ref, verbose = True, measure = True)
# ref.solve(solver = 'snspp')

# if poly = 3
#params_saga = {'n_epochs' : 200, 'gamma' : 3.}
#params_snspp = {'max_iter' : 1000, 'batch_size': 15, 'sample_style': 'constant', 'alpha_C' : .06, 'reduce_variance': True}
    

#%% solve with SAGA
Q = problem(f, phi, tol = 1e-6, params = params_saga, verbose = True, measure = True)

Q.solve(solver = 'saga')
print("f(x_t) = ", f.eval(Q.x))
print("phi(x_t) = ", phi.eval(Q.x))
print("psi(x_t) = ", f.eval(Q.x) + phi.eval(Q.x))


#%% solve with SVRG

Q2 = problem(f, phi, tol = 1e-6, params = params_svrg, verbose = True, measure = True)

Q2.solve(solver = 'svrg')
print("f(x_t) = ", f.eval(Q2.x))
print("phi(x_t) = ", phi.eval(Q2.x))
print("psi(x_t) = ", f.eval(Q2.x) + phi.eval(Q2.x))

#%% solve with ADAGRAD

#tune_params = {'n_epochs' : 300, 'batch_size': 15}
#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = tune_params)

Q1 = problem(f, phi, tol = 1e-6, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print("f(x_t) = ", f.eval(Q1.x))
print("phi(x_t) = ", phi.eval(Q1.x))
print("psi(x_t) = ", f.eval(Q1.x) + phi.eval(Q1.x))

#%% solve with SNSPP

P = problem(f, phi, tol = 1e-6, params = params_snspp, verbose = True, measure = True)
P.solve(solver = 'snspp')

print("f(x_t) = ", f.eval(P.x))
print("phi(x_t) = ", phi.eval(P.x))
print("psi(x_t) = ", f.eval(P.x) + phi.eval(P.x))

#%%

all_x = pd.DataFrame(np.vstack((P.x, Q.x, Q1.x)).T, columns = ['spp', 'saga', 'adagrad'])
#all_b = pd.DataFrame(np.vstack((b_test, A_test@P.x, A_test@Q.x, A_test@Q1.x)).T, columns = ['true', 'spp', 'saga', 'adagrad'])

#%% Test set evaluation

def tstudent_loss(x, A, b, v):
    z = A@x - b
    return 1/A.shape[0] * np.log(1+ z**2/v).sum()

kwargs2 = {"A": A_test, "b": b_test, "v": f.v}

tstudent_loss(Q.x, A_test, b_test, f.v)
tstudent_loss(Q1.x, A_test, b_test, f.v)
tstudent_loss(P.x, A_test, b_test, f.v)


L_P = eval_test_set(X = P.info["iterates"], loss = tstudent_loss, **kwargs2)
L_Q = eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2)
L_Q1 = eval_test_set(X = Q1.info["iterates"], loss = tstudent_loss, **kwargs2)
L_Q2 = eval_test_set(X = Q2.info["iterates"], loss = tstudent_loss, **kwargs2)

#all_loss_P = np.vstack([eval_test_set(X = P.info["iterates"], loss = tstudent_loss, **kwargs2) for P in allP])
#all_loss_Q = np.vstack([eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2) for Q in allQ])
#all_loss_Q1 = np.vstack([eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2) for Q in allQ1])

#%%

###########################################################################
# multiple execution and plotting
# As n is very large, we can not store all iterates for all runs and hence evaluate directly after solving and delete
############################################################################

#%% solve with SAGA (multiple times)

K = 20
allQ = list()
all_loss_Q = list()
for k in range(K):
    
    Q_k = problem(f, phi, tol = 1e-6, params = params_saga, verbose = True, measure = True)
    Q_k.solve(solver = 'saga')
    
    all_loss_Q.append(eval_test_set(X = Q_k.info["iterates"], loss = tstudent_loss, **kwargs2))
    Q_k.info.pop('iterates')
    
    allQ.append(Q_k)

all_loss_Q = np.vstack(all_loss_Q)

#%% solve with ADAGRAD (multiple times)

allQ1 = list()
all_loss_Q1 = list()
for k in range(K):
    
    Q1_k = problem(f, phi, tol = 1e-6, params = params_adagrad, verbose = True, measure = True)
    Q1_k.solve(solver = 'adagrad')
    
    all_loss_Q1.append(eval_test_set(X = Q1_k.info["iterates"], loss = tstudent_loss, **kwargs2))
    Q1_k.info.pop('iterates')
    
    allQ1.append(Q1_k)

all_loss_Q1 = np.vstack(all_loss_Q1)

#%% solve with SVRG (multiple times)

allQ2 = list()
all_loss_Q2 = list()
for k in range(K):
    
    Q2_k = problem(f, phi, tol = 1e-6, params = params_svrg, verbose = True, measure = True)
    Q2_k.solve(solver = 'svrg')
    
    all_loss_Q2.append(eval_test_set(X = Q2_k.info["iterates"], loss = tstudent_loss, **kwargs2))
    Q2_k.info.pop('iterates')
    
    allQ2.append(Q2_k)

all_loss_Q2 = np.vstack(all_loss_Q2)

#%% solve with SNSPP (multiple times, VR)

allP = list()
all_loss_P = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-6, params = params_snspp, verbose = False, measure = True)
    P_k.solve(solver = 'snspp')
    
    all_loss_P.append(eval_test_set(X = P_k.info["iterates"], loss = tstudent_loss, **kwargs2))
    P_k.info.pop('iterates')
    
    allP.append(P_k)

all_loss_P = np.vstack(all_loss_P)

#%% plot objective
save = False

# use the last objective of SAGA as surrogate optimal value / plot only psi(x^k)
#psi_star = f.eval(Q.x)+phi.eval(Q.x)
psi_star = 0


fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True, "lw": 0.4, "markersize": 1}

#Q.plot_objective(ax = ax, ls = '--', marker = '<',  **kwargs)
#Q1.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
#Q2.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
#P.plot_objective(ax = ax, **kwargs)

plot_multiple(allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
plot_multiple(allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
plot_multiple(allQ2, ax = ax , label = "svrg", ls = '--', marker = '>', **kwargs)
plot_multiple(allP, ax = ax , label = "snspp", **kwargs)

ax.set_xlim(0,140)
#ax.set_ylim(0.19,0.3)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.21,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_triazine/obj.pdf', dpi = 300)
    
#%% coefficent plot

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
Q2.plot_path(ax = ax[1,0])
P.plot_path(ax = ax[1,1], ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.2, .5)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_triazine/coeff.pdf', dpi = 300)
    
#%%
    
fig, ax = plt.subplots(1,1,  figsize = (4.5, 3.5))

kwargs = {"log_scale": False, "lw": 0.4, "markersize": 1}

#plot_test_error(Q, L_Q,  ax = ax,  marker = '<', **kwargs)
#plot_test_error(Q1, L_Q1,  ax = ax,  marker = '<', **kwargs)
#plot_test_error(Q2, L_Q2,  ax = ax,  marker = '<', **kwargs)
#plot_test_error(P, L_P,  ax = ax,  marker = 'o', **kwargs)

plot_multiple_error(all_loss_Q, allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
plot_multiple_error(all_loss_Q1, allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
plot_multiple_error(all_loss_Q2, allQ2, ax = ax , label = "svrg", ls = '--', marker = '>', **kwargs)
plot_multiple_error(all_loss_P, allP, ax = ax , label = "snspp", **kwargs)


ax.set_yscale('log')
ax.set_xlim(0,140)
#ax.set_ylim(0.08,0.15)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.21,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_triazine/error.pdf', dpi = 300)
