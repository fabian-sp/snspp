"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from snspp.helper.data_generation import get_cpusmall, tstudent_test
from snspp.solver.opt_problem import problem
from snspp.experiments.experiment_utils import params_tuner, plot_multiple, initialize_solvers, eval_test_set, plot_test_error, plot_multiple_error

#%% load data
#f, phi, X_train, y_train, X_test, y_test = get_cpusmall(lambda1 = l1, train_size = .99, v = v, poly = 0, noise = 0.)

setup = 2

if setup == 1:

    l1 = 0.01
    v = 1.
    poly = 0
    n = 5000; N = 1000; k = 10
    noise = 0.1
    
elif setup == 2:
    l1 = 0.01
    v = 1.
    poly = 0
    n = 5000; N = 1000; k = 100
    noise = 0.1


#%%

xsol, X_train, y_train, f, phi, X_test, y_test = tstudent_test(N = N, n = n, k = k, lambda1 = l1, v = v,\
                                                               noise = noise, poly = poly, kappa = 15., dist = 'ortho')

initialize_solvers(f, phi)
print("psi(0) = ", f.eval(np.zeros(n)))

#%% parameter setup

if setup == 1:
    params_saga = {'n_epochs' : 200, 'alpha' : 16}
    params_svrg = {'n_epochs' : 200, 'batch_size': 20, 'alpha': 450.}
    params_adagrad = {'n_epochs' : 300, 'batch_size': 100, 'alpha': 0.07}
    
    params_snspp = {'max_iter' : 200, 'batch_size': 10, 'sample_style': 'constant', 'alpha' : 5, 'reduce_variance': True}

elif setup == 2:
    params_saga = {'n_epochs' : 200, 'alpha' : 30}
    params_svrg = {'n_epochs' : 200, 'batch_size': 20, 'alpha': 1000.}
    params_adagrad = {'n_epochs' : 300, 'batch_size': 100, 'alpha': 0.14}
    
    params_snspp = {'max_iter' : 200, 'batch_size': 10, 'sample_style': 'fast_increasing', 'alpha' : 9, 'reduce_variance': True}



#params_tuner(f, phi, solver = "saga", alpha_range = np.linspace(20,50, 10), n_iter = 150)
#params_tuner(f, phi, solver = "svrg", alpha_range = np.linspace(400, 1000, 7), batch_range = np.array([10,20]), n_iter = 150)
#params_tuner(f, phi, solver = "adagrad", alpha_range = np.logspace(-3,0,6), batch_range = np.array([10, 50, 100]), n_iter = 150)
#params_tuner(f, phi, solver = "snspp", alpha_range = np.linspace(5,15,10), batch_range = np.array([10,20]), n_iter = 200)

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


#P.plot_objective()
#P.plot_path()
#fig = P.plot_subproblem(start = 0)

#%% plot objective
save = False

# use the last objective of SAGA as surrogate optimal value / plot only psi(x^k)
psi_star = f.eval(Q.x)+phi.eval(Q.x)
#psi_star = 0


fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True, "lw": 0.4, "markersize": 1}

Q.plot_objective(ax = ax, ls = '--', marker = '<',  **kwargs)
#Q1.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
#Q2.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
P.plot_objective(ax = ax, **kwargs)

#plot_multiple(allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
#plot_multiple(allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
#plot_multiple(allQ2, ax = ax , label = "svrg", ls = '--', marker = '>', **kwargs)
#plot_multiple(allP, ax = ax , label = "snspp", **kwargs)

ax.set_xlim(0,0.6)
#ax.set_ylim(0.19,0.3)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.21,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_cpusmall/obj.pdf', dpi = 300)
    
#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, Q1.x)).T, columns = ['sol', 'spp', 'saga', 'adagrad'])

#%% Test set evaluation

def tstudent_loss(x, A, b, v):
    z = A@x - b
    return 1/A.shape[0] * np.log(1+ z**2/v).sum()

kwargs2 = {"A": X_test, "b": y_test, "v": f.v}


L_P = eval_test_set(X = P.info["iterates"], loss = tstudent_loss, **kwargs2)
L_Q = eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2)
L_Q1 = eval_test_set(X = Q1.info["iterates"], loss = tstudent_loss, **kwargs2)
L_Q2 = eval_test_set(X = Q2.info["iterates"], loss = tstudent_loss, **kwargs2)

#all_loss_P = np.vstack([eval_test_set(X = P.info["iterates"], loss = tstudent_loss, **kwargs2) for P in allP])
#all_loss_Q = np.vstack([eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2) for Q in allQ])
#all_loss_Q1 = np.vstack([eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2) for Q in allQ1])

#%%
    
fig, ax = plt.subplots(1,1,  figsize = (4.5, 3.5))

kwargs = {"log_scale": False, "lw": 0.4, "markersize": 1}

plot_test_error(Q, L_Q,  ax = ax,  marker = '<', **kwargs)
plot_test_error(Q1, L_Q1,  ax = ax,  marker = '<', **kwargs)
plot_test_error(Q2, L_Q2,  ax = ax,  marker = '<', **kwargs)
plot_test_error(P, L_P,  ax = ax,  marker = 'o', **kwargs)

#plot_multiple_error(all_loss_Q, allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
#plot_multiple_error(all_loss_Q1, allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
#plot_multiple_error(all_loss_Q2, allQ2, ax = ax , label = "svrg", ls = '--', marker = '>', **kwargs)
#plot_multiple_error(all_loss_P, allP, ax = ax , label = "snspp", **kwargs)


ax.set_yscale('log')
ax.set_xlim(0,0.8)
#ax.set_ylim(0.08,0.15)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.21,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

