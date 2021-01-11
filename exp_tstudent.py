"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from ssnsp.helper.data_generation import tstudent_test, get_triazines
from ssnsp.solver.opt_problem import problem
from ssnsp.experiments.experiment_utils import plot_multiple, initialize_solvers, adagrad_step_size_tuner, eval_test_set, plot_test_error

#%% generate data

l1 = 1e-3
v = 1.

if v == 0.25:
    params_saga = {'n_epochs' : 200, 'gamma' : 4.}
    params_adagrad = {'n_epochs' : 500, 'batch_size': 15, 'gamma': 0.002}
    params_ssnsp = {'max_iter' : 1000, 'sample_size': 15, 'sample_style': 'constant', 'alpha_C' : 0.008, 'reduce_variance': True}
elif v == 1.:
    params_saga = {'n_epochs' : 200, 'gamma' : 5.}
    params_adagrad = {'n_epochs' : 500, 'batch_size': 15, 'gamma': 0.002}
    params_ssnsp = {'max_iter' : 1200, 'sample_size': 15, 'sample_style': 'constant', 'alpha_C' : .032, 'reduce_variance': True}


f, phi, A, b, A_test, b_test = get_triazines(lambda1 = l1, train_size = .8, v = v, poly = 4, noise = 0.)

n = A.shape[1]

initialize_solvers(f, phi)

x0 = np.zeros(n)

sns.distplot(A@x0-b)

print("psi(0) = ", f.eval(np.zeros(n)))
#%% determine psi_star

#x0 = P.x
# params_ref = {'max_iter' : 500, 'sample_size': f.N, 'sample_style': 'constant', 'alpha_C' : 10., 'reduce_variance': True}
# ref = problem(f, phi, x0 = x0, tol = 1e-6, params = params_ref, verbose = True, measure = True)
# ref.solve(solver = 'ssnsp')


#%% solve with SAGA
Q = problem(f, phi, x0 = x0, tol = 1e-6, params = params_saga, verbose = True, measure = True)

Q.solve(solver = 'saga')
print("f(x_t) = ", f.eval(Q.x))
print("phi(x_t) = ", phi.eval(Q.x))
print("psi(x_t) = ", f.eval(Q.x) + phi.eval(Q.x))


#%% solve with SVRG/ BATCH-SAGA
# params_svrg = {'n_epochs' : 100, 'gamma' : 4.}

# Q2 = problem(f, phi, x0 = x0, tol = 1e-6, params = params_svrg, verbose = True, measure = True)

# Q2.solve(solver = 'svrg')

# print(f.eval(Q2.x)+phi.eval(Q2.x))

#%% solve with ADAGRAD

#tune_params = {'n_epochs' : 300, 'batch_size': 15}
#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = tune_params)

Q1 = problem(f, phi, x0 = x0, tol = 1e-6, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print("f(x_t) = ", f.eval(Q1.x))
print("phi(x_t) = ", phi.eval(Q1.x))
print("psi(x_t) = ", f.eval(Q1.x) + phi.eval(Q1.x))

#%% solve with SSNSP

P = problem(f, phi, x0 = x0, tol = 1e-6, params = params_ssnsp, verbose = True, measure = True)
P.solve(solver = 'ssnsp')

print("f(x_t) = ", f.eval(P.x))
print("phi(x_t) = ", phi.eval(P.x))
print("psi(x_t) = ", f.eval(P.x) + phi.eval(P.x))

#%%

all_x = pd.DataFrame(np.vstack((P.x, Q.x, Q1.x)).T, columns = ['spp', 'saga', 'adagrad'])

#%%

###########################################################################
# multiple execution and plotting
############################################################################

#%% solve with SAGA (multiple times)

K = 20
allQ = list()
for k in range(K):
    
    Q_k = problem(f, phi, tol = 1e-9, params = params_saga, verbose = True, measure = True)
    Q_k.solve(solver = 'saga')
    allQ.append(Q_k)

#%% solve with ADAGRAD (multiple times)

allQ1 = list()
for k in range(K):
    
    Q1_k = problem(f, phi, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)
    Q1_k.solve(solver = 'adagrad')
    allQ1.append(Q1_k)
    
#%% solve with SSNSP (multiple times, VR)

allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-9, params = params_ssnsp, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k)
    
#%%
save = False

# use the last objective of SAGA as surrogate optimal value
#psi_star = f.eval(Q.x)+phi.eval(Q.x)
psi_star = 0


fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True, "lw": 0.4, "markersize": 3}

Q.plot_objective(ax = ax, ls = '--', marker = '<', markersize = 2., **kwargs)
Q1.plot_objective(ax = ax, ls = '--', marker = '<', markersize = 1.5, **kwargs)
P.plot_objective(ax = ax, markersize = 1.5, **kwargs)

#plot_multiple(allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
#plot_multiple(allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
#plot_multiple(allP, ax = ax , label = "ssnsp", **kwargs)

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
P.plot_path(ax = ax[1,0])
ax[1,1].axis('off')

for a in ax.ravel():
    a.set_ylim(-.2, .5)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_triazine/coeff.pdf', dpi = 300)
    
#%%

def tstudent_loss(x, A, b, v):
    z = A@x - b
    return 1/A.shape[0] * np.log(1+ z**2/v).sum()

tstudent_loss(x0, A_test, b_test, f.v)
tstudent_loss(Q.x, A_test, b_test, f.v)
tstudent_loss(Q1.x, A_test, b_test, f.v)
tstudent_loss(P.x, A_test, b_test, f.v)

all_b = pd.DataFrame(np.vstack((b_test, A_test@P.x, A_test@Q.x, A_test@Q1.x)).T, columns = ['true', 'spp', 'saga', 'adagrad'])

#%%

kwargs2 = {"A": A_test, "b": b_test, "v": f.v}

# L_P = eval_test_set(X = P.info["iterates"], loss = tstudent_loss, **kwargs2)
# L_Q = eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2)
# L_Q1 = eval_test_set(X = Q1.info["iterates"], loss = tstudent_loss, **kwargs2)

all_loss_P = np.vstack([eval_test_set(X = P.info["iterates"], loss = tstudent_loss, **kwargs2) for P in allP])
all_loss_Q = np.vstack([eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2) for Q in allQ])
all_loss_Q1 = np.vstack([eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2) for Q in allQ1])


#%%
    
fig, ax = plt.subplots(1,1,  figsize = (4.5, 3.5))

kwargs = {"log_scale": False, "lw": 0.4, "markersize": 3}

#plot_test_error(Q, L_Q,  ax = ax,  marker = '<', markersize = 2.)
#plot_test_error(Q1, L_Q1,  ax = ax,  marker = '<', markersize = 2.)
#plot_test_error(P, L_P,  ax = ax,  marker = 'o', markersize = 1.5)

plot_multiple_error(all_loss_Q, allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
plot_multiple_error(all_loss_Q1, allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
plot_multiple_error(all_loss_P, allP, ax = ax , label = "ssnsp", **kwargs)


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