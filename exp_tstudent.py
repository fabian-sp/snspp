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

# N = 1000
# n = 10000
# k = 100
# kappa = 1e6

l1 = 1e-3
v = 1.

if v == 0.25:
    params_saga = {'n_epochs' : 200, 'gamma' : 4.}
    params_adagrad = {'n_epochs' : 500, 'batch_size': 15, 'gamma': 0.002}
    params_ssnsp = {'max_iter' : 1000, 'sample_size': 15, 'sample_style': 'constant', 'alpha_C' : 0.008, 'reduce_variance': True}
elif v == 1.:
    params_saga = {'n_epochs' : 200, 'gamma' : 4.5}
    params_adagrad = {'n_epochs' : 500, 'batch_size': 15, 'gamma': 0.002}
    params_ssnsp = {'max_iter' : 1200, 'sample_size': 15, 'sample_style': 'constant', 'alpha_C' : .032, 'reduce_variance': True}


f, phi, A, b, A_test, b_test = get_triazines(lambda1 = l1, train_size = .8, v = v, poly = 4, noise = 0.)

#xsol, A, b, f, phi, A_test, b_test = tstudent_test(N, n, k, l1, v = 20, noise = 0., scale = 20, kappa = kappa)
#xsol, A, b, f, phi, A_test, b_test = tstudent_test(N, n = 20, k = 2, lambda1=l1, v = 2, noise = 0., scale = 10, poly = 5)

n = A.shape[1]

# l1 <= lambda_max, if not 0 is a solution
#lambda_max = np.abs(1/f.N * A.T @ (2*b/(f.v+b**2))).max()
#phi.lambda1 = .2 *lambda_max

initialize_solvers(f, phi)

#x0 = f.A.T @ b
x0 = np.zeros(n)

sns.distplot(A@x0-b)
#(np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)).max()

print("psi(0) = ", f.eval(np.zeros(n)))

#xsol = None

#print("f(x*) = ", f.eval(xsol))
#print("phi(x*) = ", phi.eval(xsol))
#print("psi(x*) = ", f.eval(xsol) + phi.eval(xsol))

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

for k in range(2):
    P = problem(f, phi, x0 = x0, tol = 1e-6, params = params_ssnsp, verbose = True, measure = True)
    P.solve(solver = 'ssnsp')


#%% solve with SSNSP (multiple times, VR)
   
# K = 20
# allP = list()
# for k in range(K):
    
#     P_k = problem(f, phi, tol = 1e-6, params = params_ssnsp, verbose = False, measure = True)
#     P_k.solve(solver = 'ssnsp')
#     allP.append(P_k)
 
#%%
#all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, Q1.x)).T, columns = ['true', 'spp', 'saga', 'adagrad'])
#all_x = pd.DataFrame(np.vstack((xsol,Q.x)).T, columns = ['true', 'saga'])

all_x = pd.DataFrame(np.vstack((P.x, Q.x, Q1.x)).T, columns = ['spp', 'saga', 'adagrad'])

#%%
save = False

# use the last objective of SAGA as surrogate optimal value
psi_star = f.eval(Q.x)+phi.eval(Q.x)
psi_star = 0


fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True}

Q.plot_objective(ax = ax, ls = '--', marker = '<', markersize = 2., **kwargs)
Q1.plot_objective(ax = ax, ls = '--', marker = '<', markersize = 1.5, **kwargs)
#Q2.plot_objective(ax = ax, ls = '--', marker = '<', markersize = 1., **kwargs)
P.plot_objective(ax = ax, markersize = 1.5, **kwargs)

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
    fig.savefig(f'data/plots/exp_triazine/obj_{round(v,2)}.pdf', dpi = 300)
    
#%% coefficent plot

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
P.plot_path(ax = ax[1,0])
ax[1,1].axis('off')#P.plot_path(ax = ax[1,1], mean = True, ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.2, .5)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_triazine/coeff_{round(v,2)}.pdf', dpi = 300)
    
#%%

def tstudent_loss(x, A_test, b_test, v):
    z = A_test@x - b_test
    return 1/A_test.shape[0] * np.log(1+ z**2/v).sum()

tstudent_loss(x0, A_test, b_test, f.v)
tstudent_loss(Q.x, A_test, b_test, f.v)
tstudent_loss(Q1.x, A_test, b_test, f.v)
tstudent_loss(P.x, A_test, b_test, f.v)

all_b = pd.DataFrame(np.vstack((b_test, A_test@P.x, A_test@Q.x, A_test@Q1.x)).T, columns = ['true', 'spp', 'saga', 'adagrad'])

#%%

kwargs2 = {"A_test": A_test, "b_test": b_test, "v": f.v}

L_P = eval_test_set(X = P.info["iterates"], loss = tstudent_loss, **kwargs2)
L_Q = eval_test_set(X = Q.info["iterates"], loss = tstudent_loss, **kwargs2)
L_Q1 = eval_test_set(X = Q1.info["iterates"], loss = tstudent_loss, **kwargs2)

#%%
    
fig, ax = plt.subplots(1,1,  figsize = (4.5, 3.5))

plot_test_error(Q, L_Q,  ax = ax,  marker = '<', markersize = 2.)
plot_test_error(Q1, L_Q1,  ax = ax,  marker = '<', markersize = 2.)
plot_test_error(P, L_P,  ax = ax,  marker = 'o', markersize = 1.5)

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
    fig.savefig(f'data/plots/exp_triazine/error_{round(v,2)}.pdf', dpi = 300)
    