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
from ssnsp.experiments.experiment_utils import plot_multiple, initialize_fast_gradient, adagrad_step_size_tuner


def sample_loss(x, A_test, b_test, v):
    z = A_test@x - b_test
    return 1/A_test.shape[0] * np.log(1+ z**2/v).sum()
    

#%% generate data

# N = 1000
# n = 10000
# k = 100
# kappa = 1e6

l1 = 1e-3

v = 1.

f, phi, A, b, A_test, b_test = get_triazines(lambda1 = l1, train_size = .8, v = v, poly = 4, noise = 0.)

#xsol, A, b, f, phi, A_test, b_test = tstudent_test(N, n, k, l1, v = 20, noise = 0., scale = 20, kappa = kappa)
#xsol, A, b, f, phi, A_test, b_test = tstudent_test(N, n = 20, k = 2, lambda1=l1, v = 2, noise = 0., scale = 10, poly = 5)

n = A.shape[1]

# l1 <= lambda_max, if not 0 is a solution
#lambda_max = np.abs(1/f.N * A.T @ (2*b/(f.v+b**2))).max()
#phi.lambda1 = .2 *lambda_max

initialize_fast_gradient(f, phi)

#x0 = f.A.T @ b
x0 = np.zeros(n)

sns.distplot(A@x0-b)
#(np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)).max()

print("psi(0) = ", f.eval(np.zeros(n)))

#xsol = None

#print("f(x*) = ", f.eval(xsol))
#print("phi(x*) = ", phi.eval(xsol))
#print("psi(x*) = ", f.eval(xsol) + phi.eval(xsol))

#%% solve with SAGA
params_saga = {'n_epochs' : 150, 'gamma' : 4.}

Q = problem(f, phi, x0 = x0, tol = 1e-9, params = params_saga, verbose = True, measure = True)

Q.solve(solver = 'saga')

print("f(x_t) = ", f.eval(Q.x))
print("phi(x_t) = ", phi.eval(Q.x))
print("psi(x_t) = ", f.eval(Q.x) + phi.eval(Q.x))


#%% solve with SVRG/ BATCH-SAGA
params_svrg = {'n_epochs' : 100, 'gamma' : 3.}

Q2 = problem(f, phi, x0 = x0, tol = 1e-9, params = params_svrg, verbose = True, measure = True)

Q2.solve(solver = 'svrg')

print(f.eval(Q2.x)+phi.eval(Q2.x))

#%% solve with ADAGRAD

tune_params = {'n_epochs' : 300, 'batch_size': 15}
#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = tune_params)
opt_gamma = 0.005 

params_adagrad = {'n_epochs' : 300, 'batch_size': 15, 'gamma': opt_gamma}

Q1 = problem(f, phi, x0 = x0, tol = 1e-5, params = params_adagrad, verbose = True, measure = True)
Q1.solve(solver = 'adagrad')

print("f(x_t) = ", f.eval(Q1.x))
print("phi(x_t) = ", phi.eval(Q1.x))
print("psi(x_t) = ", f.eval(Q1.x) + phi.eval(Q1.x))

#%% solve with SSNSP

params_ssnsp = {'max_iter' : 700, 'sample_size': 15, 'sample_style': 'constant',\
          'alpha_C' : 0.032, 'reduce_variance': True}

P = problem(f, phi, x0 = x0, tol = 1e-9, params = params_ssnsp, verbose = True, measure = True)

P.solve(solver = 'ssnsp')


#%% solve with SSNSP (multiple times, VR)
   
K = 20
allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-9, params = params_ssnsp, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k)
 
#%%
#all_x = pd.DataFrame(np.vstack((xsol, P.x, Q.x, Q1.x)).T, columns = ['true', 'spp', 'saga', 'adagrad'])
#all_x = pd.DataFrame(np.vstack((xsol,Q.x)).T, columns = ['true', 'saga'])

all_x = pd.DataFrame(np.vstack((P.x, Q.x, Q1.x)).T, columns = ['spp', 'saga', 'adagrad'])
#%%
save = False

#psi_star = f.eval(Q.x)+phi.eval(Q.x)
psi_star = 0

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True}

Q.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
Q1.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
#Q2.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
P.plot_objective(ax = ax, markersize = 2, **kwargs)

#plot_multiple(allP, ax = ax , label = "ssnsp", **kwargs)

#ax.set_xlim(-.1,20)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

#%% coefficent plot

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
P.plot_path(ax = ax[1,0])
P.plot_path(ax = ax[1,1], mean = True, ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.2, .5)
    
plt.subplots_adjust(hspace = 0.33)


#%%
#sample_loss(xsol, A_test, b_test, f.v)

sample_loss(x0, A_test, b_test, f.v)

sample_loss(Q.x, A_test, b_test, f.v)

sample_loss(Q1.x, A_test, b_test, f.v)

sample_loss(P.x, A_test, b_test, f.v)


#%%
  
if v == 0.25:
    params_saga = {'n_epochs' : 200, 'gamma' : 4.}
    params_adagrad = {'n_epochs' : 300, 'batch_size': 15, 'gamma': 0.002}
    params_ssnsp = {'max_iter' : 800, 'sample_size': 15, 'sample_style': 'constant', 'alpha_C' : 0.008, 'reduce_variance': True}
elif v == 1.:
    params_saga = {'n_epochs' : 200, 'gamma' : 4.}
    params_adagrad = {'n_epochs' : 300, 'batch_size': 15, 'gamma': 0.005}
    params_ssnsp = {'max_iter' : 600, 'sample_size': 15, 'sample_style': 'constant', 'alpha_C' : .032, 'reduce_variance': True}
