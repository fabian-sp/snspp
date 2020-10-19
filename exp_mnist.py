import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssnsp.solver.opt_problem import problem, color_dict
from ssnsp.helper.data_generation import get_mnist
from ssnsp.experiments.experiment_utils import plot_multiple, adagrad_step_size_tuner, initialize_fast_gradient

from sklearn.linear_model import LogisticRegression


f, phi, X_train, y_train, X_test, y_test = get_mnist()

#plt.imshow(X_train[110,:].reshape(28,28))

print("Regularization parameter lambda:", phi.lambda1)

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                        solver = 'saga', max_iter = 200, verbose = 1)


start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

#(np.sign(predict(X_train, x_sk)) == np.sign(y_train)).sum() / len(y_train)

psi_star = f.eval(x_sk) + phi.eval(x_sk)
initialize_fast_gradient(f, phi)

#%% solve with SAGA

params = {'n_epochs' : 50, 'reg': 1e-2}

Q = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))

#%% solve with ADAGRAD

#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = None)
opt_gamma = 0.018738 #0.005

params = {'n_epochs' : 200, 'batch_size': int(f.N*0.05), 'gamma': opt_gamma}

Q1 = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))

 
#%% solve with SSNSP

params = {'max_iter' : 70, 'sample_size': 1000, 'sample_style': 'fast_increasing', \
          'alpha_C' : 10., 'reduce_variance': True}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)
P.solve(solver = 'ssnsp')
  
#%% solve with SSNSP (multiple times, VR)

params = {'max_iter' : 70, 'sample_size': 1000, 'sample_style': 'fast_increasing', \
          'alpha_C' : 10., 'reduce_variance': True}
K = 20
allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-7, params = params, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k.info)

#%% solve with SSNSP (multiple times, no VR)

params1 = params.copy()
params1["reduce_variance"] = False

allP1 = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-7, params = params1, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP1.append(P_k.info)
    

#%% solve with CONSTANT SSNSP

params = {'max_iter' : 30, 'sample_size': 3000, 'sample_style': 'constant', 'alpha_C' : 10.}

P1 = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P1.solve(solver = 'ssnsp')


#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad'])

#%% objective plot

save = False

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True}

Q.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
Q1.plot_objective(ax = ax, ls = '-.', marker = '>', **kwargs)


#plot_multiple(allP, ax = ax , label = "ssnsp", **kwargs)
#plot_multiple(allP1, ax = ax , label = "ssnsp_noVR", name = "ssnsp (no VR)", **kwargs)

P.plot_objective(ax = ax, **kwargs)
#P1.plot_objective(ax = ax, label = "_constant", marker = "x")


ax.set_xlim(-1,16)
ax.legend()
#ax.set_yscale('log')

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_mnist/obj.pdf', dpi = 300)

#%% coefficent plot

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
P.plot_path(ax = ax[1,0])
P.plot_path(ax = ax[1,1], mean = True, ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.22,.22)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_mnist/coeff.pdf', dpi = 300)


#%%

def predict(A,x):
    
    h = np.exp(A@x)
    odds = h/(1+h)    
    y = (odds >= .5)*2 -1
    
    return y

def sample_error(A, b, x):
    
    b_pred = predict(A,x)
    return (np.sign(b_pred) == np.sign(b)).sum() / len(b)


sample_error(X_test, y_test, x_sk)

sample_error(X_test, y_test, Q.x)

sample_error(X_test, y_test, Q1.x)

sample_error(X_test, y_test, P.x)



