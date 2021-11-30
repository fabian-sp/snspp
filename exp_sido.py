"""
@author: Fabian Schaipp
"""
import sys

if len(sys.argv) > 2:
    save = sys.argv[1]
    l1 = float(sys.argv[2])
else:
    save = False
    l1 = 1e-3
    
#%%
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from snspp.solver.opt_problem import problem
from snspp.helper.data_generation import get_sido
from snspp.experiments.experiment_utils import params_tuner, plot_multiple, plot_multiple_error, initialize_solvers, eval_test_set,\
                                                convert_to_dict


from sklearn.linear_model import LogisticRegression


f, phi, X_train, y_train, X_test, y_test = get_sido(lambda1 = l1, scale = False)

print("Regularization parameter lambda:", phi.lambda1)

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                        solver = 'saga', max_iter = 150, verbose = 0)

start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

#(np.sign(predict(X_test, x_sk)) == np.sign(y_test)).sum() / len(y_test)

psi_star = f.eval(x_sk) + phi.eval(x_sk)
print("l1, psi(x*) = ", l1, psi_star)

initialize_solvers(f, phi)

#%% params 

if l1 == 1e-3:
    params_saga = {'n_epochs' : 20, 'alpha': 6.5}
    
    params_svrg = {'n_epochs' : 20, 'batch_size': 50, 'alpha': 160.}
    #params_svrg = {'n_epochs' : 30, 'batch_size': 100, 'alpha': 270.} # not much worse
    
    params_adagrad = {'n_epochs' : 60, 'batch_size': 20, 'alpha': 0.008}
    
    params_snspp = {'max_iter' : 400, 'batch_size': 60, 'sample_style': 'constant', 'alpha' : 0.4,\
                    "reduce_variance": True}
        
    #params_tuner(f, phi, solver = "saga", alpha_range = np.linspace(5,12,12), n_iter = 25)
    #params_tuner(f, phi, solver = "svrg", alpha_range = np.linspace(200, 500, 10), batch_range = np.array([50, 100,200]), n_iter = 40)
    #params_tuner(f, phi, solver = "adagrad", batch_range = np.array([20, 200, 500]))
    #params_tuner(f, phi, solver = "snspp", alpha_range = np.linspace(0.1, 0.6, 5), batch_range = np.array([50,75,100]), n_iter = 200)

elif l1 == 1e-2:
    
    params_saga = {'n_epochs' : 20, 'alpha': 8.}
    
    params_svrg = {'n_epochs' : 20, 'batch_size': 200, 'alpha': 500.}
    
    params_adagrad = {'n_epochs' : 60, 'batch_size': 200, 'alpha': 0.06}
    
    params_snspp = {'max_iter' : 400, 'batch_size': 100, 'sample_style': 'constant', 'alpha' : 1.,\
                    "reduce_variance": True}
    #50,1.4
    #params_tuner(f, phi, solver = "saga", alpha_range = np.linspace(5,15,10), n_iter = 20)
    #params_tuner(f, phi, solver = "svrg", alpha_range = np.linspace(400, 1000, 10), batch_range = np.array([50, 200]), n_iter = 30)
    #params_tuner(f, phi, solver = "adagrad", batch_range = np.array([20, 200, 500]))
    #params_tuner(f, phi, solver = "snspp", alpha_range = np.linspace(1,4,6), batch_range = np.array([50, 100, 200]), n_iter = 150)
    
else:
    raise KeyError("Parameters not tuned for this value of l1.")
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

#%% solve with SNSPP

P = problem(f, phi, tol = 1e-9, params = params_snspp, verbose = True, measure = True)

P.solve(solver = 'snspp')

#fig = P.plot_subproblem(M=20)

#%%

###########################################################################
# multiple execution and plotting
############################################################################

#%% solve with SAGA (multiple times)

K = 20
allQ = list()
for k in range(K):
    
    Q_k = problem(f, phi, tol = 1e-16, params = params_saga, verbose = True, measure = True)
    Q_k.solve(solver = 'saga')
    allQ.append(Q_k)

#%% solve with ADAGRAD (multiple times)

allQ1 = list()
for k in range(K):
    
    Q1_k = problem(f, phi, tol = 1e-16, params = params_adagrad, verbose = True, measure = True)
    Q1_k.solve(solver = 'adagrad')
    allQ1.append(Q1_k)

#%% solve with SVRG (multiple times)

allQ2 = list()
for k in range(K):
    
    Q2_k = problem(f, phi, tol = 1e-16, params = params_svrg, verbose = True, measure = True)
    Q2_k.solve(solver = 'svrg')
    allQ2.append(Q2_k)
    
#%% solve with SNSPP (multiple times, VR)

allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-16, params = params_snspp, verbose = False, measure = True)
    P_k.solve(solver = 'snspp')
    allP.append(P_k)


#%% eval test set loss

def logreg_loss(x, A, b):
    z = A@x
    return np.log(1 + np.exp(-b*z)).mean()

kwargs2 = {"A": X_test, "b": y_test}

for P in allP: P.info['test_error'] = eval_test_set(X = P.info["iterates"], loss = logreg_loss, **kwargs2)
for Q in allQ: Q.info['test_error'] = eval_test_set(X = Q.info["iterates"], loss = logreg_loss, **kwargs2)
for Q in allQ1: Q.info['test_error'] = eval_test_set(X = Q.info["iterates"], loss = logreg_loss, **kwargs2)
for Q in allQ2: Q.info['test_error'] = eval_test_set(X = Q.info["iterates"], loss = logreg_loss, **kwargs2)
    
#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad'])

res_to_save = dict()
res_to_save.update(convert_to_dict(allQ))
res_to_save.update(convert_to_dict(allQ1))
res_to_save.update(convert_to_dict(allQ2))
res_to_save.update(convert_to_dict(allP))

np.save(f'data/output/exp_sido_l1_{l1}.npy', res_to_save)

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


ax.set_xlim(-.1, 4)
ax.set_ylim(1e-7,)

ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_sido/l1_{l1}/obj.pdf', dpi = 300)

#%% coeffcient plot

#P = allP[-1]

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
allQ[0].plot_path(ax = ax[0,0], xlabel = False)
allQ1[0].plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
allQ2[0].plot_path(ax = ax[1,0])
allP[0].plot_path(ax = ax[1,1], ylabel = False)

for a in ax.ravel():
    #a.set_xlim(-.1, 6)
    a.set_ylim(-1.5, 1.5)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_sido/l1_{l1}/coeff.pdf', dpi = 300)

#%%
fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"log_scale": False, "lw": 0.7, "markersize": 3}

plot_multiple_error(allQ, ax = ax , label = "saga", ls = '--', **kwargs)
plot_multiple_error(allQ1, ax = ax , label = "adagrad", ls = '--', **kwargs)
plot_multiple_error(allQ2, ax = ax , label = "svrg", ls = '--', **kwargs)
plot_multiple_error(allP, ax = ax , label = "snspp", **kwargs)

ax.set_xlim(-.1, 4)

if l1 == 0.01:
    ax.set_ylim(0.104, 0.114)
else:
    ax.set_ylim(0.086, 0.096)
    
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_sido/l1_{l1}/error.pdf', dpi = 300)

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
