import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ssnsp.solver.opt_problem import problem
from ssnsp.helper.data_generation import get_gisette
from ssnsp.experiments.experiment_utils import params_tuner, plot_multiple, plot_multiple_error, initialize_solvers, eval_test_set


from sklearn.linear_model import LogisticRegression


f, phi, X_train, y_train, X_test, y_test = get_gisette(lambda1 = 0.05)


print("Regularization parameter lambda:", phi.lambda1)

def predict(A,x):
    
    h = np.exp(A@x)
    odds = h/(1+h)  
    y = (odds >= .5)*2 -1
    
    return y

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                        solver = 'saga', max_iter = 200, verbose = 0)

start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

#(np.sign(predict(X_test, x_sk)) == np.sign(y_test)).sum() / len(y_test)

psi_star = f.eval(x_sk) + phi.eval(x_sk)
print("psi(x*) = ", psi_star)
initialize_solvers(f, phi)

#%% params

params_saga = {'n_epochs' : 50, 'gamma': 5.5}

params_svrg = {'n_epochs' : 50, 'batch_size': 1, 'gamma': 12.}


params_adagrad = {'n_epochs' : 200, 'batch_size': 240, 'gamma': 0.028}


params_ssnsp = {'max_iter' : 60, 'batch_size': 400, 'sample_style': 'fast_increasing', 'alpha_C' : 7.,\
          "reduce_variance": True}

#params_tuner(f, phi, solver = "saga", gamma_range = np.linspace(4,8, 10))
#params_tuner(f, phi, solver = "svrg", gamma_range = np.linspace(1, 20, 10), batch_range = np.array([1, 10, 100]))
#params_tuner(f, phi, solver = "adagrad", batch_range = np.array([50, 250, 500]))
#params_tuner(f, phi, solver = "ssnsp", gamma_range = np.linspace(5,10, 10), batch_range = np.array([200, 400]))

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

print(f.eval(Q2.x) +phi.eval(Q2.x))

#%% solve with SSNSP

P = problem(f, phi, tol = 1e-9, params = params_ssnsp, verbose = True, measure = True)

P.solve(solver = 'ssnsp')

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

#%% solve with SSNSP (multiple times, no VR)

# params1 = params_ssnsp.copy()
# params1["reduce_variance"] = False

# allP1 = list()
# for k in range(K):
    
#     P_k = problem(f, phi, tol = 1e-9, params = params1, verbose = False, measure = True)
#     P_k.solve(solver = 'ssnsp')
#     allP1.append(P_k)

#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad'])

#%% objective plot

save = False

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True, "lw": 0.4, "markersize": 3}

#Q.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
#Q1.plot_objective(ax = ax, ls = '-.', marker = '>', **kwargs)
#Q2.plot_objective(ax = ax, ls = '-.', marker = '<', **kwargs)
#P.plot_objective(ax = ax, **kwargs)


plot_multiple(allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
plot_multiple(allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
plot_multiple(allP, ax = ax , label = "ssnsp", **kwargs)
#plot_multiple(allP1, ax = ax , label = "ssnsp_noVR", name = "ssnsp (no VR)", **kwargs)


ax.set_xlim(-.1, 6)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_gisette/obj.pdf', dpi = 300)

#%% coeffcient plot

P = allP[-1]

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
P.plot_path(ax = ax[1,0])
P.plot_path(ax = ax[1,1], mean = True, ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.5,.3)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_gisette/coeff.pdf', dpi = 300)


#%% eval test set loss

def logreg_loss(x, A, b):
    z = A@x
    return np.log(1 + np.exp(-b*z)).mean()

kwargs2 = {"A": X_test, "b": y_test}

all_loss_P = np.vstack([eval_test_set(X = P.info["iterates"], loss = logreg_loss, **kwargs2) for P in allP])
all_loss_Q = np.vstack([eval_test_set(X = Q.info["iterates"], loss = logreg_loss, **kwargs2) for Q in allQ])
all_loss_Q1 = np.vstack([eval_test_set(X = Q.info["iterates"], loss = logreg_loss, **kwargs2) for Q in allQ1])


#%%
fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"log_scale": False, "lw": 1, "markersize": 3}

plot_multiple_error(all_loss_Q, allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
plot_multiple_error(all_loss_Q1, allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
plot_multiple_error(all_loss_P, allP, ax = ax , label = "ssnsp", **kwargs)

ax.set_xlim(-.1, 6)
ax.set_ylim(all_loss_P.min()-1e-3, all_loss_P.min()+1e-1)
ax.legend(fontsize = 10)

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_gisette/error.pdf', dpi = 300)




