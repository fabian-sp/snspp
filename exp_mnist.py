import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssnsp.solver.opt_problem import problem, color_dict
from ssnsp.helper.data_generation import get_mnist
from ssnsp.experiments.experiment_utils import plot_multiple, adagrad_step_size_tuner, initialize_solvers

from sklearn.linear_model import LogisticRegression


f, phi, X_train, y_train, X_test, y_test = get_mnist()

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

psi_star = f.eval(x_sk) + phi.eval(x_sk)
print("psi(x*) = ", psi_star)

initialize_solvers(f, phi)

#%% solve with SAGA

params_saga = {'n_epochs' : 20, 'gamma': 40.}

Q = problem(f, phi, tol = 1e-9, params = params_saga, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x)+phi.eval(Q.x))

#%% solve with ADAGRAD

#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = None)
opt_gamma = 0.0123 #0.01873

params_adagrad = {'n_epochs' : 200, 'batch_size': int(f.N*0.05), 'gamma': opt_gamma}

Q1 = problem(f, phi, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x)+phi.eval(Q1.x))

#%% solve with SSNSP

# params setup for decreasing step size
# params = {'max_iter' : 70, 'sample_size': 1000, 'sample_style': 'fast_increasing', \
#           'alpha_C' : 15., 'reduce_variance': True}
params_ssnsp = {'max_iter' : 70, 'sample_size': 700, 'sample_style': 'fast_increasing', \
          'alpha_C' : 10., 'reduce_variance': True}

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

K = 20
allQ1 = list()
for k in range(K):
    
    Q1_k = problem(f, phi, tol = 1e-9, params = params_adagrad, verbose = True, measure = True)
    Q1_k.solve(solver = 'adagrad')
    allQ1.append(Q1_k)
    
#%% solve with SSNSP (multiple times, VR)

K = 20
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
    
#     P_k = problem(f, phi, tol = 1e-7, params = params1, verbose = False, measure = True)
#     P_k.solve(solver = 'ssnsp')
#     allP1.append(P_k)


#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad'])

#%% objective plot

save = False

fig,ax = plt.subplots(figsize = (4.5, 3.5))

kwargs = {"psi_star": psi_star, "log_scale": True}

#Q.plot_objective(ax = ax, ls = '--', marker = '<', **kwargs)
#Q1.plot_objective(ax = ax, ls = '-.', marker = '>', **kwargs)
#P.plot_objective(ax = ax, **kwargs)


plot_multiple(allQ, ax = ax , label = "saga", ls = '--', marker = '<', **kwargs)
plot_multiple(allQ1, ax = ax , label = "adagrad", ls = '--', marker = '>', **kwargs)
plot_multiple(allP, ax = ax , label = "ssnsp", **kwargs)

#plot_multiple(allP1, ax = ax , label = "ssnsp_noVR", name = "ssnsp (no VR)", **kwargs)

ax.set_xlim(-1,16)
ax.legend()

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_mnist/obj.pdf', dpi = 300)

#%% coefficent plot

P = allP[-1]

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
P.plot_path(ax = ax[1,0])
P.plot_path(ax = ax[1,1], mean = True, ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.25,.25)
    
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



