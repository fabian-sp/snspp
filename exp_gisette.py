import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ssnsp.solver.opt_problem import problem
from ssnsp.helper.data_generation import get_gisette
from ssnsp.experiments.experiment_utils import plot_multiple, adagrad_step_size_tuner, initialize_fast_gradient


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
                        solver = 'saga', max_iter = 100, verbose = 1)

start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

#(np.sign(predict(X_test, x_sk)) == np.sign(y_test)).sum() / len(y_test)

psi_star = f.eval(x_sk) + phi.eval(x_sk)
print("psi(x*) = ", psi_star)
initialize_fast_gradient(f, phi)

#%% solve with SAGA

params = {'n_epochs' : 50, 'gamma': 6.}

Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))

#%% solve with ADAGRAD

#opt_gamma,_,_ = adagrad_step_size_tuner(f, phi, gamma_range = None, params = None)
opt_gamma = 0.02

params = {'n_epochs' : 200, 'batch_size': 240, 'gamma': opt_gamma}

Q1 = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x)+phi.eval(Q1.x))

#%% solve with SSNSP

# params setup for decreasing step size
# params = {'max_iter' : 50, 'sample_size': 500, 'sample_style': 'fast_increasing', 'alpha_C' : 30.,\
#           "reduce_variance": True}
params = {'max_iter' : 50, 'sample_size': 500, 'sample_style': 'fast_increasing', 'alpha_C' : 5.,\
          "reduce_variance": True}

P = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')

#%% solve with SSNSP (multiple times, VR)

params = {'max_iter' : 50, 'sample_size': 500, 'sample_style': 'fast_increasing', 'alpha_C' : 5.,\
          "reduce_variance": True}

K = 20
allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-9, params = params, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k)

#%% solve with SSNSP (multiple times, no VR)

params1 = params.copy()
params1["reduce_variance"] = False

allP1 = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-9, params = params1, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP1.append(P_k)

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
#P1.plot_objective(ax = ax, label = " constant", marker = "x")

ax.set_xlim(-.1,6)
ax.legend(fontsize = 10)
#ax.set_yscale('log')

fig.subplots_adjust(top=0.96,
                    bottom=0.14,
                    left=0.165,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)

if save:
    fig.savefig(f'data/plots/exp_gisette/obj.pdf', dpi = 300)

#%% coeffcient plot

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






