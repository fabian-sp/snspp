import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ssnsp.solver.opt_problem import problem
from ssnsp.helper.data_generation import get_gisette
from ssnsp.helper.utils import stop_optimal

from sklearn.linear_model import LogisticRegression


f, phi, X_train, y_train, X_test, y_test = get_gisette(lambda1 = 0.1)


print("Regularization parameter lambda:", phi.lambda1)

def predict(A,x):
    
    h = np.exp(A@x)
    odds = h/(1+h)
    
    y = (odds >= .5)*2 -1
    
    return y

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                        solver = 'saga', max_iter = 50, verbose = 1)


start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

#(np.sign(predict(X_test, x_sk)) == np.sign(y_test)).sum() / len(y_test)

f.eval(x_sk) + phi.eval(x_sk)

#%% solve with SAGA

params = {'n_epochs' : 100}

Q = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))

#(predict(X_train, Q.x) == y_train).sum()

#%% solve with ADAGRAD

params = {'n_epochs' : 30, 'batch_size': 20, 'gamma': .005}

Q1 = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))

#(predict(X_train, Q1.x) == y_train).sum()


#%% solve with SSNSP

#params = {'max_iter' : 35, 'sample_size': f.N/12, 'sample_style': 'increasing', 'alpha_C' : 10., 'n_epochs': 5}
params = {'max_iter' : 25, 'sample_size': f.N/2, 'sample_style': 'fast_increasing', 'alpha_C' : 10.}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')


#%% solve with CONSTANT SSNSP

params = {'max_iter' : 30, 'sample_size': 3000, 'sample_style': 'constant', 'alpha_C' : 10., 'n_epochs': 5}

P1 = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P1.solve(solver = 'ssnsp')


#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad'])

#%% plotting

save = False

fig,ax = plt.subplots(figsize = (7,5))
Q.plot_objective(ax = ax, ls = '--', marker = '<')
Q1.plot_objective(ax = ax, ls = '-.', marker = '>')
P.plot_objective(ax = ax)


P1.plot_objective(ax = ax, label = "_constant", marker = "x")



if save:
    fig.savefig(f'data/plots/exp_gisette/obj.pdf', dpi = 300)


fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
P.plot_path(ax = ax[1,0])
P.plot_path(ax = ax[1,1], mean = True, ylabel = False)

for a in ax.ravel():
    a.set_ylim(-1.,1.)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_gisette/coeff.pdf', dpi = 300)






