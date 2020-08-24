import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssnsp.solver.opt_problem import problem, color_dict
from ssnsp.helper.data_generation import get_mnist
from ssnsp.helper.utils import stop_optimal

from sklearn.linear_model import LogisticRegression


f, phi, X_train, y_train, X_test, y_test = get_mnist()

#plt.imshow(X_train[110,:].reshape(28,28))

print("Regularization parameter lambda:", phi.lambda1)

def predict(A,x):
    
    h = np.exp(A@x)
    odds = h/(1+h)
    
    y = (odds >= .5)*2 -1
    
    return y

#%% solve with scikit (SAGA)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                        solver = 'saga', max_iter = 200, verbose = 1)


start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

#(np.sign(predict(X_train, x_sk)) == np.sign(y_train)).sum() / len(y_train)

f.eval(x_sk) + phi.eval(x_sk)

#%% solve with SAGA

params = {'n_epochs' : 100}

Q = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

print(f.eval(Q.x) +phi.eval(Q.x))

#(predict(X_train, Q.x) == y_train).sum()

#%% solve with ADAGRAD

params = {'n_epochs' : 30, 'batch_size': 10, 'gamma': .01}

Q1 = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q1.solve(solver = 'adagrad')

print(f.eval(Q1.x) +phi.eval(Q1.x))

#(predict(X_train, Q1.x) == y_train).sum()

#%% solve with SSNSP

#params = {'max_iter' : 35, 'sample_size': f.N/12, 'sample_style': 'increasing', 'alpha_C' : 10., 'n_epochs': 5}
params = {'max_iter' : 25, 'sample_size': f.N/9, 'sample_style': 'increasing', \
          'alpha_C' : 10., 'reduce_variance': True}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)
P.solve(solver = 'ssnsp')

#%% solve with SSNSP (multiple times)

K = 10
allP = list()
for k in range(K):
    
    P_k = problem(f, phi, tol = 1e-7, params = params, verbose = False, measure = True)
    P_k.solve(solver = 'ssnsp')
    allP.append(P_k.info)

# compute mean and std of objectives
all_obj = np.vstack([allP[k]["objective"] for k in range(K)])
all_mean = all_obj.mean(axis=0)
all_std = all_obj.std(axis=0)

all_rt = np.vstack([allP[k]["runtime"] for k in range(K)]).mean(axis=0).cumsum()

#%% solve with CONSTANT SSNSP

params = {'max_iter' : 30, 'sample_size': 3000, 'sample_style': 'constant', 'alpha_C' : 10., 'n_epochs': 5}

P1 = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P1.solve(solver = 'ssnsp')


#%% coeffcient frame

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x, Q1.x)).T, columns = ['scikit', 'spp', 'saga', 'adagrad'])

#%% objective plot

save = False

fig,ax = plt.subplots(figsize = (7,5))
Q.plot_objective(ax = ax, ls = '--', marker = '<')
Q1.plot_objective(ax = ax, ls = '-.', marker = '>')

#P.plot_objective(ax = ax)
#P1.plot_objective(ax = ax, label = "_constant", marker = "x")

ax.plot(all_rt, all_mean, marker = 'o', color = color_dict["ssnsp"], label = "ssnsp")
ax.fill_between(all_rt, all_mean-all_std, all_mean+all_std, \
                color = color_dict["ssnsp"], alpha = .5)

ax.legend()

#ax.set_yscale('log')

if save:
    fig.savefig(f'data/plots/exp_mnist/obj.pdf', dpi = 300)

#%% coefficent plot

fig,ax = plt.subplots(2, 2,  figsize = (7,5))
Q.plot_path(ax = ax[0,0], xlabel = False)
Q1.plot_path(ax = ax[0,1], xlabel = False, ylabel = False)
P.plot_path(ax = ax[1,0])
P.plot_path(ax = ax[1,1], mean = True, ylabel = False)

for a in ax.ravel():
    a.set_ylim(-.2,.2)
    
plt.subplots_adjust(hspace = 0.33)

if save:
    fig.savefig(f'data/plots/exp_mnist/coeff.pdf', dpi = 300)


#%%

def logreg_error(A, b, x):
    y = predict(A,x)
    
    return (np.sign(y) == np.sign(b)).sum() / len(b)
    

def distance_to_sol(x_hist, x_ref):
    d = list()
    for j in range(x_hist.shape[0]):
        d.append(np.linalg.norm(x_hist[j,:] - x_ref))
        
    return np.array(d)


fig,ax = plt.subplots()

x = Q.info['runtime'].cumsum()
y = distance_to_sol(Q.info['iterates'], x_sk)
plt.plot(x, y, label = Q.solver)

x = Q1.info['runtime'].cumsum()
y = distance_to_sol(Q1.info['iterates'], x_sk)
plt.plot(x, y, label = Q1.solver)

x = P.info['runtime'].cumsum()
y = distance_to_sol(P.info['iterates'], x_sk)
plt.plot(x, y, label = P.solver)


ax.legend()

#%%

logreg_error(X_test, y_test, P.x)


resQ = list()

for j in range(Q.info['iterates'].shape[0]):
    xj = Q.info['iterates'][j,:]
    resQ.append(logreg_error(X_test, y_test, xj))
      
resQ = np.hstack(resQ)


resP = list()

for j in range(P.info['iterates'].shape[0]):
    xj = P.info['iterates'][j,:]
    resP.append(logreg_error(X_test, y_test, xj))
    
resP = np.hstack(resP)    
    

plt.figure()
plt.plot(Q.info['runtime'].cumsum(), resQ)
plt.plot(P.info['runtime'].cumsum(), resP)



