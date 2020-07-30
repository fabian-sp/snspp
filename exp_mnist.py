import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ssnsp.solver.opt_problem import problem
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

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-4, \
                        solver = 'saga', max_iter = 100, verbose = 1)


start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

f.eval(x_sk) + phi.eval(x_sk)

#%% solve with SAGA

#params = {'n_epochs' : 30, 'batch_size': 10}

params = {'n_epochs' : 100}

Q = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

Q.plot_path()

#(predict(X_train, Q.x) == y_train).sum()

#%% solve with SSNSP

params = {'max_iter' : 15, 'sample_size': f.N/12, 'sample_style': 'increasing', 'alpha_C' : 10., 'n_epochs': 5}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')

P.plot_path()


#%% solve with CONSTANT SSNSP

params = {'max_iter' : 20, 'sample_size': 2000, 'sample_style': 'constant', 'alpha_C' : 10., 'n_epochs': 5}

P1 = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P1.solve(solver = 'ssnsp')


#%% plotting

all_x = pd.DataFrame(np.vstack((x_sk, P.x, Q.x)).T, columns = ['scikit', 'spp', 'saga'])


fig,ax = plt.subplots()
Q.plot_objective(ax = ax)
P.plot_objective(ax = ax)
#P1.plot_objective(ax = ax, label = "ssnsp_constant")

fig,ax = plt.subplots(1,2)
Q.plot_path(ax = ax[0])
P.plot_path(ax = ax[1])
            


#%%

resQ = list()

for j in range(Q.info['iterates'].shape[0]):
    xj = Q.info['iterates'][j,:]
    resQ.append(stop_optimal(xj, f, phi))
    
resQ = np.hstack(resQ)

print("DONE")
resP = list()

for j in range(P.info['iterates'].shape[0]):
    xj = P.info['iterates'][j,:]
    resP.append(stop_optimal(xj, f, phi))
    
resP = np.hstack(resP)    
    

plt.figure()
plt.plot(Q.info['runtime'].cumsum(), resQ)
plt.plot(P.info['runtime'].cumsum(), resP)



