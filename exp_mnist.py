import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ssnsp.solver.opt_problem import problem
from ssnsp.helper.data_generation import get_mnist

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
                        solver = 'saga', max_iter = 40, verbose = 1)


start = time.time()
sk.fit(X_train, y_train)
end = time.time()

print(f"Computing time: {end-start} sec")

x_sk = sk.coef_.copy().squeeze()

f.eval(x_sk) + phi.eval(x_sk)

#%% solve with SAGA

params = {'n_epochs' : 80}

Q = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)

Q.solve(solver = 'saga')

Q.plot_path()

#(predict(X_train, Q.x) == y_train).sum()

#%% solve with SSNSP

params = {'max_iter' : 15, 'sample_size': f.N/8, 'sample_style': 'increasing', 'alpha_C' : 10., 'n_epochs': 5}

P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)

P.solve(solver = 'ssnsp')

P.plot_path()


#%% solve with FULL SSNSP

params = {'max_iter' : 15, 'sample_size': f.N/10, 'sample_style': 'constant', 'alpha_C' : 10., 'n_epochs': 5}

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
            
# x = Q.info['runtime'].cumsum()
# y = Q.info['objective']

# ax.plot(x,y, '-o', label = 'SAGA')

# x = P.info['runtime'].cumsum()
# y = P.info['objective']

# ax.plot(x,y, '-o', label = 'SSNSP')

# ax.legend()
# ax.set_xlabel('Runtime')
# ax.set_ylabel('Objective')

#ax.set_yscale('log')
