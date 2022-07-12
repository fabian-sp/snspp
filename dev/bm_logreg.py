"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import copt as cp
from copt.loss import LogLoss
from copt.penalty import L1Norm

from sklearn.linear_model import LogisticRegression

from snspp.solver.opt_problem import problem
from snspp.experiments.experiment_utils import initialize_solvers
from snspp.helper.data_generation import  logreg_test

#%% generate problem

N = 6000
n = 5000
k = 20
l1 = 0.01

f, phi, A, X_train, y_train, _, _, xsol = logreg_test(N, n, k, lambda1 = l1, noise = 0.1, kappa = 15., dist = 'unif')

initialize_solvers(f, phi, A)

N_EPOCHS = 20

#%% solve with scikit
# docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 0, \
                            solver = 'saga', max_iter = N_EPOCHS, verbose = 0)

start = time.time()    
sk.fit(X_train, y_train)
end = time.time()

rt_sk = end-start

x_sk = sk.coef_.copy().squeeze()

print("Objective of final iterate SCIKIT:", f.eval(A@x_sk)+phi.eval(x_sk))
print("Runtime of SCIKIT:", rt_sk)



#%% solve with snspp

params_saga = {'n_epochs' : N_EPOCHS}

start = time.time()
Q = problem(f, phi, A, tol = 0, params = params_saga, verbose = False, measure = False)
Q.solve(solver = 'saga')
end = time.time()

rt_snspp = end-start

print("Objective of final iterate SNSPP:", f.eval(A@Q.x)+phi.eval(Q.x))
print("Runtime of SNSPP:", rt_snspp)

#%% solve with copt
# docs: https://github.com/openopt/copt/blob/master/copt/randomized.py

copt_b = (y_train==1).astype(int)
copt_f = LogLoss(X_train, copt_b)
copt_phi = L1Norm(l1)

ALPHA = Q.params['alpha']

start = time.time()
result_saga = cp.minimize_saga(
    copt_f.partial_deriv,
    X_train,
    copt_b,
    x0 = np.zeros(n),
    prox = copt_phi.prox_factory(n),
    step_size = ALPHA,
    callback = None,
    tol = 0,
    max_iter = N_EPOCHS,
)
end = time.time()


x_copt = result_saga.x
rt_copt = end-start

print("Objective of final iterate COPT:", f.eval(A@x_copt) +phi.eval(x_copt))
print("Runtime of COPT:", rt_copt)

#%% compare solutions


np.linalg.norm(x_copt-x_sk)
np.linalg.norm(Q.x-x_sk)
