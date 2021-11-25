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

xsol, A, b, f, phi, _, _ = logreg_test(N, n, k, lambda1 = l1, noise = 0.1, kappa = 15., dist = 'unif')


initialize_solvers(f, phi)

N_EPOCHS = 20

L = .25 * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = A)**2).max()
ALPHA = 1/(3*L)


#%% solve with scikit
# docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 0, \
                            solver = 'saga', max_iter = N_EPOCHS, verbose = 0)

start = time.time()    
sk.fit(A, b)
end = time.time()

rt_sk = end-start

x_sk = sk.coef_.copy().squeeze()

print("Objective of final iterate SCIKIT:", f.eval(x_sk)+phi.eval(x_sk))
print("Runtime of SCIKIT:", rt_sk)

#%% solve with copt
# docs: https://github.com/openopt/copt/blob/master/copt/randomized.py

copt_b = (b==1).astype(int)
copt_f = LogLoss(A, copt_b)
copt_phi = L1Norm(l1)


start = time.time()
result_saga = cp.minimize_saga(
    copt_f.partial_deriv,
    A,
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

print("Objective of final iterate COPT:", f.eval(x_copt) +phi.eval(x_copt))
print("Runtime of COPT:", rt_copt)

#%% solve with snspp

params_saga = {'n_epochs' : N_EPOCHS, 'alpha' : 1.}

start = time.time()
Q = problem(f, phi, tol = 0, params = params_saga, verbose = False, measure = False)
Q.solve(solver = 'saga')
end = time.time()

rt_snspp = end-start

print("Objective of final iterate SNSPP:", f.eval(Q.x)+phi.eval(Q.x))
print("Runtime of SNSPP:", rt_snspp)

#%% compare solutions


np.linalg.norm(x_copt-x_sk)
np.linalg.norm(Q.x-x_sk)
