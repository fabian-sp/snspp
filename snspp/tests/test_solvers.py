"""
author: Fabian Schaipp
"""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from sklearn.linear_model import Lasso, LogisticRegression

import warnings
warnings.filterwarnings("ignore")

from snspp.helper.data_generation import lasso_test, logreg_test
from snspp.solver.opt_problem import problem


N = 100
n = 20
k = 5
l1 = 1e-3


def create_test_instance(prob = 'lasso'):
    if prob == 'lasso':
        xsol, A, b, f, phi, _, _ = lasso_test(N, n, k, l1, noise = 0.1, kappa = 10., dist = 'ortho')
        sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-10, max_iter = 1000)
        
    elif prob == 'logreg':
        xsol, A, b, f, phi, _, _ = logreg_test(N, n, k, l1, noise = 0.1, kappa = 10., dist = 'ortho')
        sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-10, solver = 'saga', max_iter = 1000)
    
    sk.fit(A,b)
    x_sk = sk.coef_.copy().squeeze()
    
    return f, phi, x_sk    


def template_test(f, phi, x_sk, params, solver, assert_objective = True):
    
    Q = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)
    Q.solve(solver = solver)
    
    if assert_objective:
        obj1 = f.eval(Q.x)+phi.eval(Q.x)
        obj2 = f.eval(x_sk)+phi.eval(x_sk)
        assert_almost_equal(obj1, obj2, decimal = 4) 
    
    return Q

#%%
def test_saga_lasso():
    #params = dict()
    params = {'n_epochs' : 100, 'alpha': 1.}
    
    f, phi, x_sk = create_test_instance(prob = 'lasso')
    template_test(f, phi, x_sk, params, 'saga')
    
    return

def test_adagrad_lasso():
    params = {'n_epochs' : 100}
    
    f, phi, x_sk = create_test_instance(prob = 'lasso')
    template_test(f, phi, x_sk, params, 'adagrad')
    
    return

def test_svrg_lasso():
    params = {'n_epochs' : 100, 'batch_size': 10}
    
    f, phi, x_sk = create_test_instance(prob = 'lasso')
    template_test(f, phi, x_sk, params, 'svrg')
    
    return

#%%

def test_saga_logreg():
    params = {'n_epochs' : 100, 'alpha': 1.}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    template_test(f, phi, x_sk, params, 'saga')
    
    return

def test_adagrad_logreg():
    params = {'n_epochs' : 100}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    template_test(f, phi, x_sk, params, 'adagrad')
    
    return

def test_svrg_logreg():
    params = {'n_epochs' : 100, 'batch_size': 10}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    template_test(f, phi, x_sk, params, 'svrg')
    
    return

#%%

def test_snspp_lasso():
    params = {'max_iter' : 200, 'alpha': 1., 'reduce_variance': True}
    
    f, phi, x_sk = create_test_instance(prob = 'lasso')
    template_test(f, phi, x_sk, params, 'snspp')
    
    return

def test_snspp_logreg():
    params = {'max_iter' : 300, 'alpha': 10., 'reduce_variance': True}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    P = template_test(f, phi, x_sk, params, 'snspp')
    
    return


def test_snspp_novr():
    params = {'reduce_variance': False}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    P = template_test(f, phi, x_sk, params, 'snspp', assert_objective = False)
    
    return