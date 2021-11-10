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


N = 1000
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
    params = {'n_epochs' : 1000, 'alpha': 1.}
    
    f, phi, x_sk = create_test_instance(prob = 'lasso')
    template_test(f, phi, x_sk, params, 'saga')
    
    return

def test_adagrad_lasso():
    params = {'n_epochs' : 50}
    
    f, phi, x_sk = create_test_instance(prob = 'lasso')
    template_test(f, phi, x_sk, params, 'adagrad', assert_objective = False)
    
    return

def test_svrg_lasso():
    params = {'n_epochs' : 1000, 'batch_size': 2}
    
    f, phi, x_sk = create_test_instance(prob = 'lasso')
    template_test(f, phi, x_sk, params, 'svrg')
    
    return

#%%

def test_saga_logreg():
    params = {'n_epochs' : 1000, 'alpha': 1.}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    template_test(f, phi, x_sk, params, 'saga')
    
    return

def test_adagrad_logreg():
    params = {'n_epochs' : 50}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    template_test(f, phi, x_sk, params, 'adagrad', assert_objective = False)
    
    return

def test_svrg_logreg():
    params = {'n_epochs' : 1000, 'batch_size': 2}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    template_test(f, phi, x_sk, params, 'svrg')
    
    return

#%%

def test_snspp_lasso():
    params = {'max_iter' : 2000, 'alpha': 1., 'reduce_variance': True}
    
    f, phi, x_sk = create_test_instance(prob = 'lasso')
    template_test(f, phi, x_sk, params, 'snspp')
    
    return

def test_snspp_logreg():
    params = {'max_iter' : 2000, 'alpha': 1., 'reduce_variance': True}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    P = template_test(f, phi, x_sk, params, 'snspp')
    
    return


def test_snspp_novr():
    params = {'reduce_variance': False}
    
    f, phi, x_sk = create_test_instance(prob = 'logreg')
    P = template_test(f, phi, x_sk, params, 'snspp', assert_objective = False)
    
    return


def test_snspp_general():
    
    l1 = 1e-3
    
    xsol, A, b, f, phi, A_test, b_test = lasso_test(100, n, k, l1, block = True, dist = 'ortho')
    params = {'max_iter' : 500, 'alpha': 15., 'reduce_variance': False}
    
    P = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)
    P.solve(solver = 'snspp')
    
    sk = Lasso(alpha = l1*f.N/(2*f.A.shape[0]), fit_intercept = False, tol = 1e-12, max_iter = 1000, selection = 'cyclic')
    sk.fit(A,b)

    obj1 = f.eval(P.x)+phi.eval(P.x)
    obj2 = f.eval(sk.coef_)+phi.eval(sk.coef_)
    assert_almost_equal(obj1, obj2, decimal = 4) 
    
    return
