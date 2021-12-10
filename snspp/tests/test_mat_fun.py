"""
author: Fabian Schaipp
"""


import numpy as np
from snspp.matopt.nuclear import prox_nuclear, prox_nuclear_jacobian

p = 20
q = 30

RHO_SMALL = 1e-10

def test_smooth_prox():
    """ for rho small, the prox is approx. equal to Y"""
    Y = np.random.randn(p,q)
    eps = 1e-10
    D = prox_nuclear(Y, RHO_SMALL, eps)
    assert np.allclose(Y, D)
    
    return
    
def test_smooth_prox_jacobian():
    """for rho small the jacobian is the identity operator """
    Y = np.random.randn(p,q)
    eps = 1e-10; tau = 1e-10
    for i in range(10):
        H = np.random.randn(p,q)
        Z = prox_nuclear_jacobian(Y, RHO_SMALL, eps, tau, H)
        assert np.allclose(Z,H, rtol = 1e-5, atol = 1e-5)
    
    return    

def test_nonsmooth_prox():
    """for rho small the prox is approx. equal to Y"""
    Y = np.random.randn(p,q)
    D = prox_nuclear(Y, RHO_SMALL, 0.)
    assert np.allclose(Y, D)
    
    return
    
def test_nonsmooth_prox_jacobian():
    """for rho small the jacobian is the identity operator """
    Y = np.random.randn(p,q)
    for i in range(10):
        H = np.random.randn(p,q)
        Z = prox_nuclear_jacobian(Y, RHO_SMALL, 0., 0., H)
        assert np.allclose(Z, H, rtol = 1e-5, atol = 1e-5)
    
    return    
