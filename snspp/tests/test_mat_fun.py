"""
author: Fabian Schaipp
"""


import numpy as np
from snspp.matopt.nuclear import smooth_prox, smooth_prox_jacobian

p = 20
q = 30



def test_smooth_prox():
    """ for rho small, the prox is approx. equal to Y"""
    Y = np.random.randn(p,q)
    eps = 1e-10
    rho = 1e-10
    D = smooth_prox(Y, rho, eps)
    assert np.allclose(Y,D)
    
    return
    
def test_smooth_prox_jacobian():
    """for rho small the jacobian is the identity operator """
    Y = np.random.randn(p,q)
    eps = 1e-10; tau = 1e-10
    rho = 1e-10
    for i in range(10):
        H = np.random.randn(p,q)
        Z = smooth_prox_jacobian(Y, rho, eps, tau, H)
        assert np.allclose(Z,H, rtol = 1e-5, atol = 1e-5)
    
    return    

