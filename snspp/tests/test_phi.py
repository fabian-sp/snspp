"""
author: Fabian Schaipp
"""


import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from snspp.helper.regz import L1Norm, Ridge, Zero
# for alpha small --> prox_phi(x) = x, Dprox = Id

# TODO : adagrad_prox = prox for L = ones
    
def template_prox_id(phi, x, alpha = 1e-10):
    p = len(x)
    
    y = phi.prox(x, alpha)        
    assert_array_almost_equal(x, y)

    G = phi.jacobian_prox(x, alpha)
    if phi.name == '1norm':
        assert_array_almost_equal(G, np.ones(p))
    else:
        assert_array_almost_equal(G, np.eye(p))
        
    return    

def template_phi(phi, alpha = 1e-10):
    x = np.random.randn(10)
    
    phi.eval(x)   
    phi.prox(x, alpha)   
    L = np.ones(10)     
    phi.adagrad_prox(x, L)
    phi.jacobian_prox(x, alpha) 
    phi.moreau(x, alpha) 
    return    

#%%

x = np.random.randn(20)

def test_1norm(): 
    phi = L1Norm(1)
    template_prox_id(phi, x, alpha = 1e-10) 
    return

def test_Ridge():
    phi = Ridge(1)
    template_prox_id(phi, x, alpha = 1e-10) 
    return    
    
def test_Zero():
    phi = Zero()
    template_prox_id(phi, x, alpha = 1e-10) 
    return
 