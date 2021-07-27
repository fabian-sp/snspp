"""
author: Fabian Schaipp
"""
import numpy as np
from numba.experimental import jitclass
from numba import float64, typeof


#%% l1 norm regularizer

spec_l1 = [
    ('name', typeof('abc')),
    ('lambda1', float64)
]

@jitclass(spec_l1)
class L1Norm:
    """
    class for the regularizer x --> lambda1 ||x||_1
    """
    def __init__(self, lambda1):
        assert lambda1 > 0 
        self.name = '1norm'
        self.lambda1 = lambda1
        
    def eval(self, x):
        return self.lambda1 * np.linalg.norm(x, 1)
    
    def subg(self, x):
        return self.lambda1 * np.sign(x)
    
    def prox(self, x, alpha):
        """
        calculates prox_{alpha*phi}(x)
        """
        assert alpha > 0
        l = alpha * self.lambda1
        return np.sign(x) * np.maximum( np.abs(x) - l, 0.)
    
    def adagrad_prox(self, x, L):
        """
        calculates prox_{phi}^Lambda_t (x)
        L is the diagonal(!) of Lambda_t (in the notation of Milzarek et al.)
        """
             
        l = np.divide(self.lambda1 * np.ones_like(L), L) 
        
        return np.sign(x) * np.maximum( np.abs(x) - l, 0.)
    
    def jacobian_prox(self, x, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        
        d = np.ones_like(x)
        d[np.abs(x) <= l] = 0.
        
        # actual result is np.diag(d), but saving memory!
        return d
    
    def moreau(self, x, alpha):
        assert alpha > 0
        z = self.prox(x, alpha)
        return alpha*self.eval(z) + .5 * np.linalg.norm(z-x)**2


#%% 0-function for unregularized problems

spec_zero = [
    ('name', typeof('abc'))
]

@jitclass(spec_zero)
class Zero:
    """
    class for the zero function x --> 0
    useful if you want to solve unregularized problems
    """
    def __init__(self):
        self.name = 'zero'
        
    def eval(self, x):
        return 0.
    
    def prox(self, x, alpha):
        """
        calculates prox_{alpha*phi}(x)
        """
        assert alpha > 0
        return x
    
    def adagrad_prox(self, x, L):
        """
        calculates prox_{phi}^Lambda_t (x)
        L is the diagonal(!) of Lambda_t (in the notation of Milzarek et al.)
        """          
        return x
         
    def jacobian_prox(self, x, alpha):
        assert alpha > 0
        return np.eye(len(x))
    
    def moreau(self, x, alpha):
        assert alpha > 0
        return 0.


#%% ridge regularizer

spec_ridge = [
    ('name', typeof('abc')),
    ('lambda1', float64)
]

@jitclass(spec_ridge)
class Ridge:
    """
    class for the ridge regularizer x --> lambda1 ||x||_2^2
    """
    def __init__(self, lambda1):
        assert lambda1 > 0 
        self.name = 'ridge'
        self.lambda1 = lambda1
        
    def eval(self, x):
        return self.lambda1 * np.linalg.norm(x, 2)**2
    
    def prox(self, x, alpha):
        """
        calculates prox_{alpha*phi}(x)
        """
        assert alpha > 0
        l = alpha * self.lambda1
        return x/(1+2*l)
    
    def adagrad_prox(self, x, L):
        """
        calculates prox_{phi}^Lambda_t (x)
        """
             
        return 1/(L+2*self.lambda1)*L*x
        
    
    def jacobian_prox(self, x, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        
        return 1/(1+2*l)*np.eye(len(x))
    
    def moreau(self, x, alpha):
        assert alpha > 0
        z = self.prox(x, alpha)
        return alpha*self.eval(z) + .5 * np.linalg.norm(z-x)**2
