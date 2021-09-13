"""
author: Fabian Schaipp

This file contains regularization function objects which are then used in the optimization algorithms. 
"""
import numpy as np
from numba.experimental import jitclass
from numba import float64, typeof


#%% l1 norm

# specification of types
spec_l1 = [
    ('name', typeof('abc')),
    ('lambda1', float64)
]

@jitclass(spec_l1)
class L1Norm:
    """
    This implements the L1-norm regularizer given by
    
    .. math:
        \varphi(x) = \lambda_1 \|x\|_1.
        
    """
    def __init__(self, lambda1):
        assert lambda1 > 0 
        self.name = '1norm'
        self.lambda1 = lambda1
        
    def eval(self, x):
        """
        Method for evaluating ``phi(x)``.
        """
        return self.lambda1 * np.linalg.norm(x, 1)
    
    # only for testing
    def subg(self, x):
        return self.lambda1 * np.sign(x)
    
    def prox(self, x, alpha):
        """
        Calculates :math:`\text{prox}_{\alpha \varphi}(x)`.
        """
        assert alpha > 0
        l = alpha * self.lambda1
        return np.sign(x) * np.maximum( np.abs(x) - l, 0.)
    
    def adagrad_prox(self, x, L):
        """
        Given a vector ``L`` with positive entries, this calculates
        
        .. math:
            \arg \min_z \varphi(z) + \frac{1}{2}(z-x)^T\mathrm{diag}(L)(z-x).
        
        """            
        l = np.divide(self.lambda1 * np.ones_like(L), L)        
        return np.sign(x) * np.maximum( np.abs(x) - l, 0.)
    
    def jacobian_prox(self, x, alpha):
        """
        Calculates one element in the Clarke differential of the proximal operator, i.e. 
        .. math:
            Z \in \partial \text{prox}_{\alpha \varphi}(x).
            
        Here, the actual result is a diagonal matrix, but in order to save memory we only return the diagonal itself.
        """
        assert alpha > 0
        l = alpha * self.lambda1
        
        d = np.ones_like(x)
        d[np.abs(x) <= l] = 0.
        
        # actual result is np.diag(d), but saving memory!
        return d
    
    def moreau(self, x, alpha):
        """
        Evaluates the Moreau envelope of :math:`\alpha \varphi` at `x`.
        """
        assert alpha > 0
        z = self.prox(x, alpha)
        return alpha*self.eval(z) + .5 * np.linalg.norm(z-x)**2


#%% 0-function

# specification of types
spec_zero = [
    ('name', typeof('abc'))
]

@jitclass(spec_zero)
class Zero:
    """
    This implements the zero-function
    
    .. math:
        \varphi(x) = 0.
    
    This class is useful if you want to solve unregularized problems.
    """
    def __init__(self):
        self.name = 'zero'
        
    def eval(self, x):
        return 0.
    
    def prox(self, x, alpha):
        return x
    
    def adagrad_prox(self, x, L):
        return x
         
    def jacobian_prox(self, x, alpha):
        return np.eye(len(x))
    
    def moreau(self, x, alpha):
        return 0.


#%% ridge regularizer

# specification of types
spec_ridge = [
    ('name', typeof('abc')),
    ('lambda1', float64)
]

@jitclass(spec_ridge)
class Ridge:
    """
    This implements the squared 2-norm regularizer, given by
    
    .. math:
        \varphi(x) = \lambda_1 \|x\|_2^2.
    
    The naming is inspired by ridge regression where this regularizer is (famously) used.
    """
    def __init__(self, lambda1):
        assert lambda1 > 0 
        self.name = 'ridge'
        self.lambda1 = lambda1
        
    def eval(self, x):
        return self.lambda1 * np.linalg.norm(x, 2)**2
    
    def prox(self, x, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        return x/(1+2*l)
    
    def adagrad_prox(self, x, L):
        return 1/(L+2*self.lambda1)*L*x
        
    def jacobian_prox(self, x, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        return 1/(1+2*l)*np.eye(len(x))
    
    def moreau(self, x, alpha):
        assert alpha > 0
        z = self.prox(x, alpha)
        return alpha*self.eval(z) + .5 * np.linalg.norm(z-x)**2
