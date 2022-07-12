"""
author: Fabian Schaipp

This file contains loss function objects which are then used in the optimization algorithms. 
"""

import numpy as np
from numba.experimental import jitclass
from numba import int64, float64, typeof


#%% squared loss

# specification of types
spec = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),
    ('N', int64), 
    ('m', int64[:]),
]
        

@jitclass(spec)
class lsq:
    """ 
    Let :math:``A \in \mathbb{R}^{N\times n}`` and :math:``b \in \mathbb{R}^{N}`` be given.
    This implements the squared loss function given by
    
    .. math:
        f(x) = \frac{1}{N} \|Ax-b\|^2.
        
    Hence, we have
    
    .. math:
        f_i(z) = |z-b_i|^2.
        
    The convex conjugate is given by
    
    .. math:
        f_i^\ast(z) = \frac{1}{4} \|z\|^2 + b_i\cdot z.
    
    """
    
    def __init__(self, b):
        self.name = 'squared'
        self.convex = True 
        self.b = b
        self.N = len(b)
        self.m = np.repeat(1,self.N)
        
        return
    
    def eval(self, z):
        """
        Method for evaluating :math:`f(x)`.
        The array ``x`` should be the same type as A (we use float64).
        """
        return (1/self.N) * np.linalg.norm(z - self.b)**2      
    
    def f(self, x, i):
        """
        Method for evaluating :math:`f_i(x)`.
        """
        return (x - self.b[i])**2
    
    def g(self, x, i):
        """
        Method for evaluating :math:`f_i'(x)`.
        """
        return 2 * (x - self.b[i])
    
    # these are actually never used (see vectorized versions below instead)
    def fstar(self, x, i):
        return .25 * np.linalg.norm(x)**2 + self.b[i] * x  
    def gstar(self, x, i):
        return .5 * x + self.b[i]  
    def Hstar(self, x, i):
        return .5
    
    # vectorized versions, they return a vector where each element is fstar/gstar/Hstar of one sample
    def fstar_vec(self, x, S):
        return .25 * x**2 + self.b[S] * x
    def gstar_vec(self, x, S):
        return .5 * x + self.b[S]
    def Hstar_vec(self, x, S):
        return .5*np.ones_like(x)

#%% logistic loss

# specification of types
spec_log = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('N', int64), 
    ('m', int64[:]),
]

@jitclass(spec_log)
class logistic_loss:
    """ 
    Let :math:``A \in \mathbb{R}^{N\times n}`` and :math:``b \in \{-1,1\}^{N}`` be given.
    This implements the logistic loss function given by
    
    .. math:
        f(x) = \frac{1}{N} \sum_{i=1}^{N} \ln(1+\exp(-b_i\cdot(a_i x)))
    
    where :math:`a_i` is the i-th row of :math:`A`.
    
    Using the row-wise multiplication :math`b \cdot A`, we have
    
    .. math:
        f_i(z) = \ln(1+\exp(-z))
        
    The convex conjugate is given by
    
    .. math:
        f_i^\ast(z) = \begin{cases} -z\ln(-z)+ (1+z)\ln(1+z) \quad z\in (-1,0)\\ +\infty \quad text{else} \end{cases}.
    
    """
    
    def __init__(self, b):
        self.name = 'logistic'
        self.convex = True   
        self.b = b
        self.N = len(self.b)
        self.m = np.repeat(1,self.N)   
        return
    
    def eval(self, z):
        """
        Method for evaluating :math:`f(x)`.
        """
        y = np.log(1+ np.exp(-z)).sum()        
        return (1/self.N)*y

    def f(self, x, i):
        """
        Method for evaluating :math:`f_i(x)`.
        """
        return np.log(1+np.exp(-x))   
    
    def g(self, x, i):
        """
        Method for evaluating :math:`f_i'(x)`.
        """
        return -1/(1+np.exp(x)) 
     
    # these are actually never used (see vectorized versions below instead)
    def fstar(self, X, i):
        Y = np.zeros_like(X)
        zz = np.logical_and(X < 0 , X > -1)
        Y = -X*np.log(-X) + (1+X) * np.log(1+X)
        Y[~zz] = np.inf            
        return Y
    def gstar(self, X, i):
        Y = np.zeros_like(X)
        zz = np.logical_and(X < 0 , X > -1)
        Y = np.log(-(1+X)/X) 
        Y[~zz] = np.inf    
        return Y
    def Hstar(self, X, i):
        Y = np.zeros_like(X)
        zz = np.logical_and(X < 0 , X > -1)
        Y = -1/(X**2+X)
        Y[~zz] = np.inf    
        return Y
    
    # vectorized versions, they return a vector where each element is fstar/gstar/Hstar of one sample
    def fstar_vec(self, x, S):
        zz = np.logical_and(x < 0 , x > -1)
        y = -x*np.log(-x) + (1+x) * np.log(1+x)
        y[~zz] = np.inf
        return y
    def gstar_vec(self, x, S):
        zz = np.logical_and(x < 0 , x > -1)
        y = np.log(-(1+x)/x) 
        y[~zz] = np.inf
        return y
    def Hstar_vec(self, x, S):
        zz = np.logical_and(x < 0 , x > -1)
        y = -1/(x**2+x)
        y[~zz] = np.inf
        return y

    
#%% only needed for testing

# class block_lsq:
#     """ 
#     f is the squared loss function (1/N) * ||Ax-b||**2
#     but with block-wise splits
#     """
    
#     def __init__(self, A, b, m):
#         self.name = 'squared'
#         self.b = b
#         self.A = A
#         self.N = len(m)
#         self.m = m
#         self.ixx = np.repeat(np.arange(self.N), self.m)
#         self.convex = True
        
#     def eval(self, x):
#         y = 0
#         for i in np.arange(self.N):
#             z_i = self.A[self.ixx == i, :] @ x
#             y += self.f(z_i, i)
        
#         return (1/self.N)*y

#     def f(self, x, i):
#         return np.linalg.norm(x - self.b[self.ixx == i])**2
    
#     def g(self, x, i):
#         return 2 * (x - self.b[self.ixx == i])
    
#     def fstar(self, x, i):
#         return .25 * np.linalg.norm(x)**2 + np.sum(self.b[self.ixx == i] * x)
    
#     def gstar(self, x, i):
#         return .5 * x + self.b[self.ixx == i]
    
#     def Hstar(self, x, i):
#         return .5 * np.eye(self.m[i])


