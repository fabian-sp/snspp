import numpy as np
from numba.experimental import jitclass
from numba import int64, float32, float64, typeof
from numba.typed import List



#%% squared hinge loss 

spec_log = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('A', float64[:,:]),
    ('N', int64), 
    ('m', int64[:]),
]
        

@jitclass(spec_log)
class squared_hinge_loss:
    """ 
    f(x) = (1/N) \sum max(0, 1-b_i*A_i@x)**2
    
    f_i is the squared hinge loss function i.e. f_i(z) = max(0,1-t)**2 
    """
    
    def __init__(self, A, b):
        self.name = 'squared_hinge'
        self.convex = True
        
        self.b = b
        self.A = A * np.ascontiguousarray(self.b).reshape((-1,1))
        self.N = len(self.b)
        self.m = np.repeat(1,self.N)
        
        return
    
    def eval(self, x):
        """
        method for evaluating f(x)
        """
        z = self.A@x
        y = (np.maximum(0,1-z)**2).sum()
         
        return (1/self.N)*y

    def f(self, x, i):
        """
        evaluate f_i(x)
        """
        return np.maximum(0,1-x)**2
    
    def g(self, x, i):
        
        return (x<=1)*(2*x-2)
        
    def fstar(self, X, i):
        Y = np.zeros_like(X)
        zz = np.less_equal(X,0)
        Y = X*(1+X/4)
        Y[~zz] = np.inf
        return Y
    
    def gstar(self, X, i):
        Y = np.zeros_like(X)
        zz = np.less_equal(X,0)
        Y = 1+X/2
        Y[~zz] = np.inf
        return Y
    
    
    def Hstar(self, X, i):
        Y = 1/2 * np.ones_like(X)
        zz = np.less_equal(X,0)
        Y[~zz] = np.inf
        return Y
    
    def fstar_vec(self, x, S):
        zz = np.less_equal(x,0)
        y = x*(1+x/4)
        y[~zz] = np.inf
        return y
    
    def gstar_vec(self, x, S):
        zz = np.less_equal(x,0)
        y = 1+x/2 
        y[~zz] = np.inf
        return y
    
    def Hstar_vec(self, x, S):
        zz = np.less_equal(x,0)
        y = 1/2 * np.ones_like(x)
        y[~zz] = np.inf
        return y