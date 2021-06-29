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
    
#%% Huber loss 

spec_log = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('A', float64[:,:]),
    ('mu', float64[:]),
    ('N', int64), 
    ('m', int64[:]),
]
        

@jitclass(spec_log)
class huber_loss:
    """ 
    f(x) = (1/N) \sum phi_i(A_i@x -b_i)
    
    where 
    phi_i is the Huber loss function i.e.
        phi_i(z) = z^2/(2*mu_i) if |z| <= mu_i
                 = |z| - mu/2 else
        
    """
    
    def __init__(self, A, b, mu):
        self.name = 'huber'
        self.convex = True
        
        self.b = b
        self.A = A * np.ascontiguousarray(self.b).reshape((-1,1))
        self.mu = mu
        self.N = len(self.b)
        self.m = np.repeat(1,self.N)
        
        return
    
    def eval(self, x):
        """
        method for evaluating f(x)
        """
        z = self.A@x
        y = 0
        for i in np.arange(self.N):
            y += self.f(z[i],i)
         
        return (1/self.N)*y

    def f(self, x, i):
        """
        evaluate f_i(x)
        """
        if np.abs(x-self.b[i]) <= self.mu[i]:
            y = (x-self.b[i])**2/(2*self.mu[i])
        else:
            y = np.abs(x-self.b[i]) - self.mu[i]/2
        return y
    
    def g(self, x, i):
        ixx = np.abs(x-self.b[i]) <= self.mu[i]
        return (1-ixx)*np.sign(x-self.b[i])  + ixx*((x-self.b[i])/self.mu[i])
        
    def fstar(self, X, i):
        Y = np.zeros_like(X)
        zz = np.less_equal(np.abs(X),1)
        Y = 0.5*self.mu[i]*X**2 +self.b[i]*X
        Y[~zz] = np.inf
        return Y
    
    def gstar(self, X, i):
        Y = np.zeros_like(X)
        zz = np.less_equal(np.abs(X),1)
        Y = self.mu[i]*X + self.b[i]
        Y[~zz] = np.inf
        return Y
     
    def Hstar(self, X, i):
        Y = self.mu[i] * np.ones_like(X)
        zz = np.less_equal(np.abs(X),1)
        Y[~zz] = np.inf
        return Y
    
    def fstar_vec(self, x, S):
        zz = np.less_equal(np.abs(x),1)
        y = 0.5*self.mu[S]*x**2 + self.b[S]*x
        y[~zz] = np.inf
        return y
    
    def gstar_vec(self, x, S):
        zz = np.less_equal(np.abs(x),1)
        y = self.mu[S]*x + self.b[S]
        y[~zz] = np.inf
        return y
    
    def Hstar_vec(self, x, S):
        zz = np.less_equal(np.abs(x),1)
        y = self.mu[S]
        y[~zz] = np.inf
        return y