import numpy as np
from numba.experimental import jitclass
from numba import int64, float32, float64, typeof
from numba.typed import List


#%% squared loss

spec = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('A', float64[:,:]),
    ('N', int64), 
    ('m', int64[:]),
]
        

@jitclass(spec)
class lsq:
    """ 
    f is the squared loss function (1/N) * ||Ax-b||**2
    each f_i is of the form x --> |x-b_i|**2
    _star denotes the convex conjugate
    N is the sample size (i.e. number of summands)
    """
    
    def __init__(self, A, b):
        self.name = 'squared'
        self.convex = True
        
        self.b = b
        self.A = A
        self.N = len(b)
        self.m = np.repeat(1,self.N)
        
        return
    
    def eval(self, x):
        """
        method for evaluating f(x)
        x has to be the same type as A if numba is used (typicall use float64)
        """
        return (1/self.N) * np.linalg.norm(self.A@x - self.b)**2      
        
    def f(self, x, i):
        """
        evaluate f_i(x)
        """
        return (x - self.b[i])**2
    
    def g(self, x, i):
        return 2 * (x - self.b[i])
    
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

spec_log = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('A', float64[:,:]),
    ('N', int64), 
    ('m', int64[:]),
]
        

@jitclass(spec_log)
class logistic_loss:
    """ 
    f is the logistic loss function i.e. 1/N sum_i log(1+exp(b_i*(a_i @ x)))
    transform input such that each A_i (notation of paper) is a_i*b_i 
    
    """
    
    def __init__(self, A, b):
        self.name = 'logistic'
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
        y = np.log(1+ np.exp(-z)).sum()
         
        return (1/self.N)*y

    def f(self, x, i):
        """
        evaluate f_i(x)
        """
        return np.log(1+np.exp(-x))
    
    def g(self, x, i):
        
        return -1/(1+np.exp(x)) 
        
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

################
# The above functions are explicitly written for numba usage
# fstar, gstar, Hstar are actually never used
################
        
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
        L is the diagonal(!) of Lambda_t (in the notation of Milzarek et al.)
        """
             
        return 1/(L+2*self.lambda1)*L*x
        
    
    def jacobian_prox(self, x, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        
        return 1/(1+2*l)
    
    def moreau(self, x, alpha):
        assert alpha > 0
        z = self.prox(x, alpha)
        return alpha*self.eval(z) + .5 * np.linalg.norm(z-x)**2
    
    
#%% only needed for testing

class block_lsq:
    """ 
    f is the squared loss function (1/N) * ||Ax-b||**2
    _star denotes the convex conjugate
    n is sample size
    """
    
    def __init__(self, A, b, m):
        self.name = 'squared'
        self.b = b
        self.A = A
        self.N = len(m)
        self.m = m
        self.ixx = np.repeat(np.arange(self.N), self.m)
        self.convex = True
        
    def eval(self, x):
        y = 0
        for i in np.arange(self.N):
            z_i = self.A[self.ixx == i, :] @ x
            y += self.f(z_i, i)
        
        return (1/self.N)*y

    def f(self, x, i):
        return np.linalg.norm(x - self.b[self.ixx == i])**2
    
    def g(self, x, i):
        return 2 * (x - self.b[self.ixx == i])
    
    def fstar(self, x, i):
        return .25 * np.linalg.norm(x)**2 + np.sum(self.b[self.ixx == i] * x)
    
    def gstar(self, x, i):
        return .5 * x + self.b[self.ixx == i]
    
    def Hstar(self, x, i):
        return .5 * np.eye(self.m[i])


