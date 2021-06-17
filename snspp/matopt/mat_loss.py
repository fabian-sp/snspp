"""
author: Fabian Schaipp
"""
import numpy as np
from numba.experimental import jitclass
from numba import int64, float32, float64, typeof
from numba.typed import List
from numba import njit



p = 20
q = 30
N = 100


r = 5

A = np.zeros((N,p,q))
b = np.zeros(N)

for i in np.arange(N):
    A[i,:,:] = np.random.randn(p,q)
    
    
X = np.random.randn(p,q)
    
@njit()
def mat_inner(Y,X):
    """
    calculates <Y,X> = Tr(Y.T @ X)
    """
    (p,q) = X.shape
    res = 0
    for j in np.arange(q):
        res += np.dot(Y[:,j], X[:,j])
        
    return res

#%timeit mat_inner(A[0,:,:], X)
#%timeit np.trace(A[0,:,:].T@X)

#%% squared loss

spec = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('A', float64[:,:,:]),
    ('N', int64), 
    ('m', int64[:]),
]
        

@jitclass(spec)
class mat_lsq:
    """ 
    f is the matrix-valued squared loss function (1/N) * ||AX-b||**2
    where A = linear operator from R^{pxq} \to R^N.
    A has form (N,p,q) and A_iX = Tr(A[i,:,:].T @ X)
    
    each f_i is of the form y --> |y-b_i|**2
    
    _star denotes the convex conjugate
    """
    
    def __init__(self, A, b):
        self.name = 'mat_squared'
        self.convex = True
        
        self.b = b
        self.A = A
        self.N = len(b)
        self.m = np.repeat(1,self.N)
        
        return
    
    def eval(self, X):
        """
        method for evaluating f(X)
        x has to be the same type as A if numba is used (typicall use float64)
        """
        Z = np.zeros(self.N)
        
        for i in np.arange(self.N):
            Z[i] = mat_inner(self.A[i,:,:],X)
        
        return (1/self.N) * np.linalg.norm(Z - self.b)**2      
    
    
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
    
    
    
#%%

f = mat_lsq(A, b)

f.eval(X)
