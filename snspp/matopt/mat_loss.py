"""
author: Fabian Schaipp
"""
import numpy as np
from numba.experimental import jitclass
from numba import int64, float32, float64, typeof
from numba.typed import List
from numba import njit

from .utils import matdot

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
    A has form (p,q,N) and A_iX = Tr(A[i,:,:].T @ X)
    
    each f_i is of the form y --> |y-b_i|**2
    
    _star denotes the convex conjugate
    """
    
    def __init__(self, A, b):
        self.name = 'mat_squared'
        self.convex = True
        
        # nuclear norm needs p <= q
        # Tr(A_i^TX) = Tr(A_iX^T)
        if A.shape[0] > A.shape[1]:
            A = A.transpose(1,0,2)
        
        assert A.shape[0] <= A.shape[1]
        
        self.p = A.shape[0]
        self.q = A.shape[1]
        
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
            Z[i] = matdot(self.A[:,:,i],X)
        
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
    
    
    


