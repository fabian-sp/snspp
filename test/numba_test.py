import numpy as np
from numba import jit
from numba.typed import List
from numba.experimental import jitclass
from numba import int32, int64, float32, typeof

from ssnsp.helper.lasso import lsq

N = 10000
n = 10

A = np.random.randn(N,n).astype('float32')
b = np.random.randn(N).astype('float32') 
x = np.random.rand(n).astype('float32')


#%%
spec = [
    ('name', typeof('abc')),
    ('b', float32[:]),               
    ('A', float32[:,:]),
    ('N', int64), 
    ('m', int64[:]),
]
#self.name = 'squared'
        

@jitclass(spec)
class lsq:
    """ 
    f is the squared loss function (1/N) * ||Ax-b||**2
    each f_i is of the form x --> |x-b_i|**2
    _star denotes the convex conjugate
    N is the sample size (i.e. number of summands)
    """
    
    def __init__(self, A, b):
        self.name = 'tester'
        self.b = b
        self.A = A
        self.N = len(b)
        self.m = np.repeat(1,self.N)
        
    
    def eval(self, x):
        """
        method for evaluating f(x)
        """

        return (1/self.N) * np.linalg.norm(self.A@x - self.b)**2      
        
    def f(self, x, i):
        """
        evaluate f_i(x)
        """
        return (x - self.b[i])**2
    
    def g(self, x, i):
        return 2 * (x - self.b[i])
    
#%%

test_f = lsq(A,b)

%timeit [test_f.g(A[i,:] @ x, i) for i in range(N)]

#%%

xi = dict(zip(np.arange(test_f.N), [ -.5 * np.ones(test_f.m[i]) for i in np.arange(test_f.N)]))

S = range(N)

def fast_hessian(f, xi, S):
    
    nb_xi = List()
    nb_S = List()
    for i in S:
        nb_xi.append(xi[i].copy())
        nb_S.append(i)
    
    nb_res = fast_hessian_inner(f, nb_xi, nb_S)
    return list(nb_res)
    
@jit()  
def fast_hessian_inner(f, nb_xi, nb_S):
    res = List()
    for i in nb_S:
        res.append(f.Hstar(nb_xi[i], i))
    
    return res

res1 = [test_f.Hstar(xi[i], i) for i in S]
res2 = fast_hessian(test_f, xi, S)
assert res1 == res2


%timeit fast_hessian(test_f, xi, S)
%timeit [test_f.Hstar(xi[i], i) for i in S]


#%%
from numba.experimental import jitclass
from numba import int32, int64, float32, typeof


spec = [
    ('name', typeof('abc')),
    ('lambda1', float32)
]
#self.name = 'squared'
        

@jitclass(spec)
class Norm1:
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
    
    def jacobian_prox(self, x, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        #d = np.float32((np.abs(x) > l))
        d = np.zeros_like(x)
        for j in np.arange(len(x)):
            if np.abs(x[j]) > l:
                d[j] = 1
            
        
        #d = np.int8((np.abs(x) > l))
        
        return np.diag(d)
    
    def moreau(self, x, alpha):
        assert alpha > 0
        z = self.prox(x, alpha)
        return alpha*self.eval(z) + .5 * np.linalg.norm(z-x)**2
    
#%%
import numpy as np
x = np.random.rand(10).astype('float32')

Norm1(1).prox(x,1.)


Norm1(1).moreau(x, 1.)

Norm1(1).jacobian_prox(x, .01)

