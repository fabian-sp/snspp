import numpy as np


class lsq:
    """ 
    f is the squared loss function (1/N) * ||Ax-b||**2
    each f_i is of the form x --> |x-b_i|**2
    _star denotes the convex conjugate
    n is sample size
    """
    
    def __init__(self, A, b):
        self.b = b
        self.A = A
        self.N = len(b)
        self.m = np.ones(self.N, dtype = 'int')
        
    # f = lambda x, i: (x - b[i])**2
    # g = lambda x, i: 2 * (x - b[i])
    
    # fstar = lambda x, i: .25 * np.linalg.norm(x)**2 + b[i] * x
    # gstar = lambda x, i: .5 * x + b[i]
    
    # Hstar = lambda x, i: .5 
    
    def eval(self, x):
        y = 0
        z = self.A@x
        for i in np.arange(self.N):
            y += self.f(z[i], i)
        
        return (1/self.N)*y

    def f(self, x, i):
        return (x - self.b[i])**2
    
    def g(self, x, i):
        return 2 * (x - self.b[i])
    
    def fstar(self, x, i):
        return .25 * np.linalg.norm(x)**2 + self.b[i] * x
    
    def gstar(self, x, i):
        return .5 * x + self.b[i]
    
    def Hstar(self, x, i):
        return .5
    

#%%

class Norm1:
    """
    class for the regularizer x --> lambda1 \|x\|_1
    """
    def __init__(self, lambda1):
        assert lambda1 > 0 
        self.lambda1 = lambda1
        
    def eval(self, x):
        return self.lambda1 * np.linalg.norm(x, 1)
    
    def prox(self, x, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        return np.sign(x) * np.maximum(abs(x) - l, 0)
    
    def jacobian_prox(self, x, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        d = (abs(x) > l).astype(int)
        
        return np.diag(d)
    
    def moreau(self, x, alpha):
        assert alpha > 0
        z = self.prox(x, alpha)
        return self.eval(z) + .5 * np.linalg.norm(z-x)**2
    
#%%
class block_lsq:
    """ 
    f is the squared loss function (1/N) * ||Ax-b||**2
    _star denotes the convex conjugate
    n is sample size
    """
    
    def __init__(self, A, b, m):
        self.b = b
        self.A = A
        self.N = len(m)
        self.m = m
        self.ixx = np.repeat(np.arange(self.N), self.m)

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


