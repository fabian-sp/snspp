"""
@author: Fabian Schaipp
"""
import numpy as np
from numba.experimental import jitclass
from numba import int64, float32, float64, typeof
from numba.typed import List


spec_tstud = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('A', float64[:,:]),
    ('v', float64),
    ('N', int64), 
    ('m', int64[:]),
    ('tol', float64),
]

#@jitclass(spec_tstud)
class tstudent_loss:
    """ 
    f is the t-student loss function i.e. 1/N sum_i log(1+((a_i @ x)-b_i)^2/v)
    transform input such that each A_i (notation of paper) is a_i*b_i 
    
    """
    
    def __init__(self, A, b, v):
        self.name = 'tstudent'
        self.convex = False
        
        self.b = b
        self.A = A
        self.v = v
        self.N = len(self.b)
        self.m = np.repeat(1,self.N)
        
        # Hstar is numerically instable close to 0, hence evaluate within tol-ball of 0
        self.tol = 1e-2
        
    def eval(self, x):
        """
        method for evaluating f(x)
        """
        z = self.A@x - self.b
        y = np.log(1+ z**2/self.v).sum()
         
        return (1/self.N)*y

    def f(self, x, i):
        """
        evaluate f_i(x)
        """
        return np.log(1+(x-self.b[i])**2/self.v)
    
    def g(self, x, i):
        
        a = x-self.b[i]
        return 2*a/(self.v+a**2)
    
    def weak_conv(self, i):
        return 1/(4*self.v)
    
    @staticmethod
    def _zstar(x,v,b):
    
        c3 = 1
        c2 = -4*v*x-2*b
        c1 = 8*x*v*b +9*v+b**2
        c0 = -4*v* (x*v+x*b**2+2*b)
        
        z = np.roots(np.array([c3,c2,c1,c0]))  
        z = z[np.isreal(z)]
        
        if len(z) == 0 or len(z) > 1:
            res = np.nan
        else:
            res = np.real(z)[0]
        return res
    
    def _fstar(self, x ,i):
        obj = lambda z: x*z - self.f(z,i) - 1/(8*self.v)*z**2
        z = self._zstar(x, self.v, self.b[i])
        if np.isnan(z):
            res = np.inf
        else:
            res = obj(z)
        return res

    def fstar(self, X, i):
        Y = np.zeros_like(X)
        
        for j in range(len(X)):
            x = X[j]
            Y[j] = self._fstar(x,i)
            
        return Y
    
    def gstar(self, X, i):
        Y = np.zeros_like(X)
        for j in range(len(X)):
            x = X[j]            
            h = 1e-2
            Y[j] = (self._fstar(x+h, i) - self._fstar(x, i) ) / h
            
        return Y
    
    
    def Hstar(self, X, i):
        Y = np.zeros_like(X)
        for j in range(len(X)):
            x = X[j]
            h = 1e-2*x
            Y[j] = (self._fstar(x+h, i) - 2*self._fstar(x, i) + self._fstar(x-h, i) ) / h**2
            
        return Y

#%%    
A= np.random.randn(50,100)
b = np.random.randn(50)
x = np.random.randn(100)

f = tstudent_loss(A, b, v=.25)

# for i in range(1000):
    
#     x = np.random.randn(1)
#     y = np.random.randn(1)
    
#     print(f.f(x,3) + 1/(8*f.v)*x**2 + f.fstar(y,3) - x*y)
#     assert (f.f(x,3) + + 1/(8*f.v)*x**2 + f.fstar(y,3) - x*y) >= 0




# all_x = np.linspace(-5, 5, 100)
# all_f = np.zeros_like(all_x)
# all_g = np.zeros_like(all_x)
# all_h = np.zeros_like(all_x)

# for j in range(len(all_x)):
    
#     xx = all_x[j]
#     print(xx)
#     all_f[j] = f.fstar(np.array([xx]), 1)
#     all_g[j] = f.gstar(np.array([xx]), 1)
#     all_h[j] = f.Hstar(np.array([xx]), 1)
    
# plt.plot(all_x, all_f)
# plt.plot(all_x, all_g)
# plt.plot(all_x, all_h)



