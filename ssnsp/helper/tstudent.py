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
    
    def fstar(self, X, i):
        Y = np.zeros_like(X)
        for j in range(len(X)):
            x = X[j]
            if self.v*x**2 >= 1:
                Y[j] = np.inf
            elif np.abs(x) <= 1e-4:
                Y[j] = 0
            else:
                a = np.sqrt(1-self.v*x**2)
                Y[j] = self.b[i]*x -a-np.log(2*(1-a)/(self.v*x**2)) +1
            
        return Y
    
    def gstar(self, X, i):
        Y = np.zeros_like(X)
        for j in range(len(X)):
            x = X[j]
            if self.v*x**2 >= 1:
                Y[j] = np.inf
            elif np.abs(x) <= 1e-4:
                Y[j] = self.b[i]
            else:
                a = np.sqrt(1-self.v*x**2)
                nom = self.b[i]*x*a - self.b[i]*x + self.v*x**2 + 2*a - 2
                denom = x*(a-1)
                Y[j] = nom/denom
        return Y
    
    
    def Hstar(self, X, i):
        Y = np.zeros_like(X)
        for j in range(len(X)):
            x = X[j]
            if self.v*x**2 >= 1-1e-4:
                Y[j] = np.inf
            elif np.abs(x) <= 1e-4:
                Y[j] = self.v/2
            else:
                a = np.sqrt(1-self.v*x**2)
                nom = -self.v*x**2*a + 3*self.v*x**2 + 4*a -4
                denom = x**2*(self.v*x**2*a - 2*self.v*x**2 - 2*a + 2)
                Y[j] = nom/denom
        return Y

#%%    
# A= np.random.randn(50,100)
# b = np.random.randn(50)
# x = np.random.randn(100)

# f = tstudent_loss(A, b, v=1)

# for i in range(1000):
    
#     x = np.random.randn(1)
#     y = np.random.randn(1)
    
#     print(t.f(x,3) + t.fstar(y,3) - x*y)

# t.eval(x)


# z = np.random.rand(1)
# z = np.zeros(1)

# t.fstar(z,1)
# t.gstar(z,1)
# t.Hstar(z,1)



# all_x = np.linspace(-np.sqrt(1/f.v), +np.sqrt(1/f.v), 100)
# all_y = np.zeros_like(all_x)

# for j in range(len(all_x)):
    
#     xx = all_x[j]
#     print(xx)
#     all_y[j] = f.Hstar(np.array([xx]), 1)




# f.Hstar(np.array([2*1e-6]), 1)




