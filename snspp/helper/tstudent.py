"""
@author: Fabian Schaipp
"""
import numpy as np
from numba.experimental import jitclass
from numba import njit
from numba import int64, float32, float64, typeof
from numba.typed import List

def lipschitz_bound(f, x, b = 15):
    """
    estimate Lipschitz constant of nabla f_S by sampling at point x
    """
    L=np.zeros(100)
    for i in range(100):
        S = np.random.randint(low = 0, high = f.N, size = b)
        b_S = f.b[S]
        z_S = f.A[S,:] @ x
        h_S = f._h(z_S, b_S) - f.gamma
        
        L[i] = 1/len(S) * np.sum( abs(h_S) * (np.apply_along_axis(np.linalg.norm, axis = 1, arr = f.A[S,:]))**2)
        
    return L

################################################################
# MAIN
################################################################

spec_tstud = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('A', float64[:,:]),
    ('v', float64),
    ('N', int64), 
    ('m', int64[:]),
    ('eps', float64),
    ('gamma', float64),
    ('z', float64[:]),
]

    
@jitclass(spec_tstud)
class tstudent_loss:
    """ 
    f is the t-student loss function i.e. 1/N sum_i log(1+((a_i @ x)-b_i)^2/v)
    transform input such that each A_i (notation of paper) is a_i*b_i 
    
    """
    
    def __init__(self, A, b, v):
        self.name = 'tstudent'
        self.convex = False
        
        self.b = b
        self.A = np.ascontiguousarray(A)
        self.v = v
        self.N = len(self.b)
        self.m = np.repeat(1,self.N)
        
        # epsilon for regularization
        self.eps = 1e-3
        self.gamma = 1/(4*self.v) + self.eps
        
        # helper array to save yet computed results from self._zstar --> do not recompute in Hstar
        self.z = np.zeros(self.N)
        
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
        res = 2*a/(self.v+a**2)
        return res
       
    def weak_conv(self, S):
        return self.gamma * np.ones(len(S))
    
    # computes solution to cubic polynomial
    def _zstar0(self, x, b):
        
        c3 = -self.gamma
        c2 = x + 2*self.gamma*b
        c1 = -2*b*x - 2 - self.gamma*self.v - self.gamma*(b**2)
        c0 = x*self.v + x*b**2 + 2*b
        
        z = np.roots(np.array([c3,c2,c1,c0], dtype=np.complex64))  
        # get root with minimal imaginary part
        ixx = np.abs(np.imag(z)).argmin()
        return np.real(z[ixx])
    
    def _zstar(self, x, b, tol=1e-12, max_iter=10):

        a2 = -(x + 2*self.gamma*b)/self.gamma
        a1 = -(-2*b*x - 2 - self.gamma*self.v - self.gamma*(b**2))/self.gamma
        a0 = -(x*self.v + x*b**2 + 2*b)/self.gamma
        
        xinfl = -a2/3
        yinfl = xinfl**3+ a2*xinfl**2 + a1*xinfl +a0
        
        d = a2**2 - 3*a1
        if d >= 0:
            if yinfl < 0 :
                z = xinfl + (2/3)*np.sqrt(d)
            else:
                z = xinfl - (2/3)*np.sqrt(d)
        else:
            z = xinfl
        
        for k in np.arange(max_iter):
            fun =     z**3 +   a2*z**2 + a1*z + a0
            deriv = 3*z**2 + 2*a2*z    + a1
            
            if np.abs(fun) <= tol:
                break   
            z = z - fun/deriv       
        return z
    
    def _fstar(self, x ,i):
        z = self._zstar(x, self.b[i])
        if np.isnan(z):
            res = np.inf
        else:
            res = x*z - self.f(z,i) - (self.gamma/2)*z**2
        return res
    
    # loop versions
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
            self.z[i] = self._zstar(x, self.b[i])      
            Y[j] = self.z[i]
        return Y
    
    def _h(self, x, b):
        """
        second derivative of f+gamma/2 \|.\|^2
        """
        nom = 2*(self.v-(b-x)**2)
        denom = (self.v+(b-x)**2)**2
        
        res = nom/denom + self.gamma
        return res
    
    def Hstar(self, X, i):
        
        Y = np.zeros_like(X)
        for j in range(len(X)):
            x = X[j]
            g_i = self._zstar(x, self.b[i])
            Y[j] = 1/(self._h(g_i, self.b[i]))
            
        return Y
    
    # vectorized versions
    def fstar_vec(self, x, S):
        """
        x = xi[S]
        """
        Y = np.zeros_like(x)        
        for j in range(len(x)):
            Y[j] = self._fstar(x[j],S[j])
            
        return Y   
    
    def gstar_vec(self, x, S):
        Y = np.zeros_like(x)
        for j in range(len(x)):
            z_j =self._zstar(x[j], self.b[S[j]])      
            self.z[S[j]] = z_j
            Y[j] = z_j
        return Y
    
    def Hstar_vec(self, x, S):
        b_S = self.b[S]
        g_S = self.z[S]
        # g_S = np.zeros_like(x)
        # for j in range(len(x)):
        #     z_j =self._zstar(x[j], self.b[S[j]])      
        #     #self.z[S[j]] = z_j
        #     g_S[j] = z_j
        return 1/(self._h(g_S, b_S))

#%%    
# A= np.random.randn(50,100)
# b = np.random.randn(50)
# b[0] = 0.
# x = np.random.randn(100)

# f = tstudent_loss(A, b, v=1.)

# z = np.array([1.], dtype = np.float64)

# f._zstar(1,1)

# f.fstar(z,1)
# f.gstar(z,1)
# f.Hstar(z,1)


# f.g(np.array([4]),4)

# for i in range(1000):
    
#     x = np.random.randn(1)
#     y = np.random.randn(1)
    
#     print(f.f(x,3) + f.gamma/2 * x**2 + f.fstar(y,3) - x*y)
#     assert (f.f(x,3) + f.gamma/2*x**2 + f.fstar(y,3) - x*y) >= 0




# all_x = np.linspace(-100,100, 2000)
# all_f = np.zeros_like(all_x)
# all_g = np.zeros_like(all_x)
# all_h = np.zeros_like(all_x)

# for j in range(len(all_x)):
    
#     xx = all_x[j]
#     all_f[j] = f.fstar(np.array([xx]), 166)
#     all_g[j] = f.gstar(np.array([xx]), 166)
#     all_h[j] = f.Hstar(np.array([xx]), 166)

    
# import matplotlib.pyplot as plt
# plt.plot(all_x, all_f)
# plt.plot(all_x, all_g)
# plt.plot(all_x, all_h)


