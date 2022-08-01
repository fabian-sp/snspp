import numpy as np
from numba.experimental import jitclass
from numba import int64, float64, typeof

#%% squared hinge loss 

spec_log = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]),               
    ('N', int64), 
    ('m', int64[:]),
]
        
@jitclass(spec_log)
class squared_hinge_loss:
    """ 
    
    Let :math:``A \in \mathbb{R}^{N\times n}`` and :math:``b \in \{-1,1\}^{N}`` be given.
    This implements the squared hinge loss function given by
    
    .. math:
        f(x) = \frac{1}{N} \sum_{i=1}^{N}  \max(0, 1 - b_i\cdot(a_i x))^2
    
    where :math:`a_i` is the i-th row of :math:`A`.
    
    Using the row-wise multiplication :math`b \cdot A`, we have
    
    .. math:
        f_i(z) = \max(0, 1-z)^2
        
    The convex conjugate is given by
    
    .. math:
        f_i^\ast(z) = \begin{cases} z(1+z/4) \quad z \leq 0 \\ +\infty \quad text{else} \end{cases}.
    
    """
    
    def __init__(self, b):
        self.name = 'squared_hinge'
        self.convex = True
        
        self.b = b
        self.N = len(self.b)
        self.m = np.repeat(1,self.N)
        
        return
    
    def eval(self, z):
        """
        Method for evaluating :math:`f(x)`.
        The array ``x`` should be the same type as A (we use float64).
        """
        y = (np.maximum(0,1-z)**2).sum()
         
        return (1/self.N)*y

    def f(self, x, i):
        """
        Method for evaluating :math:`f_i(x)`.
        """
        return np.maximum(0,1-x)**2
    
    def g(self, x, S):
        """
        Method for evaluating :math:`f_i'(x)`.
        """
        return (x<=1)*(2*x-2)
    
    # these are actually never used (see vectorized versions below instead)
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
    ('mu', float64[:]),
    ('N', int64), 
    ('m', int64[:]),
]
        

@jitclass(spec_log)
class huber_loss:
    """ 
          
    Let :math:``A \in \mathbb{R}^{N\times n}`` and :math:``b \in \mathbb{R}^{N}`` be given.
    This implements the squared loss function given by
    
    .. math:
        f(x) = \frac{1}{N} \sum_{i=1}^{N} f_i(A_i x)
        
    with
    
    .. math:
        f_i(z) = \begin{cases}\frac{(z-b_i)^2}{2\mu} \quad |z-b_i| \leq \mu \\ |z-b_i| - \mu/2 \quad \text{else} \end{cases}.
        
    The convex conjugate is given by
    
    .. math:
        f_i^\ast(z) = \begin{cases} \frac{1}{2} \mu z^2 + b_i\cdot z \quad |z| \leq 1 \\ +\infty \quad \text{else} \end{cases}.
        
    """
    
    def __init__(self, A, b, mu):
        self.name = 'huber'
        self.convex = True
        
        self.b = b
        self.mu = mu
        self.N = len(self.b)
        self.m = np.repeat(1,self.N)
        
        return
    
    def eval(self, z):
        """
        Method for evaluating :math:`f(x)`.
        The array ``x`` should be the same type as A (we use float64).
        """
        
        ixx = np.abs(z-self.b) <= self.mu
        
        t1 = ixx * (z-self.b)**2/(2*self.mu)
        t2 = (1-ixx) * np.abs(z-self.b) - self.mu/2
        return (1/self.N) * np.sum(t1+t2)

    def f(self, x, i):
        """
        Method for evaluating :math:`f_i(x)`.
        """
        if np.abs(x-self.b[i]) <= self.mu[i]:
            y = (x-self.b[i])**2/(2*self.mu[i])
        else:
            y = np.abs(x-self.b[i]) - self.mu[i]/2
        return y
    
    def g(self, x, S):
        """
        Method for evaluating :math:`f_i'(x)`.
        """
        ixx = np.abs(x-self.b[S]) <= self.mu[S]
        return (1-ixx)*np.sign(x-self.b[S])  + ixx*((x-self.b[S])/self.mu[S])
    
    # these are actually never used (see vectorized versions below instead)
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
    
    
#%% Pseudo-Huber loss 

spec_log = [
    ('name', typeof('abc')),
    ('convex', typeof(True)),
    ('b', float64[:]), 
    ('mu', float64[:]),
    ('N', int64), 
    ('m', int64[:]),
]
        

@jitclass(spec_log)
class pseudohuber_loss:
    """ 
          
    Let :math:``A \in \mathbb{R}^{N\times n}`` and :math:``b \in \mathbb{R}^{N}`` be given.
    This implements the squared loss function given by
    
    .. math:
        f(x) = \frac{1}{N} \sum_{i=1}^{N} f_i(A_i x)
        
    with
    
    .. math:
        f_i(z) = \sqrt{\mu^2+(z-b_i)^2} - \mu.
        
    The convex conjugate is given by
    
    .. math:
        f_i^\ast(z) = \begin{cases} b_i z+\frac{\mu z^2-\mu}{\sqrt{1-z^2}} \quad |z| < 1 \\ +\infty \quad \text{else} \end{cases}.
        
    """
    
    def __init__(self, A, b, mu):
        self.name = 'pseudohuber'
        self.convex = True
        
        self.b = b
        self.mu = mu
        self.N = len(self.b)
        self.m = np.repeat(1,self.N)
        
        return
    
    def eval(self, z):
        """
        Method for evaluating :math:`f(x)`.
        The array ``x`` should be the same type as A (we use float64).
        """
        y = np.sqrt(self.mu**2 + (z-self.b)**2) - self.mu
        return (1/self.N) * y.sum()

    def f(self, x, i):
        """
        Method for evaluating :math:`f_i(x)`.
        """
        y = np.sqrt(self.mu[i]**2 + (x-self.b[i])**2) - self.mu[i]
        return y
    
    def g(self, x, S):
        """
        Method for evaluating :math:`f_i'(x)`.
        """
        y = (x-self.b[S])/np.sqrt(self.mu[S]**2 + (x-self.b[S])**2)
        return y
        
    def fstar_vec(self, x, S):
        zz = np.less(np.abs(x),1)
        y = self.b[S] * x + (self.mu[S]*x**2 - self.mu[S])/np.sqrt(1-x**2) + self.mu[S]
        y[~zz] = np.inf
        return y
    def gstar_vec(self, x, S):
        zz = np.less(np.abs(x),1)
        y = self.b[S] + self.mu[S]*x/np.sqrt(1-x**2)
        y[~zz] = np.inf
        return y
    def Hstar_vec(self, x, S):
        zz = np.less(np.abs(x),1)
        y = self.mu[S]/(1-x**2)**(1.5)
        y[~zz] = np.inf
        return y