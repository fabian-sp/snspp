"""
@author: Fabian Schaipp
"""

import numpy as np
from numba import njit

from numba.experimental import jitclass
from numba import float64, typeof

@njit()
def huber(t, eps):  
    y = (t>=eps/2)*t + (np.abs(t)<=eps/2) * 1/(2*eps)*(t+eps/2)**2        
    return y

@njit()
def softt(v, rho):
    """ soft threshholding operator"""
    # equivalent:
    #np.maximum(0, v-rho) - np.maximum(0, -v-rho)
    return np.sign(v) * np.maximum(np.abs(v) - rho, 0.)    

@njit()
def deriv_softt(v, rho):
    """ returns element of subdifferential of the soft threshholding operator"""  
    u = 1.*(np.abs(v) > rho) # 1. makes bool to float
    return u 

@njit()
def smooth_softt(v, rho, eps):
    """ smoothed soft threshholding operator"""
    return huber(v-rho, eps) - huber(-v-rho, eps)

@njit()
def deriv_smooth_softt(v, rho, eps):
    """ derivative of the smoothed soft threshholding operator"""
    
    ix1 = np.logical_and(v>=rho-eps/2, v<=rho+eps/2)
    ix2 = np.logical_and(v>=-rho-eps/2, v<=-rho+eps/2)
    ix3 = (v>=rho+eps/2)
    ix4 = (v<=-rho-eps/2)
    
    val1 = 1/eps*(v-rho+eps/2)
    val2 = -1/eps*(v+rho-eps/2)
    val3 = 1
    val4 = 1
    return val1*ix1 + val2*ix2 + val3*ix3 + val4*ix4 

@njit()
def deriv_eps_smooth_softt(v, rho, eps):
    """ derivative of the smoothed soft threshholding operator (after eps variable)"""
    
    ix1 = np.logical_and(v>=rho-eps/2, v<=rho+eps/2)
    ix2 = np.logical_and(v>=-rho-eps/2, v<=-rho+eps/2)
    ix3 = (v>=rho+eps/2)
    ix4 = (v<=-rho-eps/2)
    
    val1 = 1/(2*eps)*(v-rho+eps/2)*(1-(1/eps)*(v-rho+eps/2))
    val2 = 1/(2*eps)*(v+rho-eps/2)*(1+(1/eps)*(v+rho-eps/2))
    val3 = 0
    val4 = 0
    return val1*ix1 + val2*ix2 + val3*ix3 + val4*ix4            
  
@njit()  
def prox_nuclear(Y, rho, eps = 1e-3):
    """
    (smoothed) proximal operator of the nuclear norm
    uses the smoothing oft soft-thresholding with Huber if eps>0
    """
    
    (p,q) = Y.shape
    U,S,Vt = np.linalg.svd(Y, full_matrices=False)
    
    if eps > 0:
        S_bar = smooth_softt(S, rho, eps) 
    else:
        S_bar = softt(S, rho) 
        
    return (U*S_bar)@Vt

@njit()
def construct_gamma(v1, v2, rho, eps):
    """
    v1, v2 vector of sg. values
    if v2 is copy of v1, we can save computations. In this case, specify v2 as vector of nan
    """
    p1 = len(v1)
    if eps > 0:
        s_v1 = smooth_softt(v1, rho, eps)
        ds_v1 = deriv_smooth_softt(v1, rho, eps)
    else:
        s_v1 = softt(v1, rho)
        ds_v1 = deriv_softt(v1, rho)
    
    if np.all(np.isnan(v2)):
        v2 = v1.copy()
        s_v2 = s_v1.copy()
    else:
        if eps > 0:
            s_v2 = smooth_softt(v2, rho, eps)
        else:
            s_v2 = softt(v2, rho)
            
    p2 = len(v2)
        
    # np.tile not supported by numba    
    h1 = tile(s_v1, p2).T - tile(s_v2, p1)
    h2 = tile(v1, p2).T   - tile(v2, p1)
    h3 = tile(ds_v1, p2).T
    
    # ixx are indices where v_i == v_j --> use derivative at these indices
    ixx = (h2 == 0)
    # avoid nans from dividing by zero
    Gam = h3*ixx + (1-ixx)*(h1/(1e-12+h2))
        
    #assert np.all(Gam == Gam.T)
    
    return Gam

@njit()
def tile(v, q):
    p=len(v)
    return np.repeat(v,q).reshape(p,q).T

@njit()    
def prox_nuclear_jacobian(Y, rho, eps, tau, H):
    """
    Jacobian of the (smoothed) proximal operator of the nuclear norm
    uses the smoothing oft soft-thresholding with Huber if eps>0
    """
    (p,q) = Y.shape
    assert H.shape == (p,q)
    assert q>=p, "only possible for q>=p"
    
    U,S,Vt = np.linalg.svd(Y)
    
    # how to reconstruct from full svd
    #tmp = U @ np.hstack((np.diag(S),np.zeros((p,q-p)))) @ Vt
    #assert np.allclose(Y,tmp)

    V1T = Vt[:p,:]
    V2T = Vt[p:,:]
    
    fullH = U.T@H@Vt.T
    H1 = fullH[:,:p]
    H2 = fullH[:,p:]
    
    Hs = (H1+H1.T)/2
    Ha = (H1-H1.T)/2
    
    if eps > 0:
        D = np.diag(deriv_eps_smooth_softt(S, rho, eps))

    Gam_aa = construct_gamma(v1 = S, v2 = np.ones_like(S)*np.nan, rho = rho, eps = eps)
    Gam_ay = construct_gamma(v1 = S, v2 = -S,                     rho = rho, eps = eps)
    Gam_ab = construct_gamma(v1 = S, v2 = np.zeros(q-p),          rho = rho, eps = eps)
    
    if eps > 0:
        term1 = (Gam_aa*Hs + Gam_ay*Ha + tau*D) @ V1T
    else:
        term1 = (Gam_aa*Hs + Gam_ay*Ha) @ V1T
    
    term2 = (Gam_ab*H2) @ V2T
    
    return U @ (term1 + term2)

#%%

spec_l1 = [
    ('name', typeof('abc')),
    ('lambda1', float64)
]

@jitclass(spec_l1)
class NuclearNorm:
    """
    class for the regularizer X --> lambda1 ||X||_\star
    """
    def __init__(self, lambda1):
        assert lambda1 > 0 
        self.name = 'nuclear_norm'
        self.lambda1 = lambda1
        self.eps = 1e-5
        
    def eval(self, X):
        # for numba, some options of svd are not available
        _,S,_ = np.linalg.svd(X, full_matrices = False)       
        return self.lambda1 * S.sum()  #self.lambda1 * np.linalg.norm(X, 'nuc')
    
    def prox(self, X, alpha):
        """
        calculates prox_{alpha*phi}(X)
        """
        assert alpha > 0
        l = alpha * self.lambda1
        return prox_nuclear(X, l, eps = self.eps)
    

    def jacobian_prox(self, X, H, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        
        return prox_nuclear_jacobian(X, l, eps = self.eps, tau = 1e-5, H = H)
    
    def moreau(self, X, alpha):
        assert alpha > 0
        Z = self.prox(X, alpha)
        return alpha*self.eval(Z) + .5 * np.linalg.norm(Z-X)**2
    
    
#%% tests

# p = 100
# q = 200

# Y = np.random.randn(p,q)
# U,S,Vt = np.linalg.svd(Y)
# v = np.random.randn(p)
# eps = 1e-5
# rho = S[-5]

# v1 = np.random.randn(p)
# v2 = np.random.randn(p+2)

# H = np.random.randn(p,q)
# tau = 0.01


# Z1 = prox_nuclear_jacobian(Y, rho, 1e-5, 1e-5, H)

# Z2 = prox_nuclear_jacobian(Y, rho, 0., 0., H)


# x = np.linspace(-1,1,1000)
# y = smooth_softt(x, 0.5, 0.2)
# y2 = deriv_smooth_softt(x, 0.5, 0.2)
# import matplotlib.pyplot as plt
# plt.plot(x,y)
# plt.plot(x,y2)


# v = 0.1
# eps = np.linspace(0.01,3,100)
# f = list()
# g = list()
# for e in eps:
#     f.append(smooth_softt(v, rho, e))
#     g.append(deriv_eps_smooth_softt(v, rho, e))

# plt.plot(eps,f)
# plt.plot(eps,g)



    