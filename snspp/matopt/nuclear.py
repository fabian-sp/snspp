"""
author: Fabian Schaipp
"""


import numpy as np

def huber(t, eps):  
    y = (t>=eps/2)*t + (np.abs(t)<=eps/2) * 1/(2*eps)*(t+eps/2)**2        
    return y

def softt(v, rho):
    """ soft threshholding operator"""
    # equivalent:
    #np.maximum(0, v-rho) - np.maximum(0, -v-rho)
    return np.sign(v) * np.maximum(np.abs(v) - rho, 0.)    

def smooth_softt(v, rho, eps):
    """ smoothed soft threshholding operator"""
    return huber(v-rho, eps) - huber(-v-rho, eps)

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
    
def smooth_prox(Y, rho, eps = 1e-3):
    """smoothed proximal operator of the nuclear norm"""
    
    (p,q) = Y.shape
    U,S,Vt = np.linalg.svd(Y, full_matrices=False)
    S_bar = smooth_softt(S, rho, eps) 
    
    return (U*S_bar)@Vt

def construct_gamma(v1, rho, eps, v2 = None):
    """
    v1, v2 vector of sg. values
    """
    p1 = len(v1)
    if v2 is None:
        v2 = v1.copy()
        p2 = p1
    else:
        p2 = len(v2)
    
    #Gam2 = np.zeros((p,p))
    s_v1 = smooth_softt(v1, rho, eps)
    ds_v1 = deriv_smooth_softt(v1, rho, eps)
    
    if v2 is None:
        s_v2 = s_v1.copy()
    else:
        s_v2 = smooth_softt(v2, rho, eps)
        
    # for i in np.arange(p):
    #     for j in np.arange(start=i+1,stop=p):
    #         Gam2[i,j] = (s_v[i]-s_v[j])/(v[i]-v[j])
    
    # np.tile not supported by numba    
    # h1 = np.tile(s_v1, (p2,1)).T - np.tile(s_v2, (p1,1))
    # h2 = np.tile(v1, (p2,1)).T - np.tile(v2, (p1,1))
    # h3 = np.tile(ds_v1, (p2,1)).T
    
    h1 = tile(s_v1, p2).T - tile(s_v2, p1)
    h2 = tile(v1, p2).T - tile(v2, p1)
    h3 = tile(ds_v1, p2).T
    # ixx are indices where v_i == v_j --> use derivative at these indices
    ixx = (h2 == 0)
    # avoid nans from dividing by zero
    Gam = h3*ixx + (1-ixx)*np.nan_to_num(h1/h2)
        
    if v2 is None:
        assert np.all(Gam == Gam.T)
    
    return Gam

def tile(v, q):
    p=len(v)
    return np.repeat(v,q).reshape(p,q).T
    
def smooth_prox_jacobian(Y, rho, eps, tau, H):
    (p,q) = Y.shape
    assert H.shape == (p,q)
    
    U,S,Vt = np.linalg.svd(Y)
    
    # how to reconstruct from full svd
    #tmp = U @ np.hstack((np.diag(S),np.zeros((p,q-p)))) @ Vt
    #assert np.allclose(Y,tmp)

    V1 = Vt.T[:,:p]
    V2 = Vt.T[:,p:]
    
    fullH = U.T@H@Vt.T
    H1 = fullH[:,:p]
    H2 = fullH[:,p:]
    
    Hs = (H1+H1.T)/2
    Ha = (H1-H1.T)/2
    
    D = np.diag(deriv_eps_smooth_softt(S, rho, eps))
    Gam_aa = construct_gamma(v1 = S, v2 = None, rho = rho, eps = eps)
    Gam_ay = construct_gamma(v1 = S, v2 = -S, rho = rho, eps = eps)
    Gam_ab = construct_gamma(v1 = S, v2 = np.zeros(q-p), rho = rho, eps = eps)
    
    term1 = U@(Gam_aa*Hs + Gam_ay*Ha + tau*D) @ V1.T
    term2 = U @ (Gam_ab*H2) @ V2.T
    
    return term1 + term2

#%%

class NuclearNorm:
    """
    class for the regularizer x --> lambda1 ||X||_\star
    """
    def __init__(self, lambda1):
        assert lambda1 > 0 
        self.name = 'nuclear_norm'
        self.lambda1 = lambda1
        
    def eval(self, X):
        #_,S,_ = np.linalg.svd(X, full_matrices = False)
        #S.sum()
        return self.lambda1 * np.linalg.norm(X, 'nuc')
    
    def prox(self, X, alpha):
        """
        calculates prox_{alpha*phi}(x)
        """
        assert alpha > 0
        l = alpha * self.lambda1
        return smooth_prox(X, l, eps = 1e-3)
    

    def jacobian_prox(self, X, H, alpha):
        assert alpha > 0
        l = alpha * self.lambda1
        
        return smooth_prox_jacobian(X, l, eps = 1e-3, tau = 1e-3, H = H)
    
    def moreau(self, x, alpha):
        assert alpha > 0
        z = self.prox(x, alpha)
        return alpha*self.eval(z) + .5 * np.linalg.norm(z-x)**2
    
    
#%% tests

# p = 100
# q = 200

# Y = np.random.randn(p,q)
# v = np.random.randn(p)
# eps = 1e-5
# rho = 0.5

# v1 = np.random.randn(p)
# v2 = np.random.randn(p+2)

# H = np.random.randn(p,q)
# tau = 0.01



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



    