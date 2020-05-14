import numpy as np


def create_lsq_oracle(b):
    """ 
    f is the squared loss function (1/N) * ||Ax-b||**2
    each f_i is of the form x --> |x-b_i|**2
    _star denotes the convex conjugate
    n is sample size
    """
    n = len(b)
    
    f = lambda x, i: (x - b[i])**2
    g = lambda x, i: 2 * (x - b[i])
    
    fstar = lambda x, i: .25 * np.linalg.norm(x)**2 + b[i] * x
    gstar = lambda x, i: .5 * x + b[i]
    
    Hstar = lambda x, i: .5 
    
 
    def lsq_oracle(S):
        """

        Parameters
        ----------
        S : np.array
            sample of {1,...,n} for which we calculate the gradients/hessians

        Returns
        -------
        gstar_S : dict
            for each i in S, this contains the mapping nabla f^\star_i(xi) at key i 
        Hstar_S : dict
            for each i in S, this contains the mapping \partial(nabla f^\star_i(xi)) at key i 

        """
        assert np.all(np.isin(S, np.arange(n)))
        
        gstar_S = dict()
        Hstar_S = dict()
        
        for i in S:
            
            gstar_S[i] = lambda x: gstar(x,i)
            Hstar_S[i] = lambda x: Hstar(x,i)
        
        return gstar_S, Hstar_S
    
        
    return lsq_oracle

#%%

def prox_1norm(v, l): 
    return np.sign(v) * np.maximum(abs(v) - l, 0)


def jacobian_1norm(v, l):  
    d = (abs(v) > l).astype(int)
    return np.diag(d)

def create_1norm_func(lambda1):
    
    assert lambda1 > 0 
    
    prox_phi = lambda v, alpha: prox_1norm(v, alpha* lambda1)
    jacobian_prox_phi = lambda v, alpha: jacobian_1norm(v, alpha*lambda1)
    
    return prox_phi, jacobian_prox_phi
    
    

#%%
n=1000
p=100

b = np.random.randn(n)

lsq_oracle = create_lsq_oracle(b)

S = np.array([1,2,3])

res = lsq_oracle(S)
