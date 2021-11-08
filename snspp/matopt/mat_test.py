#%%
from nuclear import NuclearNorm
from utils import multiple_matdot, matdot
from mat_loss import mat_lsq
from mat_spp import solve_subproblem

p = 20
q = 30
N = 100

r = 5

A = np.zeros((p,q,N))
b = np.zeros(N)

for i in np.arange(N):
    A[:,:,i] = np.random.randn(p,q)
    
    
X = np.random.randn(p,q)
    
phi = NuclearNorm(0.1)

Y = phi.prox(X, 0.1)
Y = phi.jacobian_prox(X, np.zeros_like(X), 0.1)

f = mat_lsq(A, b)


xi = np.ones(N)*1000
S = np.arange(50, dtype = int)
reduce_variance = False
alpha = 0.1

def get_default_newton_params():
    
    params = {'tau': .9, 'eta' : 1e-5, 'rho': .5, 'mu': .4, 'eps': 1e-3, \
              'cg_max_iter': 12, 'max_iter': 20}
    
    return params

newton_params = get_default_newton_params()


new_X, xi, info = solve_subproblem(f, phi, X, xi, alpha, A, S, newton_params = newton_params, reduce_variance = False, xi_tilde = None, full_g = None, verbose = True)
