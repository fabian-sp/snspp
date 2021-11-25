import numpy as np
import matplotlib.pyplot as plt
from snspp.matopt.nuclear import NuclearNorm
from snspp.matopt.mat_loss import mat_lsq
from snspp.matopt.mat_spp import stochastic_prox_point, solve_subproblem
from snspp.matopt.utils import compute_full_xi

p = 5
q = 8
N = 10

r = 5

A = np.zeros((p,q,N))
b = np.random.randn(N)

for i in np.arange(N):
    A[:,:,i] = np.random.randn(p,q)
    
    
X = np.random.randn(p,q)
    
phi = NuclearNorm(1.)

Y = phi.prox(X, 0.1)
Y = phi.jacobian_prox(X, np.zeros_like(X), 0.1)

f = mat_lsq(A, b)


params = {'alpha': 1., 'batch_size': f.N, 'reduce_variance': True, 'max_iter' : 10}
X0 = np.zeros((p,q))
X0 = np.random.randn(p,q)


f.eval(X0)
xi = compute_full_xi(f, X0)


X, info = stochastic_prox_point(f, phi, X0, xi = None, tol = 1e-4, params = params, verbose = True, measure = True)

plt.plot(info['objective'])

#%%
xi = np.ones(N)*1000
S = np.arange(50, dtype = int)
reduce_variance = False
alpha = 0.1

new_X, xi, info = solve_subproblem(f, phi, X, xi, alpha, A, S, newton_params = newton_params, reduce_variance = False, xi_tilde = None, full_g = None, verbose = True)


#%%
from snspp.matopt.mat_spp import calc_AUA, calc_AUA2

subA = A[:,:,np.arange(50)]
Z = np.random.randn(p,q)
alpha = 0.1

%timeit calc_AUA(phi, Z, alpha, subA)
%timeit calc_AUA2(phi, Z, alpha, subA)
