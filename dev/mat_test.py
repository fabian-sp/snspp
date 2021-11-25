import numpy as np
import matplotlib.pyplot as plt

from snspp.helper.data_generation import lowrank_test

from snspp.matopt.nuclear import NuclearNorm
from snspp.matopt.mat_loss import mat_lsq
from snspp.matopt.mat_spp import stochastic_prox_point, solve_subproblem
from snspp.matopt.utils import compute_full_xi

p = 5
q = 8
N = 10
r = 5
l1 = 0.001

Xhat, A, b, f, phi, _, _ = lowrank_test(N=N,p=p,q=q,r=r,lambda1=l1,noise=0)


params = {'alpha': 1.1, 'batch_size': f.N, 'reduce_variance': True, 'max_iter' : 2000}
X0 = np.zeros((p,q))
X0 = np.random.randn(p,q)

Y = phi.prox(Xhat, 1.)
Y = phi.jacobian_prox(Xhat, np.zeros_like(X), 1.)


f.eval(Xhat)
phi.eval(Xhat)

xi = compute_full_xi(f, Xhat)

X, info = stochastic_prox_point(f, phi, X0, xi = None, tol = 1e-4, params = params, verbose = True, measure = True)

fig = plt.subplots()
plt.plot(info['objective'])

fig,axs = plt.subplots(1,2)
axs[0].imshow(Xhat)
axs[1].imshow(X)

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
