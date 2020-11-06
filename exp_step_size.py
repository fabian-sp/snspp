import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import Lasso


from ssnsp.helper.data_generation import lasso_test
from ssnsp.solver.opt_problem import problem

N = 2000
n = 300
k = 20
l1 = .001

xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False, noise = 0.01)


#%% solve with scikt to get true solution
sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-8, selection = 'cyclic', max_iter = 1e6)

start = time.time()
sk.fit(f.A,b)
end = time.time()

print(end-start)

xsol = sk.coef_.copy()

psi_star = f.eval(xsol) + phi.eval(xsol)
print("Optimal value: ", psi_star)

#%%

alpha_0 = np.logspace(-1,2,50)

batch_size = np.array([0.01, 0.05, 0.1])

A, B = np.meshgrid(alpha_0, batch_size)

# K ~ len(batch_size), L ~ len(alpha_0)
K,L = A.shape


psi_tol = 1e-3*psi_star 
EPOCHS = 20

TIME = np.zeros_like(A)
OBJ = np.zeros_like(A)
CONVERGED = np.zeros_like(A)
COST = np.zeros_like(A)

for l in np.arange(L):
    for k in np.arange(K):
        
        print("######################################")
        
        params = {'sample_style': 'constant', 'reduce_variance': True}
        
        # target M epochs 
        params["max_iter"] = int(EPOCHS *  1/B[k,l])
        params['sample_size'] = max(1, int(B[k,l] * f.N))
        params['alpha_C'] = A[k,l]
        
        print(f"ALPHA = {params['alpha_C']}")
        print(f"BATCH = {params['sample_size']}")
        
        P = problem(f, phi, tol = 1e-6, params = params, verbose = False, measure = True)
        P.solve(solver = 'ssnsp')
        
        
        xhist = P.info['iterates'].copy()
        obj = P.info['objective'].copy()
        
        print(f"OBJECTIVE = {obj[-1]}")
        
        if np.any(obj <= psi_star + psi_tol):
            stop = np.where(obj <= psi_star + psi_tol)[0][0]
            this_time = P.info['runtime'].cumsum()[stop]
            
            CONVERGED[k,l] = 1
        else:
            this_time = P.info['runtime'].sum()
            print("NO CONVERGENCE!")
            
        OBJ[k,l] = obj[-1]
        TIME[k,l] = this_time
        COST[k,l] = P.info['runtime'].sum()/ len(P.info['runtime'])

OBJ_ERR = (OBJ - psi_star)/psi_star
OBJ_ERR[OBJ_ERR >= 10] = np.nan

CONVERGED = CONVERGED.astype(bool)       

#%% 
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

colors = sns.color_palette("GnBu_d", K)
colors = sns.color_palette("viridis", K)

#%% plot objective of last iterate vs step size

fig, ax = plt.subplots()

for k in np.arange(K):
    c_arr = np.array(colors[k]).reshape(1,-1)
    #ax.scatter(A[k,:], OBJ_ERR[k,:], c = c_arr, edgecolors = 'k', label = rf"$b =  N \cdot$ {batch_size[k]} ")
    ax.plot(A[k,:], OBJ_ERR[k,:], c = colors[k], linestyle = '--', marker = 'o', label = rf"$b =  N \cdot$ {batch_size[k]} ")

ax.set_xlabel(r"Step size $\alpha$")    
ax.set_ylabel(r"$(\psi(x^k)-\psi^\star)/\psi^\star$")   

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(1e-12,1)
ax.legend()



#%% plot runtime (until convergence) vs step size

fig, ax = plt.subplots()

for k in np.arange(K):
    
    RT = TIME[k,:]
    RT[~CONVERGED[k,:]] = 10
    
    ax.plot(alpha_0, TIME[k,:], c = colors[k], linestyle = '--', marker = 'o', label = rf"$b =  N \cdot$ {batch_size[k]} ")
    
    #nc = ~CONVERGED[k,:]
    
    #c_arr = np.array(colors[k]).reshape(1,-1)
    #ax.scatter(alpha_0[nc], TIME[k,:][nc], marker = 'x', c = c_arr)
    #ax.scatter(alpha_0[~nc], TIME[k,:][~nc], marker = 'o', c = c_arr, s = 5)


ax.set_xlabel(r"Initial step size $\alpha_0$")    
ax.set_ylabel(r"Runtime until convergence [sec]")    

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()