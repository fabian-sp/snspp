import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import Lasso


from ssnsp.helper.data_generation import lasso_test, tstudent_test
from ssnsp.solver.opt_problem import problem

N = 1000
n = 1000
k = 20
l1 = 1e-3

problem_type = "lasso"

if problem_type == "lasso":
    xsol, A, b, f, phi,_,_ = lasso_test(N, n, k, l1, block = False, noise = 0.1)
elif problem_type == "tstudent":
    xsol, A, b, f, phi, A_test, b_test = tstudent_test(N, n, k, l1, v = 4, noise = 0.01, scale = 10)

#%% solve with scikt or large max_iter to get psi_star

if problem_type == "lasso":
    sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-8, selection = 'cyclic', max_iter = 1e6)

    start = time.time()
    sk.fit(f.A,b)
    end = time.time()

    print(end-start)
    xsol = sk.coef_.copy()
    psi_star = f.eval(xsol) + phi.eval(xsol)
    print("Optimal value: ", psi_star)

elif problem_type == "tstudent":
    
    ref_params = {'max_iter': 2000, 'alpha_C': 0.1, 'sample_size': 50, 'sample_style': 'constant', 'reduce_variance': True}
    ref_P = problem(f, phi, tol = 1e-6, params = ref_params, verbose = True, measure = True)
    ref_P.solve(solver = 'ssnsp')
    
    ref_P.plot_objective()
    
    psi_star = ref_P.info["objective"][-1]
    
    
#%% do grid testing

ALPHA = np.logspace(-1,3,50)

batch_size = np.array([0.01, 0.05, 0.1])

GRID_A, GRID_B = np.meshgrid(ALPHA, batch_size)

# K ~ len(batch_size), L ~ len(ALPHA)
K,L = GRID_A.shape


psi_tol = 1e-2*psi_star 
EPOCHS = 30

TIME = np.zeros_like(GRID_A)
OBJ = np.ones_like(GRID_A) * 100
CONVERGED = np.zeros_like(GRID_A)


for l in np.arange(L):
    
    for k in np.arange(K):
        
        print("######################################")
        
        params = {'sample_style': 'constant', 'reduce_variance': True}
        
        # target M epochs 
        params["max_iter"] = int(EPOCHS *  1/GRID_B[k,l])
        params['sample_size'] = max(1, int(GRID_B[k,l] * f.N))
        params['alpha_C'] = GRID_A[k,l]
        
        print(f"ALPHA = {params['alpha_C']}")
        print(f"BATCH = {params['sample_size']}")
        
        try:
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
        except:
            obj=[np.inf]
            this_time = np.inf
            
            
            
        OBJ[k,l] = obj[-1]
        TIME[k,l] = this_time
     

OBJ_ERR = (OBJ - psi_star)/psi_star
OBJ_ERR[OBJ_ERR >= 10] = np.nan

CONVERGED = CONVERGED.astype(bool)     

#%% stability test for SAGA / SVRG

GAMMA = np.logspace(-1, 3, 20)

TIME_Q = np.zeros_like(GAMMA)
OBJ_Q = np.zeros_like(GAMMA)
CONVERGED_Q = np.zeros_like(GAMMA)
ALPHA_Q = np.zeros_like(GAMMA)


for l in np.arange(len(GAMMA)):
  print("######################################")
  params_saga = {'n_epochs': EPOCHS, 'batch_size': int(0.01*N), 'gamma': GAMMA[l]}
  
  Q = problem(f, phi, tol = 1e-6, params = params_saga, verbose = False, measure = True)
  Q.solve(solver = 'svrg')
  
  obj = Q.info['objective'].copy()
  
  if np.any(obj <= psi_star + psi_tol):
      stop = np.where(obj <= psi_star + psi_tol)[0][0]
      this_time = Q.info['runtime'].cumsum()[stop]
            
      CONVERGED_Q[l] = 1
  else:
      this_time = Q.info['runtime'].sum()
      print("NO CONVERGENCE!")
            
  OBJ_Q[l] = obj[-1]
  TIME_Q[l] = this_time
  ALPHA_Q[l] = Q.info["step_sizes"][-1]


CONVERGED_Q = CONVERGED_Q.astype(bool)   

#%% 
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

colors = sns.color_palette("GnBu_d", K)
colors = sns.color_palette("viridis_r", K)

#%% plot objective of last iterate vs step size

fig, ax = plt.subplots()

for k in np.arange(K):
    c_arr = np.array(colors[k]).reshape(1,-1)
    #ax.scatter(A[k,:], OBJ_ERR[k,:], c = c_arr, edgecolors = 'k', label = rf"$b =  N \cdot$ {batch_size[k]} ")
    ax.plot(ALPHA, OBJ_ERR[k,:], c = colors[k], linestyle = '--', marker = 'o', label = rf"$b =  N \cdot$ {batch_size[k]} ")

ax.set_xlabel(r"Step size $\alpha$")    
ax.set_ylabel(r"$(\psi(x^k)-\psi^\star)/\psi^\star$")   

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(1e-12,1)
ax.legend()



#%% plot runtime (until convergence) vs step size
save = False

Y_MAX = 3.#TIME.max()*2 +1

fig, ax = plt.subplots(figsize = (7,5))

for k in np.arange(K):    
    TIME[k,:][~CONVERGED[k,:]] = Y_MAX    
    
    ax.plot(ALPHA, TIME[k,:], c = colors[k], linestyle = '--', marker = 'o', markersize = 4,  label = rf"$b =  N \cdot$ {batch_size[k]} ")

TIME_Q[~CONVERGED_Q] =  Y_MAX    
ax.plot(ALPHA_Q, TIME_Q, c = "#FFB03B", linestyle = '--', marker = 'o', markersize = 4,  label = Q.solver)
    

ax.hlines( Y_MAX-1 , ALPHA[0], ALPHA[-1], 'grey', ls = '-')
ax.annotate("no convergence", (ALPHA[0], Y_MAX-.9 ), color = "grey", fontsize = 14)

ax.set_xlabel(r"Step size $\alpha$")    
ax.set_ylabel(r"Runtime until convergence [sec]")    

ax.set_xscale('log')
#ax.set_yscale('log')

ax.legend()

if save:
    fig.savefig(f'data/plots/exp_other/step_size_tuning.pdf', dpi = 300)
