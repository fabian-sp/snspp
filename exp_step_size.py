import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import Lasso, LogisticRegression


from snspp.helper.data_generation import lasso_test, logreg_test, get_gisette
from snspp.solver.opt_problem import problem, color_dict

N = 1000
n = 40
k = 5
l1 = 1e-3

EPOCHS = 40
problem_type = "gisette"

if problem_type == "lasso":
    xsol, A, b, f, phi, _, _ = lasso_test(N, n, k, l1, block = False, noise = 0.1, kappa = 15., dist = 'ortho')

elif problem_type == "logreg":
    xsol, A, b, f, phi, _, _ = logreg_test(N, n, k, lambda1 = l1, noise = 0.3, kappa = 15., dist = 'ortho')

elif problem_type == "gisette":
    f, phi, A, b, _, _ = get_gisette(lambda1 = 0.05)

#%% solve with scikt or large max_iter to get psi_star

if problem_type == "lasso":
    sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-8, selection = 'cyclic', max_iter = 1e6)
    sk.fit(f.A,b)

    xsol = sk.coef_.copy()
    psi_star = f.eval(xsol) + phi.eval(xsol)
    print("Optimal value: ", psi_star)

elif problem_type in ["logreg", "gisette"]:
    sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-9, \
                            solver = 'saga', max_iter = 200, verbose = 1)
    sk.fit(A, b)
    
    xsol = sk.coef_.copy().squeeze()
    psi_star = f.eval(xsol) + phi.eval(xsol)
    print("Optimal value: ", psi_star)
    
    P = problem(f, phi, tol = 1e-6, params = {'n_epochs': 200, 'alpha': 1.}, verbose = False, measure = False)
    P.solve(solver = 'saga')
    
    print(np.linalg.norm(P.x-xsol))
    

# elif problem_type == "tstudent":
    
#     ref_params = {'max_iter': 2000, 'alpha': 0.1, 'batch_size': 50, 'sample_style': 'constant', 'reduce_variance': True}
#     ref_P = problem(f, phi, tol = 1e-6, params = ref_params, verbose = True, measure = True)
#     ref_P.solve(solver = 'snspp')
    
#     ref_P.plot_objective()
    
#     psi_star = ref_P.info["objective"][-1]
#     print("Optimal value: ", psi_star)
    

#%%
def do_grid_run(step_size_range, batch_size_range = None, psi_star = 0, solver = "snspp", solver_params = dict()):
    
    ALPHA = step_size_range
    if batch_size_range is None:
        batch_size_range = np.array([1/f.N])
    BATCH = batch_size_range
    
    GRID_A, GRID_B = np.meshgrid(ALPHA, BATCH)
    
    # K ~ len(batch_size_range), L ~ len(step_size_range)
    K,L = GRID_A.shape
        
    psi_tol = 1e-2
    
    TIME = np.zeros_like(GRID_A)
    OBJ = np.ones_like(GRID_A) * 100
    CONVERGED = np.zeros_like(GRID_A)
    
    
    for l in np.arange(L):
        
        for k in np.arange(K):
            
            print("######################################")
            
            # target M epochs 
            if solver == "snspp":
                solver_params["max_iter"] = 200
            else:
                solver_params["max_iter"] = int(EPOCHS *  1/GRID_B[k,l])
                
                
            solver_params['batch_size'] = max(1, int(GRID_B[k,l] * f.N))
            solver_params['alpha'] = GRID_A[k,l]
            
            print(f"ALPHA = {solver_params['alpha']}")
            print(f"BATCH = {solver_params['batch_size']}")
            print(f"MAX_ITER = {solver_params['max_iter']}")
            try:
                P = problem(f, phi, tol = 1e-6, params = solver_params, verbose = False, measure = True)
                P.solve(solver = solver)
                      
                #xhist = P.info['iterates'].copy()
                obj = P.info['objective'].copy()
                
                print(f"OBJECTIVE = {obj[-1]}")
                
                if np.any(obj <= psi_star *(1+psi_tol)):
                    stop = np.where(obj <= psi_star *(1+psi_tol))[0][0]
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
            ALPHA[l] = P.info["step_sizes"][-1]
         
    
    OBJ_ERR = (OBJ - psi_star)/psi_star
    OBJ_ERR[OBJ_ERR >= 10] = np.nan
    
    CONVERGED = CONVERGED.astype(bool)     
    
    return OBJ, TIME, CONVERGED, ALPHA, BATCH

#%%


solver_params = {'sample_style': 'fast_increasing', 'reduce_variance': True, 'm_iter': 10}

step_size_range = np.logspace(-2, 2, 10)
batch_size_range = np.array([0.01, 0.05, 0.1])

obj, time, conv, alpha, batch = do_grid_run(step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                                            solver = "snspp", solver_params = solver_params)


#%%

solver_params = {'n_epochs': EPOCHS}

step_size_range = np.logspace(-2,3,20)
batch_size_range = None

obj1, time1, conv1, alpha1, batch1 = do_grid_run(step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                                                 solver = "saga", solver_params = solver_params)



#%% plot runtime (until convergence) vs step size
save = False

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

Y_MAX = 3.

def plot_result(TIME, CONVERGED, ALPHA, BATCH, ax = None, color = 'k', solver = "snspp"):
    K,L = TIME.shape
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (7,5))
    
    colors = sns.light_palette(color, K+2, reverse = True)
    
    for k in np.arange(K):    
        TIME[k,:][~CONVERGED[k,:]] = Y_MAX    
        ax.plot(ALPHA, TIME[k,:], c = colors[k], linestyle = '--', marker = 'o', markersize = 4,\
                label = solver + ", " + rf"$b =  N \cdot$ {BATCH[k]} ")
    
      
    
    ax.set_xlabel(r"Step size $\alpha$")    
    ax.set_ylabel(r"Runtime until convergence [sec]")    
    
    ax.set_xscale('log')
    #ax.set_yscale('log')
    
    ax.legend()

    return ax


fig, ax = plt.subplots(figsize = (7,5))

plot_result(time, conv, alpha, batch, ax = ax, color = color_dict["snspp"], solver = "snspp")
plot_result(time1, conv1, alpha1, batch1, ax = ax, color = color_dict["saga"], solver = "saga")

ax.hlines(Y_MAX-1 , ax.get_xlim()[0], ax.get_xlim()[1], 'grey', ls = '-')
ax.annotate("no convergence", (ax.get_xlim()[0]*1.1, Y_MAX-.9 ), color = "grey", fontsize = 14)

if save:
    fig.savefig(f'data/plots/exp_other/step_size_tuning.pdf', dpi = 300)

    
#%% plot objective of last iterate vs step size

# fig, ax = plt.subplots()

# for k in np.arange(K):
#     c_arr = np.array(colors[k]).reshape(1,-1)
#     #ax.scatter(A[k,:], OBJ_ERR[k,:], c = c_arr, edgecolors = 'k', label = rf"$b =  N \cdot$ {batch_size[k]} ")
#     ax.plot(ALPHA, OBJ_ERR[k,:], c = colors[k], linestyle = '--', marker = 'o', label = rf"$b =  N \cdot$ {batch_size[k]} ")

# ax.set_xlabel(r"Step size $\alpha$")    
# ax.set_ylabel(r"$(\psi(x^k)-\psi^\star)/\psi^\star$")   

# ax.set_xscale('log')
# ax.set_yscale('log')

# ax.set_ylim(1e-12,1)
# ax.legend()



#%% do grid testing

# ALPHA = np.logspace(-2,1,20)

# batch_size = np.array([0.01, 0.05, 0.1])

# GRID_A, GRID_B = np.meshgrid(ALPHA, batch_size)

# # K ~ len(batch_size), L ~ len(ALPHA)
# K,L = GRID_A.shape


# psi_tol = 1e-3*psi_star 


# TIME = np.zeros_like(GRID_A)
# OBJ = np.ones_like(GRID_A) * 100
# CONVERGED = np.zeros_like(GRID_A)


# for l in np.arange(L):
    
#     for k in np.arange(K):
        
#         print("######################################")
        
#         params = {'sample_style': 'constant', 'reduce_variance': True, 'm_iter':10}
        
#         # target M epochs 
#         params["max_iter"] = int(EPOCHS *  1/GRID_B[k,l])
#         # m = 10 in SSNSP
#         params["max_iter"] = int( EPOCHS * (GRID_B[k,l] + 1/params['m_iter'])**(-1)  )
#         params['batch_size'] = max(1, int(GRID_B[k,l] * f.N))
#         params['alpha'] = GRID_A[k,l]
        
#         print(f"ALPHA = {params['alpha']}")
#         print(f"BATCH = {params['batch_size']}")
#         print(f"MAX_ITER = {params['max_iter']}")
#         try:
#             P = problem(f, phi, tol = 1e-6, params = params, verbose = False, measure = True)
#             P.solve(solver = 'snspp')
            
            
#             xhist = P.info['iterates'].copy()
#             obj = P.info['objective'].copy()
            
#             print(f"OBJECTIVE = {obj[-1]}")
            
#             if np.any(obj <= psi_star + psi_tol):
#                 stop = np.where(obj <= psi_star + psi_tol)[0][0]
#                 this_time = P.info['runtime'].cumsum()[stop]
                
#                 CONVERGED[k,l] = 1
#             else:
#                 this_time = P.info['runtime'].sum()
#                 print("NO CONVERGENCE!")
#         except:
#             obj=[np.inf]
#             this_time = np.inf
            
            
            
#         OBJ[k,l] = obj[-1]
#         TIME[k,l] = this_time
     

# OBJ_ERR = (OBJ - psi_star)/psi_star
# OBJ_ERR[OBJ_ERR >= 10] = np.nan

# CONVERGED = CONVERGED.astype(bool)     

# #%% stability test for SAGA / SVRG

# GAMMA = np.logspace(-1, 1, 20)

# TIME_Q = np.zeros_like(GAMMA)
# OBJ_Q = np.zeros_like(GAMMA)
# CONVERGED_Q = np.zeros_like(GAMMA)
# ALPHA_Q = np.zeros_like(GAMMA)


# for l in np.arange(len(GAMMA)):
#     print("######################################")
#     params_saga = {'n_epochs': EPOCHS, 'batch_size': int(0.05*N), 'alpha': GAMMA[l]}
    
#     Q = problem(f, phi, tol = 1e-6, params = params_saga, verbose = False, measure = True)
#     Q.solve(solver = 'saga')
    
#     obj = Q.info['objective'].copy()
    
#     if np.any(obj <= psi_star + psi_tol):
#         stop = np.where(obj <= psi_star + psi_tol)[0][0]
#         this_time = Q.info['runtime'].cumsum()[stop]
              
#         CONVERGED_Q[l] = 1
#     else:
#         this_time = Q.info['runtime'].sum()
#         print("NO CONVERGENCE!")
              
#     OBJ_Q[l] = obj[-1]
#     TIME_Q[l] = this_time
#     ALPHA_Q[l] = Q.info["step_sizes"][-1]


# CONVERGED_Q = CONVERGED_Q.astype(bool)   
