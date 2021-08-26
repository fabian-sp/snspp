import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import Lasso, LogisticRegression


from snspp.helper.data_generation import lasso_test, logreg_test, get_gisette
from snspp.solver.opt_problem import problem, color_dict
from snspp.experiments.experiment_utils import initialize_solvers

N = 2000
n = 1000
k = 20
l1 = 0.01


# setup of parameters

EPOCHS = 50 # epochs for SAGA/SVRG
MAX_ITER = 150 # max iter for SNSPP
PSI_TOL = 1e-5 # relative accuracy for objective to be considered as converged
N_REP = 5 # number of repetitions for each setting
Y_MAX = 1. # y-value of not-converged stepsizes

#%%

problem_type = "logreg"

if problem_type == "lasso":
    xsol, A, b, f, phi, _, _ = lasso_test(N, n, k, l1, block = False, noise = 0.1, kappa = 15., dist = 'ortho')

elif problem_type == "logreg":
    xsol, A, b, f, phi, _, _ = logreg_test(N, n, k, lambda1 = l1, noise = 0.1, kappa = 10., dist = 'ortho')

elif problem_type == "gisette":
    f, phi, A, b, _, _ = get_gisette(lambda1 = 0.01)

initialize_solvers(f, phi)

#%% solve with scikt or large max_iter to get psi_star

if problem_type == "lasso":
    sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-8, selection = 'cyclic', max_iter = 1e6)
    sk.fit(f.A,b)

    xsol = sk.coef_.copy()
    psi_star = f.eval(xsol) + phi.eval(xsol)
    print("Optimal value: ", psi_star)

elif problem_type in ["logreg", "gisette"]:
    sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-20, \
                            solver = 'saga', max_iter = 200, verbose = 1)
    sk.fit(A, b)
    
    xsol = sk.coef_.copy().squeeze()
    psi_star = f.eval(xsol) + phi.eval(xsol)
    print("Optimal value: ", psi_star)
    
    
# elif problem_type == "tstudent":
#     ref_params = {'n_epochs': 100, 'alpha': 2}
#     ref_P = problem(f, phi, tol = 1e-6, params = ref_params, verbose = True, measure = True)
#     ref_P.solve(solver = 'saga')
#     ref_P.plot_objective()
#     psi_star = ref_P.info["objective"][-1]
#     print("Optimal value: ", psi_star)
    
#%%

def do_grid_run(f, phi, step_size_range, batch_size_range = None, psi_star = 0, psi_tol = 1e-3, n_rep = 5, \
                solver = "snspp", solver_params = dict()):
    
    ALPHA = step_size_range.copy()
    
    if batch_size_range is None:
        batch_size_range = np.array([1/f.N])
    BATCH = batch_size_range.copy()
    
    GRID_A, GRID_B = np.meshgrid(ALPHA, BATCH)
    
    # K ~ len(batch_size_range), L ~ len(step_size_range)
    K,L = GRID_A.shape
        
    RTIME = np.ones_like(GRID_A) * np.inf
    OBJ = np.ones_like(GRID_A) * np.inf
    CONVERGED = np.zeros_like(GRID_A)
    NITER = np.ones_like(GRID_A) * np.inf
    
    for l in np.arange(L):
        
        for k in np.arange(K):
            this_params = solver_params.copy()
            
            print("######################################")
            
            # target M epochs 
            #if solver == "snspp":
            #    this_params["max_iter"] = 200 #int(EPOCHS *  1/GRID_B[k,l])
            
            this_params['batch_size'] = max(1, int(GRID_B[k,l] * f.N))
            this_params['alpha'] = GRID_A[k,l]
            
            print(f"ALPHA = {this_params['alpha']}")
            print(f"BATCH = {this_params['batch_size']}")
            
            # repeat n_rep times
            this_obj = list(); this_time = list(); this_stop_iter = list()
            for j_rep in np.arange(n_rep):
                try:
                    # SOLVE
                    P = problem(f, phi, tol = 1e-20, params = this_params, verbose = False, measure = True)
                    P.solve(solver = solver)
                          
                    obj_arr = P.info['objective'].copy()
                    
                    print(f"OBJECTIVE = {obj_arr[-1]}")
                    this_alpha = P.info["step_sizes"][-1]
                    
                    if np.any(obj_arr <= psi_star *(1+psi_tol)):
                        stop = np.where(obj_arr <= psi_star *(1+psi_tol))[0][0]
                        this_stop_iter.append(stop)
                        this_time.append(P.info['runtime'].cumsum()[stop])
                        this_obj.append(obj_arr[-1])
                        
                    else:
                        this_stop_iter.append(np.inf)
                        this_time.append(np.inf)
                        this_obj.append(obj_arr[-1])
                        
                        print("NO CONVERGENCE!")
                except:
                    this_stop_iter.append(np.inf)
                    this_time.append(np.inf)
                    this_obj.append(np.inf)
                    this_alpha = np.nan
            
            # set as CONVERGED if all repetiitions converged
            CONVERGED[k,l] = np.all(~np.isinf(this_stop_iter))
            OBJ[k,l] = np.mean(this_obj)
            RTIME[k,l] = np.mean(this_time)
            NITER[k,l] = np.mean(this_stop_iter)
            
            # TO DO: fix if run into exception
            ALPHA[l] = this_alpha
    
    CONVERGED = CONVERGED.astype(bool)     
    
    assert np.all(~np.isinf(RTIME) == CONVERGED), "Runtime and convergence arrays are incosistent!"
    assert np.all(~np.isnan(ALPHA)), "actual step size not available"
    
    results = {'step_size': ALPHA, 'batch_size': BATCH, 'objective': OBJ, 'runtime': RTIME,\
               'n_iter': NITER, 'converged': CONVERGED, 'solver': solver}
    
    return results

def plot_result(res, ax = None, color = 'k', replace_inf = 10.):
    
    K,L = res['runtime'].shape
    rt = res['runtime'].copy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (7,5))
    
    colors = sns.light_palette(color, K+2, reverse = True)
    
    for k in np.arange(K):    
        rt[k,:][~res['converged'][k,:]] = replace_inf   
        
        if K == 1:
            label = res['solver']
        else:
            label = res['solver'] + ", " + rf"$b =  N \cdot$ {res['batch_size'][k]} "
        
        ax.plot(res['step_size'], rt[k,:], c = colors[k], linestyle = '--', marker = 'o', markersize = 4,\
                label = label)
    
      
    ax.set_xlabel(r"Step size $\alpha$")    
    ax.set_ylabel(r"Runtime until convergence [sec]")    
    
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend()

    return ax

#%%

solver_params = {'max_iter': MAX_ITER, 'sample_style': 'constant', 'reduce_variance': True}

step_size_range = np.logspace(-2,3,20)
batch_size_range = np.array([0.01,0.05,0.1])

res_spp = do_grid_run(f, phi, step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                      psi_tol = PSI_TOL, n_rep = N_REP, solver = "snspp", solver_params = solver_params)


#%%

solver_params = {'n_epochs': EPOCHS}

step_size_range = np.logspace(-2,3,20)
batch_size_range = None

res_saga = do_grid_run(f, phi, step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                       psi_tol = PSI_TOL, n_rep = N_REP, solver = "saga", solver_params = solver_params)


#%%

solver_params = {'n_epochs': EPOCHS}

step_size_range = np.logspace(-2,3,20)
batch_size_range = np.array([0.01,0.05,0.1])

res_svrg = do_grid_run(f, phi, step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                       psi_tol = PSI_TOL, n_rep = N_REP, solver = "svrg", solver_params = solver_params)

    
#%% plot runtime (until convergence) vs step size
save = False

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

#%%

fig, ax = plt.subplots(figsize = (7,5))

plot_result(res_spp, ax = ax, color = color_dict["snspp"], replace_inf = Y_MAX)
plot_result(res_saga, ax = ax, color = color_dict["saga"], replace_inf = Y_MAX)
plot_result(res_svrg, ax = ax, color = color_dict["svrg"], replace_inf = Y_MAX)


annot_y = Y_MAX * 0.9 # y value for annotation

ax.hlines(annot_y , ax.get_xlim()[0], ax.get_xlim()[1], 'grey', ls = '-')
ax.annotate("no convergence", (ax.get_xlim()[0]*1.1, annot_y+0.1), color = "grey", fontsize = 14)

if save:
    fig.savefig(f'data/plots/exp_other/step_size_tuning_{problem_type}.pdf', dpi = 300)

    
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


