"""
@author: Fabian Schaipp
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import Lasso, LogisticRegression

from snspp.helper.data_generation import lasso_test, logreg_test, get_gisette, get_mnist
from snspp.solver.opt_problem import problem, color_dict, marker_dict
from snspp.experiments.experiment_utils import initialize_solvers

setup_id = 'gisette1'

def load_setup(setup_id = ''):
    
    f = open('data/setups/' + setup_id + '.json',)
    setup = json.load(f)

    return setup

def create_instance(setup = dict()):
    
    if setup['instance']['dataset'] == "lasso":
        xsol, A, b, f, phi, _, _ = lasso_test(setup['instance']['N'], setup['instance']['n'], setup['instance']['k'], setup['instance']['l1'], \
                                              noise = 0.1, kappa = 15., dist = 'ortho')

    elif setup['instance']['dataset'] == "logreg":
        xsol, A, b, f, phi, _, _ = logreg_test(setup['instance']['N'], setup['instance']['n'], setup['instance']['k'], lambda1 = setup['instance']['l1'],\
                                               noise = 0.1, kappa = 15., dist = 'ortho')
    
    elif setup['instance']['dataset'] == "gisette":
        f, phi, A, b, _, _ = get_gisette(lambda1 = setup['instance']['l1'])
    
    elif setup['instance']['dataset'] == "mnist":
        f, phi, A, b, _, _ = get_mnist(lambda1 = setup['instance']['l1'])
    
    initialize_solvers(f, phi)

    return f, phi



#%% MAIN FUNCTION

def do_grid_run(f, phi, step_size_range, batch_size_range = [], psi_star = 0, psi_tol = 1e-3, n_rep = 5, \
                solver = "snspp", x0 = None, solver_params = dict()):
    
    ALPHA = step_size_range.copy()
    
    if len(batch_size_range) == 0:
        batch_size_range = [1/f.N]
    BATCH = batch_size_range.copy()
    
    GRID_A, GRID_B = np.meshgrid(ALPHA, BATCH)
    
    # K ~ len(batch_size_range), L ~ len(step_size_range)
    K,L = GRID_A.shape
        
    RTIME = np.ones_like(GRID_A) * np.inf
    RT_STD = np.ones_like(GRID_A) * np.inf
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
                    P = problem(f, phi, x0 = x0, tol = 1e-20, params = this_params, verbose = False, measure = True)
                    P.solve(solver = solver)
                          
                    obj_arr = P.info['objective'].copy()
                    
                    print(f"OBJECTIVE = {obj_arr[-1]}")
                    this_alpha = P.info["step_sizes"][-1]
                    
                    if np.any(obj_arr <= psi_star *(1+psi_tol)):
                        stop = np.where(obj_arr <= psi_star *(1+psi_tol))[0][0]
                        
                        # account for possibility of reaching accuracy inside the epoch --> take lower bound for runtime
                        if solver != 'snspp':
                            if stop > 0:
                                stop -= 1
                            else:
                                print("Convergence during first EPOCH!")
                        
                        this_stop_iter.append(stop)
                        this_time.append(P.info['runtime'].cumsum()[stop])
                        this_obj.append(obj_arr[-1])
                        
                    else:
                        this_stop_iter.append(np.inf)
                        this_time.append(np.inf)
                        this_obj.append(obj_arr[-1])
                        
                        print("NO CONVERGENCE!")
                except:
                    print("SOLVER FAILED!")
                    this_stop_iter.append(np.inf)
                    this_time.append(np.inf)
                    this_obj.append(np.inf)
                    this_alpha = np.nan
            
            # set as CONVERGED if all repetiitions converged
            CONVERGED[k,l] = np.all(~np.isinf(this_stop_iter))
            OBJ[k,l] = np.mean(this_obj)
            RTIME[k,l] = np.mean(this_time)
            RT_STD[k,l] = np.std(this_time)
            NITER[k,l] = np.mean(this_stop_iter)
            
            # TO DO: fix if run into exception
            ALPHA[l] = this_alpha
    
    CONVERGED = CONVERGED.astype(bool)     
    
    assert np.all(~np.isinf(RTIME) == CONVERGED), "Runtime and convergence arrays are incosistent!"
    assert np.all(~np.isnan(ALPHA)), "actual step size not available"
    
    results = {'step_size': ALPHA, 'batch_size': BATCH, 'objective': OBJ, 'runtime': RTIME, 'runtime_std': RT_STD,\
               'n_iter': NITER, 'converged': CONVERGED, 'solver': solver}
    
    return results


#%% PLOTTING


def plot_result(res, ax = None, replace_inf = 10., sigma = 0., psi_tol = 1e-3):
    
    K,L = res['runtime'].shape
    rt = res['runtime'].copy()
    rt_std = res['runtime_std'].copy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (7,5))
    
    try:
        color = color_dict[res["solver"]]
        marker = marker_dict[res["solver"]]
    except:
        color = color_dict["default"]
        marker = marker_dict["default"]

    colors = sns.light_palette(color, K+2, reverse = True)
    
    for k in np.arange(K):    
        rt[k,:][~res['converged'][k,:]] = replace_inf
        rt_std[k,:][~res['converged'][k,:]] = 0 
        
        if K == 1:
            label = res['solver']
        else:
            label = res['solver'] + ", " + rf"$b =  N \cdot${res['batch_size'][k]} "
        
        ax.plot(res['step_size'], rt[k,:], c = colors[k], linestyle = '-', marker = marker, markersize = 4,\
                label = label)
        
        # add standard dev of runtime
        if sigma > 0.:
            ax.fill_between(res['step_size'], rt[k,:] - sigma*rt_std[k,:], rt[k,:] + sigma*rt_std[k,:],\
                            color = colors[k], alpha = 0.5)
            
    
    ax.set_xlabel(r"Step size $\alpha$")    
    ax.set_ylabel(r"Runtime until convergence [sec]")    
    
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend(loc = 'lower left', fontsize = 8)
    ax.set_title(rf'Convergence = objective less than {1+psi_tol}$\psi^\star$')

    return ax
