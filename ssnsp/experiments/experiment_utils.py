"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt
from ..solver.opt_problem import problem, color_dict

def plot_multiple(allP, ax = None, label = "ssnsp", name = None, psi_star = 0, log_scale = False):
    
    if name is None:
        name = label
    
    if ax is None:
        fig, ax = plt.subplots()
            
    K = len(allP)
    
    all_obj = np.vstack([allP[k]["objective"] for k in range(K)])
    
    all_obj = all_obj - psi_star
    all_mean = all_obj.mean(axis=0)
    all_std = all_obj.std(axis=0)
    

    all_rt = np.vstack([allP[k]["runtime"] for k in range(K)]).mean(axis=0).cumsum()
    
    try:
        c = color_dict[label]
    except:
        c = color_dict["default"]
    
    sigma = 1.
    ax.plot(all_rt, all_mean, marker = 'o', markersize = 4, color = c, label = name)
    ax.fill_between(all_rt, \
                    all_mean -sigma*all_std\
                    , all_mean+sigma*all_std, \
                    color = c, alpha = .5)
    
    
    if log_scale:
        ax.set_yscale('log')
            
    return

def initialize_fast_gradient(f, phi):
    """
    initializes SAGA and Adagrad jitiing
    """
    params = {'n_epochs' : 10}
    tmpP = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)
    tmpP.solve(solver = 'saga')
    
    params = {'n_epochs' : 10, 'batch_size': int(f.N*0.05), 'gamma': 0.01}  
    tmpP = problem(f, phi, tol = 1e-5, params = params, verbose = True, measure = True)   
    tmpP.solve(solver = 'adagrad')
    
    return

def adagrad_step_size_tuner(f, phi, gamma_range = None, params = None):
    """
    performs step size tuning for ADAGRAD
    either provide range of gamma values or it is automatically tuned on log-scale (see range below)
    """
    if params is None:
        params = {'n_epochs' : 200, 'batch_size': min(2000, int(0.05*f.N))}
    
    if gamma_range is None:
        
        gamma_range = np.logspace(-3,-1,12)
        
    K = len(gamma_range)
    all_obj = np.zeros(K)
    
    fig, axs = plt.subplots(3,4)
    for k in range(K):
        
        params["gamma"] = gamma_range[k]
        
        print("Step size: ", gamma_range[k])
        
        Q1 = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)
        Q1.solve(solver = 'adagrad')

        this_obj = f.eval(Q1.x) +phi.eval(Q1.x)
        print(this_obj)
        
        all_obj[k] = this_obj
        
        Q1.plot_objective(ax = axs.ravel()[k])
        
    opt_gamma = gamma_range[np.argmin(all_obj)]
    print("Optimal step size: ", opt_gamma)
    
    return opt_gamma, gamma_range, all_obj