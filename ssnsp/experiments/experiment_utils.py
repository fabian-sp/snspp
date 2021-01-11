"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt
from ..solver.opt_problem import problem, color_dict

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

##########################################################################
## Plotting
##########################################################################

def plot_multiple(allP, ax = None, label = "ssnsp", name = None, marker = 'o', markersize = 3, ls = '-', lw = 0.4, psi_star = 0, log_scale = False, sigma = 0):
 
  
    if name is None:
        name = label
    
    if ax is None:
        fig, ax = plt.subplots()
            
    K = len(allP)
    
    for k in range(K):
        assert allP[k].solver == label, "solver attribute and label are not matching!"
    
    all_obj = np.vstack([allP[k].info["objective"] for k in range(K)])
    
    all_obj = all_obj - psi_star
    all_mean = all_obj.mean(axis=0)
    all_std = all_obj.std(axis=0)
    
    all_rt = np.vstack([allP[k].info["runtime"] for k in range(K)]).mean(axis=0).cumsum()
    
    try:
        c = color_dict[label]
    except:
        c = color_dict["default"]
    
    ax.plot(all_rt, all_mean, marker = marker, ls = ls, markersize = markersize, color = c, label = name)
    
    # plot band of standard deviation
    if sigma > 0:
        ax.fill_between(all_rt, all_mean - sigma*all_std, all_mean+sigma*all_std, color = c, alpha = .5)
    
    ax.grid(ls = '-', lw = .5) 
    ax.set_xlabel("Runtime [sec]", fontsize = 12)
    
    if psi_star == 0:
        ax.set_ylabel(r"$\psi(x^k)$", fontsize = 12)
    else:
        ax.set_ylabel(r"$\psi(x^k) - \psi^\star$", fontsize = 12)

    if log_scale:
        ax.set_yscale('log')
            
    return


def plot_test_error(P, L,  ax = None, name = None, marker = 'o', markersize = 3, ls = '-', lw = 0.4, log_scale = True):
    
    assert len(P.info["iterates"]) == len(L), "the vector of losses has a different length than the stored iterates"
    
    x = P.info["runtime"].cumsum()
    y = L.copy()
    
    if name is None:
        name = P.solver
        
    try:
        c = color_dict[P.solver]
    except:
        c = color_dict["default"]
        
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(x, y, marker = marker, lw = lw, markersize = markersize, color = c, label = name)
    
    ax.grid(ls = '-', lw = .5) 
    ax.set_xlabel("Runtime [sec]", fontsize = 12)
    ax.set_ylabel("Test error", fontsize = 12)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend(fontsize = 10)
    
    return

def plot_multiple_error(all_loss, allP, ax = None, label = "ssnsp", name = None, marker = 'o', markersize = 3, ls = '-', lw = 0.4, log_scale = False, sigma = 0):
    
    if name is None:
        name = label

    if ax is None:
        fig, ax = plt.subplots()
            
    K = len(allP)
    
    for k in range(K):
        assert allP[k].solver == label, "solver attribute and label are not matching!"
    
    y = all_loss.mean(axis=0)
    all_std = all_loss.std(axis=0)
    all_rt = np.vstack([allP[k].info["runtime"] for k in range(K)]).mean(axis=0).cumsum()
    
    try:
        c = color_dict[label]
    except:
        c = color_dict["default"]
    
    ax.plot(all_rt, y, marker = marker, ls = ls, lw = lw, markersize = markersize, color = c, label = name)
    
    # plot band of standard deviation
    if sigma > 0:
        ax.fill_between(all_rt, all_mean - sigma*all_std, all_mean+sigma*all_std, color = c, alpha = .5)
    
    ax.grid(ls = '-', lw = .5) 
    ax.set_xlabel("Runtime [sec]", fontsize = 12)
    
    ax.set_ylabel(r"Test error", fontsize = 12)
    
    if log_scale:
        ax.set_yscale('log')
            
    return

def eval_test_set(X, loss, **kwargs):
    """
    evaluates a given loss function on each row of X
    """
    L = np.zeros(len(X))
    
    for j in range(len(X)):
        L[j] = loss(x=X[j,:], **kwargs)
 
    return L



##########################################################################
## Fast gradient methods utils
##########################################################################

def initialize_solvers(f, phi):
    """
    initializes jitiing
    """
    params = {'n_epochs' : 2}
    tmpP = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)
    tmpP.solve(solver = 'saga')
    
    params = {'n_epochs' : 2}
    tmpP = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)
    tmpP.solve(solver = 'svrg')
    
    params = {'n_epochs' : 2, 'batch_size': 10, 'gamma': 0.01}  
    tmpP = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)   
    tmpP.solve(solver = 'adagrad')
    
    params = {'max_iter' : 5, 'sample_size': 10, 'alpha_C': 0.01}  
    tmpP = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)   
    tmpP.solve(solver = 'ssnsp')
    
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
        axs.ravel()[k].set_title(f"step size {gamma_range[k]}")
        
    opt_gamma = gamma_range[np.argmin(all_obj)]
    print("Optimal step size: ", opt_gamma)
    
    return opt_gamma, gamma_range, all_obj