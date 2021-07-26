"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from ..solver.opt_problem import problem, color_dict

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

##########################################################################
## Plotting
##########################################################################

def plot_multiple(allP, ax = None, label = "snspp", name = None, marker = 'o', markersize = 3, ls = '-', lw = 0.4, psi_star = 0, log_scale = False, sigma = 0):
 
  
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

def plot_multiple_error(all_loss, allP, ax = None, label = "snspp", name = None, marker = 'o', markersize = 3, ls = '-', lw = 0.4, log_scale = False, sigma = 0):
    
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
        ax.fill_between(all_rt, y - sigma*all_std, y+sigma*all_std, color = c, alpha = .5)
    
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
    
    params = {'n_epochs' : 2, 'batch_size': 10, 'alpha': 0.01}  
    tmpP = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)   
    tmpP.solve(solver = 'adagrad')
    
    params = {'max_iter' : 15, 'batch_size': 10, 'alpha': 1}  
    tmpP = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)   
    tmpP.solve(solver = 'snspp')
    
    return

def params_tuner(f, phi, solver = 'adagrad', alpha_range = None, batch_range = None, n_iter = 50, relative = True):
    
    if alpha_range is None:
        if solver in ['saga', 'batch saga', 'svrg']:
            # for SAGA/SVRG, input for alpha is multiplied with theoretical stepsize --> choose larger than 1
            alpha_range = np.logspace(0, 2, 10)
    
        else:
            alpha_range = np.logspace(-3, -1, 10)
    
    if batch_range is None:
        if solver == 'saga':
            batch_range = np.array([1])
        else:
            batches = np.array([0.01, 0.03, 0.05])
            batch_range = np.maximum(1, (f.N*batches).astype(int))
            
    all_time_min = np.inf
    current_best = ()
    current_best_val = np.inf    
    
    # initial parameter setup
    if solver == 'snspp':
        params = {'max_iter' : n_iter, 'reduce_variance': True}
    else:
        params =  {'n_epochs' : n_iter}  
        
    res = dict()
    
    
    for b in batch_range:
        res[b] = dict()
        for al in alpha_range:
            
            # update step size and batch size
            if solver == 'snspp':
                params["alpha"] = al
            else:
                params["alpha"] = al
            
            if solver != 'saga':
                params["batch_size"] = b
            print(params)
            
            Q = problem(f, phi, tol = 1e-5, params = params, verbose = False, measure = True)
            
            try: 
                Q.solve(solver = solver)
            except:
                Q.info = dict()
                Q.info["runtime"] = np.nan*np.zeros(params["n_epochs"])
                Q.info["objective"] = np.nan*np.zeros(params["n_epochs"])
                
            # store
            res[b][al] = {'runtime': Q.info["runtime"], 'objective': Q.info["objective"]}
            all_time_min = min(all_time_min, Q.info["objective"].min())
            
            # update current best params if necessary
            this_val = Q.info["objective"][-5:].mean()
            if this_val <= current_best_val:
                current_best = (b, al)
                current_best_val = this_val
            
    # plotting
    fig, ax = plt.subplots(1,1)
    
    markers = ['x', 'o', '^']
    lstyles = ["-", "--", ":"]
    cmap = plt.cm.YlGnBu
    colors = cmap(np.linspace(0,1,len(alpha_range)))

    for j in range(len(batch_range)):
        b = batch_range[j]
        for i in range(len(alpha_range)):  
            
            al = alpha_range[i]
            x = res[b][al]["runtime"].cumsum()
            y = res[b][al]["objective"] - all_time_min*relative
            ax.plot(x, y, color = colors[i], marker = markers[j], ls = lstyles[j], markersize = 6)
    
    
    # legend for batch sizes
    if solver != 'saga':
        b_leg = list()
        for j in range(len(batch_range)):
            b_leg.append(mlines.Line2D([], [], color=colors[-1], marker=markers[j], ls = lstyles[j], markersize=10, \
                                       label= f"b = {batch_range[j]}"))
        
        b_leg = plt.legend(handles = b_leg, title = "batch size", loc = 'upper left')
        ax.add_artist(b_leg)
    
    # legend for step sizes 
    g_leg = list()
    for i in range(len(alpha_range)):
        g_leg.append(mpatches.Patch(color = colors[i], label = np.round(alpha_range[i], 3)))
        
    plt.legend(handles = g_leg, title = "step size (al)", loc = 'upper right')
        
    # other stuff    
    ax.grid(ls = '-', lw = .5)
    ax.set_yscale('log')
    
    ax.set_xlabel('Runtime [sec]')
    if relative:
        ax.set_ylabel('Objective - best observed.')
    else:
        ax.set_ylabel('Objective')
        
    ax.set_title(f'Parameter tuning for {solver}')
            
    return res, current_best, alpha_range

        