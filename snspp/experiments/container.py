"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt

from ..solver.opt_problem import problem, color_dict, marker_dict

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)


class Experiment:
    """
        
    """
    def __init__(self, name = ''):
        """

        Returns
        -------
        None.

        """
        
        self.name = name
        self.solvers = list()
        self.results = dict()
        
    def store(self, P = problem, k = 0, suffix = ''):
        
        label = P.solver + suffix
        
        # initialize
        if label not in self.solvers:
            self.solvers.append(label)
            self.results[label] = dict()
        
        self.results[label][k] = dict()
        
        self.results[label][k]['runtime'] = P.info['runtime'].copy()
        self.results[label][k]['objective'] = P.info['objective'].copy()
        self.results[label][k]['evaluations'] = P.info['evaluations'].copy()
        
        return
    
    def store_by_key(self, res = dict(), label = '', k = 0):
        
        for key, val in res.items():
            assert key not in self.results[label][k].keys()
            self.results[label][k][key] = val
            
        return
    
    def save_to_disk(self, path = ''):           
        np.save(path + self.name + '.npy', self.results)
        return

#########################################################################
#########################################################################
#########################################################################
## PLOTTING

#########################################################################
#########################################################################
#########################################################################

    
    def plot_objective(self, ax = None, runtime = True, median = False, markersize = 3, ls = '-', lw = 0.4, psi_star = 0, log_scale = False, sigma = 0):
 
      
        if ax is None:
            fig, ax = plt.subplots()
        
        for s in self.solvers:
        
            this_res = self.results[s]
            K = len(this_res.keys())
        
            all_obj = np.vstack([this_res[k]["objective"] for k in range(K)])
            
            all_obj = all_obj - psi_star
            if median:
                y = np.median(all_obj, axis=0)
            else:
                y = all_obj.mean(axis=0)
                
            all_std = all_obj.std(axis=0)
            
            if runtime:
                all_xax = np.vstack([this_res[k]["runtime"] for k in range(K)]).mean(axis=0).cumsum()
            else: 
                all_xax = np.vstack([this_res[k]["evaluations"] for k in range(K)]).mean(axis=0).cumsum() 
                
            try:
                c = color_dict[s]
                marker = marker_dict[s]
            except:
                c = color_dict["default"]
                marker = marker_dict["default"]
        
            ax.plot(all_xax, y, marker = marker, ls = ls, markersize = markersize, color = c, label = s)
            
            # plot band of standard deviation
            if sigma > 0:
                ax.fill_between(all_xax, y-sigma*all_std, y+sigma*all_std, color = c, alpha = .5)
            
        ax.grid(ls = '-', lw = .5) 
        
        if runtime:
            ax.set_xlabel("Runtime [sec]", fontsize = 12)
        else:
            ax.set_xlabel(r"Evaluations/$N$", fontsize = 12)
        
        if psi_star == 0:
            ax.set_ylabel(r"$\psi(x^k)$", fontsize = 12)
        else:
            ax.set_ylabel(r"$\psi(x^k) - \psi^\star$", fontsize = 12)
    
        if log_scale:
            ax.set_yscale('log')
            
        return
    
#########################################################################
#########################################################################
#########################################################################

    def plot_error(self, error_key = '', ax = None, runtime = True, median = True, markersize = 3, ls = '-', lw = 0.4, log_scale = False, sigma = 0, ylabel = None):
        
        if ax is None:
            fig, ax = plt.subplots()
                
        for s in self.solvers:
        
            this_res = self.results[s]
            K = len(this_res.keys())
            
            all_error = np.vstack([this_res[k][error_key] for k in range(K)])
            
            ## Y VAL
            if median:
                y = np.median(all_error, axis=0)
            else:
                y = all_error.mean(axis=0)
                
            all_std = all_error.std(axis=0)
            
            ## X VAL
            if runtime:
                all_xax = np.vstack([this_res[k]["runtime"] for k in range(K)]).mean(axis=0).cumsum()
            else: 
                all_xax = np.vstack([this_res[k]["evaluations"] for k in range(K)]).mean(axis=0).cumsum()
             
            try:
                c = color_dict[s]
                marker = marker_dict[s]
            except:
                c = color_dict["default"]
                marker = marker_dict["default"]
            
            ax.plot(all_xax, y, marker = marker, ls = ls, lw = lw, markersize = markersize, color = c, label = s)
            
            # plot band of standard deviation
            if sigma > 0:
                ax.fill_between(all_xax, y - sigma*all_std, y+sigma*all_std, color = c, alpha = .5)
            
        ax.grid(ls = '-', lw = .5) 
        
        if runtime:
            ax.set_xlabel("Runtime [sec]", fontsize = 12)
        else:
            ax.set_xlabel("Epoch", fontsize = 12)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize = 12)
            
        if log_scale:
            ax.set_yscale('log')
            
        return