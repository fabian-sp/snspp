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
    A container object to store experiment results. Structure of ``self.results``
    
    self.results[<solver>][<k>] is a dictionary with keys
    
    - runtime (per iteration/epoch)
    - objective
    - number of (gradient) evaluations
    
    Additional info can be added via ``self.store_by_key``, for measurements not directly relate to optimization, e.g. test error or out of sample accuracy per iterate.
    
    """
    def __init__(self, name = ''):
        """
        """
        
        self.name = name
        self.solvers = list()
        self.params = dict()
        self.results = dict()
        
        self.psi_star = None
        
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
        """
        stores additional results, e.g. test set evaluations
        
        res: dict with resulty, e.g. key = 'test_loss'
        label: solver to which the results belong, needs to match the ones from self.store()
        k: run index (for multiple runs)
        """
        for key, val in res.items():
            assert key not in self.results[label][k].keys()
            self.results[label][k][key] = val
            
        return
    
    def save_to_disk(self, path = '', path_suffix = ''):   
        
        to_save = dict()
        to_save['psi_star'] = self.psi_star
        to_save['params'] = self.params
        to_save['results'] = self.results
        
        
        np.save(path + self.name + path_suffix +  '.npy', to_save)
        return
    
    def load_from_disk(self, path = '', path_suffix = ''):           
        from_save = np.load(path + self.name + path_suffix +  '.npy', allow_pickle = True)[()]
        
        self.results = from_save['results']
        self.psi_star = from_save['psi_star']
        self.params = from_save['params']
        self.solvers = self.results.keys()
        return

#########################################################################
#########################################################################
#########################################################################
## PLOTTING
#########################################################################
#########################################################################
#########################################################################

    
    def plot_objective(self, ax = None, runtime = True, median = False, markersize = 3, markevery_dict = dict(), ls = '-', lw = 0.4, psi_star = 0, log_scale = False, sigma = 0):
 
      
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
                if s == 'snspp':
                    # floor to only count grad evluations
                    all_xax = np.floor(np.vstack([this_res[k]["evaluations"] for k in range(20)]).mean(axis=0))
                    all_xax = np.insert(all_xax[1::10], 0, all_xax[0]).cumsum()
                    y = np.insert(y[1::10], 0, y[0])
                else:
                    all_xax = np.vstack([this_res[k]["evaluations"] for k in range(K)]).mean(axis=0).cumsum()

                
            try:
                c = color_dict[s]
                marker = marker_dict[s]
            except:
                c = color_dict["default"]
                marker = marker_dict["default"]
                
            mk_evy = markevery_dict.get(s, 1)
            
            ax.plot(all_xax, y, marker = marker, ls = ls, lw = lw, markersize = markersize, markevery = mk_evy, color = c, label = s)
            
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

    def plot_error(self, error_key = '', ax = None, runtime = True, median = True, markersize = 3, markevery_dict = dict(), ls = '-', lw = 0.4, log_scale = False, sigma = 0, ylabel = None):
        
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
            
            mk_evy = markevery_dict.get(s, 1)
            ax.plot(all_xax, y, marker = marker, ls = ls, lw = lw, markersize = markersize, markevery = mk_evy, color = c, label = s)
            
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